# Date : 2024.01.24
# Writer : justin 
import streamlit as st
import os
# Dagan
import torch
import torch.nn.functional as F
from skimage import img_as_ubyte
import argparse
import imageio
from skimage.transform import resize
from scipy.spatial import ConvexHull
from tqdm import tqdm
import numpy as np
import modules.generator as G
import modules.keypoint_detector as KPD
import yaml
from collections import OrderedDict
import depth
from stqdm import stqdm
# Real-ESRGAN
import cv2
import glob
import mimetypes
import shutil
import subprocess
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from os import path as osp
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
#GFPGAN
from basicsr.utils import imwrite
from gfpgan import GFPGANer
import time

try:
    import ffmpeg
except ImportError:
    import pip
    pip.main(['install', '--user', 'ffmpeg-python'])
    import ffmpeg


st.set_page_config(layout="wide")
st.markdown("""<style>.big-font {font-size:50px !important;}</style>""", unsafe_allow_html=True)
st.markdown('<p class="big-font">LastHouse PhotoAnimation GUI</p>', unsafe_allow_html=True)

#parser = argparse.ArgumentParser(description='Test DaGAN on your own images')
#parser.add_argument('--source_image', default='./temp/source.png', type=str, help='Directory of input source image')
#parser.add_argument('--driving_video', default='./temp/driving.mp4', type=str, help='Directory for driving video')
#parser.add_argument('--output', default='./temp/result.mp4', type=str, help='Directory for driving video')
#args = parser.parse_args()

def DeleteAllFile(filePath):
    if os.path.exists(filePath):
        for file in os.scandir(filePath):
            os.remove(file.path)
        return 'Remove All File'
    else:
        return 'Directory Not Found'


def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])
    return kp_new

def find_best_frame(source, driving, cpu=False):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0

    st_title = st.empty()
    st_progress_bar = st.empty()
    
    class tqdm_vi:
        def __init__(self, iterable, title=""):
            if title:
                st_title.write(title)
            self.prog_bar = st_progress_bar.progress(0)
            self.iterable = iterable
            self.length = len(iterable)
            self.i = 0
        
        def __iter__(self):
            for obj in self.iterable:
                yield obj
                self.i += 1
                current_prog = self.i / self.length
                self.prog_bar.progress(current_prog)

    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num

def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
    sources = []
    drivings = []
    with torch.no_grad():
        predictions = []
        depth_gray = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        if not cpu:
            source = source.cuda()
            driving = driving.cuda()
        outputs = depth_decoder(depth_encoder(source))
        depth_source = outputs[("disp", 0)]

        outputs = depth_decoder(depth_encoder(driving[:, :, 0]))
        depth_driving = outputs[("disp", 0)]
        source_kp = torch.cat((source,depth_source),1)
        driving_kp = torch.cat((driving[:, :, 0],depth_driving),1)
       
        kp_source = kp_detector(source_kp)
        kp_driving_initial = kp_detector(driving_kp) 

        # kp_source = kp_detector(source)
        # kp_driving_initial = kp_detector(driving[:, :, 0])
        # stqdm    
        st_title = st.empty()
        st_progress_bar = st.empty()
    
        class tqdm_vi:
            def __init__(self, iterable, title=""):
                if title:
                    st_title.write(title)
                self.prog_bar = st_progress_bar.progress(0)
                self.iterable = iterable
                self.length = len(iterable)
                self.i = 0
        
            def __iter__(self):
                for obj in self.iterable:
                    yield obj
                    self.i += 1
                    current_prog = self.i / self.length
                    self.prog_bar.progress(current_prog)

        for frame_idx in tqdm_vi(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            drivenamevi = str(frame_idx) 
            st_title.write(drivenamevi)
            if not cpu:
                driving_frame = driving_frame.cuda()
            outputs = depth_decoder(depth_encoder(driving_frame))
            depth_map = outputs[("disp", 0)]

            gray_driving = np.transpose(depth_map.data.cpu().numpy(), [0, 2, 3, 1])[0]
            gray_driving = 1-gray_driving/np.max(gray_driving)

            frame = torch.cat((driving_frame,depth_map),1)
            kp_driving = kp_detector(frame)

            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm,source_depth = depth_source, driving_depth = depth_map)

            drivings.append(np.transpose(driving_frame.data.cpu().numpy(), [0, 2, 3, 1])[0])
            sources.append(np.transpose(source.data.cpu().numpy(), [0, 2, 3, 1])[0])
            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
            depth_gray.append(gray_driving)
        
        st_title.empty()
        st_progress_bar.empty()
    return sources, drivings, predictions,depth_gray


with open("config/vox-adv-256.yaml") as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
    
generator = G.SPADEDepthAwareGenerator(**config['model_params']['generator_params'],**config['model_params']['common_params'])
config['model_params']['common_params']['num_channels'] = 4
kp_detector = KPD.KPDetector(**config['model_params']['kp_detector_params'],**config['model_params']['common_params'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu = False if torch.cuda.is_available() else True

g_checkpoint = torch.load("generator.pt", map_location=device)
kp_checkpoint = torch.load("kp_detector.pt", map_location=device)

ckp_generator = OrderedDict((k.replace('module.',''),v) for k,v in g_checkpoint.items())
generator.load_state_dict(ckp_generator)
ckp_kp_detector = OrderedDict((k.replace('module.',''),v) for k,v in kp_checkpoint.items())
kp_detector.load_state_dict(ckp_kp_detector)

depth_encoder = depth.ResnetEncoder(18, False)
depth_decoder = depth.DepthDecoder(num_ch_enc=depth_encoder.num_ch_enc, scales=range(4))
loaded_dict_enc = torch.load('encoder.pth',map_location=device)
loaded_dict_dec = torch.load('depth.pth',map_location=device)
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in depth_encoder.state_dict()}
depth_encoder.load_state_dict(filtered_dict_enc)
ckp_depth_decoder= {k: v for k, v in loaded_dict_dec.items() if k in depth_decoder.state_dict()}
depth_decoder.load_state_dict(ckp_depth_decoder)
depth_encoder.eval()
depth_decoder.eval()
    
# device = torch.device('cpu')
# stx()

generator = generator.to(device)
kp_detector = kp_detector.to(device)
depth_encoder = depth_encoder.to(device)
depth_decoder = depth_decoder.to(device)

generator.eval()
kp_detector.eval()
depth_encoder.eval()
depth_decoder.eval()

img_multiple_of = 8

def dagan_process(Import_file_name, f, driving_file_path):
    parser = argparse.ArgumentParser(description='Test DaGAN on your own images')
    parser.add_argument('--source_image', default='./temp/source.png', type=str, help='Directory of input source image')
    #parser.add_argument('--driving_video', default='./temp/driving.mp4', type=str, help='Directory for driving video')
    parser.add_argument('--driving_video', default=driving_file_path, type=str, help='Directory for driving video')
    #parser.add_argument('--driving_video', default='./drive_video/driving.mp4', type=str, help='Directory for driving video')
    parser.add_argument('--output', default='./temp/result.mp4', type=str, help='Directory for driving video')
    args = parser.parse_args()
    
    with torch.inference_mode():
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
        
        file_path = "Import_image"+"/"+str(Import_file_name)
        source_image = imageio.imread(file_path)
        reader = imageio.get_reader(args.driving_video)
        fps = reader.get_meta_data()['fps']
        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        
        reader.close()
        source_image = resize(source_image, (256, 256))[..., :3]
        driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
        i = find_best_frame(source_image, driving_video,cpu)
        print ("Best frame: " + str(i))
        driving_forward = driving_video[i:]
        driving_backward = driving_video[:(i+1)][::-1]
        sources_forward, drivings_forward, predictions_forward,depth_forward = make_animation(source_image, driving_forward, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=cpu)
        sources_backward, drivings_backward, predictions_backward,depth_backward = make_animation(source_image, driving_backward, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=cpu)
        predictions = predictions_backward[::-1] + predictions_forward[1:]
        sources = sources_backward[::-1] + sources_forward[1:]
        drivings = drivings_backward[::-1] + drivings_forward[1:]
        depth_gray = depth_backward[::-1] + depth_forward[1:]

        imageio.mimsave(args.output, [np.concatenate((img_as_ubyte(s),img_as_ubyte(d),img_as_ubyte(p)),1) for (s,d,p) in zip(sources, drivings, predictions)], fps=fps)
        imageio.mimsave("gray.mp4", depth_gray, fps=fps)
        # merge the gray video
        animation = np.array(imageio.mimread(args.output,memtest=False))
        gray = np.array(imageio.mimread("gray.mp4",memtest=False))
        src_dst = animation[:,:,:512,:]
        animate = animation[:,:,512:,:]
        #imageio.mimsave(animate.output)
        # 0 , 1 , 2    
        merge = np.concatenate((src_dst,gray,animate),2)                
        imageio.mimsave(args.output, merge, fps=fps)
        
        mpegcmd = "ffmpeg -i temp/result.mp4 -filter:v \"crop=243:248:781:0, scale=512:512\" -c:a copy "+"output_video/out_"+str(f)+".mp4" 
        #print(mpegcmd)
        os.system(mpegcmd)
        #print(f"\nRestored images are saved at {out_dir}")

# Real-ESRGAN
def get_video_meta_info(video_path):
    ret = {}
    probe = ffmpeg.probe(video_path)
    video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
    has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
    ret['width'] = video_streams[0]['width']
    ret['height'] = video_streams[0]['height']
    ret['fps'] = eval(video_streams[0]['avg_frame_rate'])
    ret['audio'] = ffmpeg.input(video_path).audio if has_audio else None
    ret['nb_frames'] = int(video_streams[0]['nb_frames'])
    return ret

def get_sub_video(args, num_process, process_idx):
    if num_process == 1:
        return args.input
    meta = get_video_meta_info(args.input)
    duration = int(meta['nb_frames'] / meta['fps'])
    part_time = duration // num_process
    print(f'duration: {duration}, part_time: {part_time}')
    os.makedirs(osp.join(args.output, f'{args.video_name}_inp_tmp_videos'), exist_ok=True)
    out_path = osp.join(args.output, f'{args.video_name}_inp_tmp_videos', f'{process_idx:03d}.mp4')
    cmd = [
        args.ffmpeg_bin, f'-i {args.input}', '-ss', f'{part_time * process_idx}',
        f'-to {part_time * (process_idx + 1)}' if process_idx != num_process - 1 else '', '-async 1', out_path, '-y'
    ]
    print(' '.join(cmd))
    subprocess.call(' '.join(cmd), shell=True)
    return out_path

class Reader:
    def __init__(self, args, total_workers=1, worker_idx=0):
        self.args = args
        input_type = mimetypes.guess_type(args.input)[0]
        self.input_type = 'folder' if input_type is None else input_type
        self.paths = []  # for image&folder type
        self.audio = None
        self.input_fps = None
        
        if self.input_type.startswith('video'):
            video_path = get_sub_video(args, total_workers, worker_idx)
            self.stream_reader = (
                ffmpeg.input(video_path).output('pipe:', format='rawvideo', pix_fmt='bgr24',
                                                loglevel='error').run_async(
                                                    pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))
            meta = get_video_meta_info(video_path)
            self.width = meta['width']
            self.height = meta['height']
            self.input_fps = meta['fps']
            self.audio = meta['audio']
            self.nb_frames = meta['nb_frames']

        else:
            if self.input_type.startswith('image'):
                self.paths = [args.input]
            else:
                paths = sorted(glob.glob(os.path.join(args.input, '*')))
                tot_frames = len(paths)
                num_frame_per_worker = tot_frames // total_workers + (1 if tot_frames % total_workers else 0)
                self.paths = paths[num_frame_per_worker * worker_idx:num_frame_per_worker * (worker_idx + 1)]

            self.nb_frames = len(self.paths)
            assert self.nb_frames > 0, 'empty folder'
            from PIL import Image
            tmp_img = Image.open(self.paths[0])
            self.width, self.height = tmp_img.size
        self.idx = 0
    
    def get_resolution(self):
        return self.height, self.width

    def get_fps(self):
        if self.args.fps is not None:
            return self.args.fps
        elif self.input_fps is not None:
            return self.input_fps
        return 24

    def get_audio(self):
        return self.audio

    def __len__(self):
        return self.nb_frames

    def get_frame_from_stream(self):
        img_bytes = self.stream_reader.stdout.read(self.width * self.height * 3)  # 3 bytes for one pixel
        if not img_bytes:
            return None
        img = np.frombuffer(img_bytes, np.uint8).reshape([self.height, self.width, 3])
        return img

    def get_frame_from_list(self):
        if self.idx >= self.nb_frames:
            return None
        img = cv2.imread(self.paths[self.idx])
        self.idx += 1
        return img

    def get_frame(self):
        if self.input_type.startswith('video'):
            return self.get_frame_from_stream()
        else:
            return self.get_frame_from_list()

    def close(self):
        if self.input_type.startswith('video'):
            self.stream_reader.stdin.close()
            self.stream_reader.wait()

class Writer:
    def __init__(self, args, audio, height, width, video_save_path, fps):
        out_width, out_height = int(width * args.outscale), int(height * args.outscale)
        if out_height > 2160:
            print('You are generating video that is larger than 4K, which will be very slow due to IO speed.',
                  'We highly recommend to decrease the outscale(aka, -s).')

        if audio is not None:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_width}x{out_height}',
                             framerate=fps).output(
                                 audio,
                                 video_save_path,
                                 pix_fmt='yuv420p',
                                 vcodec='libx264',
                                 loglevel='error',
                                 acodec='copy').overwrite_output().run_async(
                                     pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))
        else:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_width}x{out_height}',
                             framerate=fps).output(
                                 video_save_path, pix_fmt='yuv420p', vcodec='libx264',
                                 loglevel='error').overwrite_output().run_async(
                                     pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))
    
    def write_frame(self, frame):
        frame = frame.astype(np.uint8).tobytes()
        self.stream_writer.stdin.write(frame)

    def close(self):
        self.stream_writer.stdin.close()
        self.stream_writer.wait()

def inference_video(args, video_save_path,Upscale_video_check,fix_face_check,device=None, total_workers=1, worker_idx=0):
    print(Upscale_video_check,fix_face_check)
    # ---------------------- determine models according to model names ---------------------- #
    args.model_name = args.model_name.split('.pth')[0]
    if args.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif args.model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif args.model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif args.model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif args.model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif args.model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]

    # ---------------------- determine model paths ---------------------- #
    model_path = os.path.join('weights', args.model_name + '.pth')
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        for url in file_url:
            # model_path will be updated
            model_path = load_file_from_url(
                url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    # use dni to control the denoise strength
    dni_weight = None
    if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        device=device,
    )

    if 'anime' in args.model_name and args.face_enhance:
        print('face_enhance is not supported in anime models, we turned this option off for you. '
              'if you insist on turning it on, please manually comment the relevant lines of code.')
        args.face_enhance = False

    #if args.face_enhance:  # Use GFPGAN for face enhancement
        #from gfpgan import GFPGANer
        #face_enhancer = GFPGANer(
            #model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            #upscale=args.outscale,
            #arch='clean',
            #channel_multiplier=2,
            #bg_upsampler=upsampler)  # TODO support custom device
    #else:
        #face_enhancer = None
    
    from gfpgan import GFPGANer
    face_enhancer = GFPGANer(
        model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
        upscale=args.outscale,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=upsampler)  # TODO support custom device

    face_enhancer1 = GFPGANer(
        model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
        upscale=args.outscale,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None)  # TODO support custom device
    
    reader = Reader(args, total_workers, worker_idx)
    audio = reader.get_audio()
    height, width = reader.get_resolution()
    fps = reader.get_fps()
    writer = Writer(args, audio, height, width, video_save_path, fps)
    # stqdm
    
    st_title = st.empty()
    st_progress_bar = st.empty()
    
    class tqdm_vi:
        def __init__(self, iterable, title=""):
            if title:
                st_title.write(title)
            self.prog_bar = st_progress_bar.progress(0)
            self.iterable = iterable
            self.length = len(iterable)
            self.i = 0
    
        def __iter__(self):
            for obj in self.iterable:
                yield obj
                self.i += 1
                current_prog = self.i / self.length
                self.prog_bar.progress(current_prog)
    
    pbar = tqdm(total=len(reader), unit='frame', desc='inference')
    with st.spinner('Wait for it...'):
        while True:
            img = reader.get_frame()
            if img is None:
                break
            try:
                if Upscale_video_check == True and fix_face_check == False:
                    output, _ = upsampler.enhance(img, outscale=args.outscale)
                if Upscale_video_check == False and fix_face_check == True:
                    _, _, output = face_enhancer1.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
                if Upscale_video_check == True and fix_face_check == True:
                    _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
                if Upscale_video_check == False and fix_face_check == False:
                    print("None")                
                #if args.face_enhance:
                    #_, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
                #else:
                    #output, _ = upsampler.enhance(img, outscale=args.outscale)
                    #_, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            except RuntimeError as error:
                print('Error', error)
                print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
            else:
                writer.write_frame(output)
            torch.cuda.synchronize(device)
            pbar.update(1)
        reader.close()
        writer.close()
        st.success('Done!')

def run(args,Upscale_video_check,fix_face_check):
    args.video_name = osp.splitext(os.path.basename(args.input))[0]
    video_save_path = osp.join(args.output, f'{args.video_name}_{args.suffix}.mp4')

    if args.extract_frame_first:
        tmp_frames_folder = osp.join(args.output, f'{args.video_name}_inp_tmp_frames')
        os.makedirs(tmp_frames_folder, exist_ok=True)
        os.system(f'ffmpeg -i {args.input} -qscale:v 1 -qmin 1 -qmax 1 -vsync 0  {tmp_frames_folder}/frame%08d.png')
        args.input = tmp_frames_folder

    #num_gpus = torch.cuda.device_count()
    num_gpus = 1
    #num_process = num_gpus * args.num_process_per_gpu
    num_process = num_gpus * args.num_process_per_gpu if num_gpus > 0 else 1
    if num_process == 1:
        inference_video(args,video_save_path,Upscale_video_check,fix_face_check)
        return

    ctx = torch.multiprocessing.get_context('spawn')
    pool = ctx.Pool(num_process)
    os.makedirs(osp.join(args.output, f'{args.video_name}_out_tmp_videos'), exist_ok=True)
    
    st_title = st.empty()
    st_progress_bar = st.empty()
    
    class tqdm_vi:
        def __init__(self, iterable, title=""):
            if title:
                st_title.write(title)
            self.prog_bar = st_progress_bar.progress(0)
            self.iterable = iterable
            self.length = len(iterable)
            self.i = 0
    
        def __iter__(self):
            for obj in self.iterable:
                yield obj
                self.i += 1
                current_prog = self.i / self.length
                self.prog_bar.progress(current_prog)
    
    pbar = tqdm(total=num_process, unit='sub_video', desc='inference')
    
    for i in range(num_process):
        sub_video_save_path = osp.join(args.output, f'{args.video_name}_out_tmp_videos', f'{i:03d}.mp4')
        pool.apply_async(
            inference_video,
            args=(args, sub_video_save_path, torch.device(i % num_gpus), num_process, i),
            callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()

    # combine sub videos
    # prepare vidlist.txt
    with open(f'{args.output}/{args.video_name}_vidlist.txt', 'w') as f:
        for i in range(num_process):
            f.write(f'file \'{args.video_name}_out_tmp_videos/{i:03d}.mp4\'\n')

    cmd = [
        args.ffmpeg_bin, '-f', 'concat', '-safe', '0', '-i', f'{args.output}/{args.video_name}_vidlist.txt', '-c',
        'copy', f'{video_save_path}'
    ]
    print(' '.join(cmd))
    subprocess.call(cmd)
    shutil.rmtree(osp.join(args.output, f'{args.video_name}_out_tmp_videos'))
    if osp.exists(osp.join(args.output, f'{args.video_name}_inp_tmp_videos')):
        shutil.rmtree(osp.join(args.output, f'{args.video_name}_inp_tmp_videos'))
    os.remove(f'{args.output}/{args.video_name}_vidlist.txt')

def Real_ESRGAN_Video_Process(Import_file_name,f,Video_UpScale_check,fix_face_check,Resize_value,Real_ESRModel0,Denoise_value):
    """Inference demo for Real-ESRGAN.
    It mainly for restoring anime videos.
    """
    print(Import_file_name)
    print(f)
    
    input_path = "inputs_video"+"/"+str(Import_file_name)
    
    parser = argparse.ArgumentParser()
    #parser.add_argument('-i', '--input', type=str, default='inputs\out_5.mp4', help='Input video, image or folder')
    parser.add_argument('-i', '--input', type=str, default=input_path, help='Input video, image or folder')
    
    if Real_ESRModel0 == "RealESRGAN_x2plus":
        parser.add_argument(
            '-n',
            '--model_name',
            type=str,
            default='RealESRGAN_x2plus',
            help=('Model names: realesr-animevideov3 | RealESRGAN_x4plus_anime_6B | RealESRGAN_x4plus | RealESRNet_x4plus |'
                ' RealESRGAN_x2plus | realesr-general-x4v3'
                'Default:realesr-animevideov3'))
    
    if Real_ESRModel0 == "realesr-animevideov3":
        parser.add_argument(
            '-n',
            '--model_name',
            type=str,
            default='realesr-animevideov3',
            help=('Model names: realesr-animevideov3 | RealESRGAN_x4plus_anime_6B | RealESRGAN_x4plus | RealESRNet_x4plus |'
                ' RealESRGAN_x2plus | realesr-general-x4v3'
                'Default:realesr-animevideov3'))
    
    if Real_ESRModel0 == "RealESRGAN_x4plus_anime_6B":
        parser.add_argument(
            '-n',
            '--model_name',
            type=str,
            default='RealESRGAN_x4plus_anime_6B',
            help=('Model names: realesr-animevideov3 | RealESRGAN_x4plus_anime_6B | RealESRGAN_x4plus | RealESRNet_x4plus |'
                ' RealESRGAN_x2plus | realesr-general-x4v3'
                'Default:realesr-animevideov3'))
    
    if Real_ESRModel0 == "RealESRGAN_x4plus":
        parser.add_argument(
            '-n',
            '--model_name',
            type=str,
            default='RealESRGAN_x4plus',
            help=('Model names: realesr-animevideov3 | RealESRGAN_x4plus_anime_6B | RealESRGAN_x4plus | RealESRNet_x4plus |'
                ' RealESRGAN_x2plus | realesr-general-x4v3'
                'Default:realesr-animevideov3'))
    
    if Real_ESRModel0 == "RealESRNet_x4plus":
        parser.add_argument(
            '-n',
            '--model_name',
            type=str,
            default='RealESRNet_x4plus',
            help=('Model names: realesr-animevideov3 | RealESRGAN_x4plus_anime_6B | RealESRGAN_x4plus | RealESRNet_x4plus |'
                ' RealESRGAN_x2plus | realesr-general-x4v3'
                'Default:realesr-animevideov3'))
    
    if Real_ESRModel0 == "realesr-general-x4v3":
        parser.add_argument(
            '-n',
            '--model_name',
            type=str,
            default='realesr-general-x4v3',
            help=('Model names: realesr-animevideov3 | RealESRGAN_x4plus_anime_6B | RealESRGAN_x4plus | RealESRNet_x4plus |'
                ' RealESRGAN_x2plus | realesr-general-x4v3'
                'Default:realesr-animevideov3'))
    
    parser.add_argument('-o', '--output', type=str, default='save_video/fixed', help='Output folder')
    parser.add_argument(
        '-dn',
        '--denoise_strength',
        type=float,
        default=float(Denoise_value),
        help=('Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability. '
              'Only used for the realesr-general-x4v3 model'))
    
    now = time
    save_recent_time = now.strftime('%Y-%m-%d_%H%M%S')
    
    parser.add_argument('-s', '--outscale', type=float, default=int(Resize_value), help='The final upsampling scale of the image')
    #parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored video')
    parser.add_argument('--suffix', type=str, default=str(save_recent_time), help='Suffix of the restored video')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument(
        '--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
    parser.add_argument('--fps', type=float, default=None, help='FPS of the output video')
    parser.add_argument('--ffmpeg_bin', type=str, default='ffmpeg', help='The path to ffmpeg')
    parser.add_argument('--extract_frame_first', action='store_true')
    parser.add_argument('--num_process_per_gpu', type=int, default=1)

    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='realesrgan',
        help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    args = parser.parse_args()

    args.input = args.input.rstrip('/').rstrip('\\')
    os.makedirs(args.output, exist_ok=True)

    if mimetypes.guess_type(args.input)[0] is not None and mimetypes.guess_type(args.input)[0].startswith('video'):
        is_video = True
    else:
        is_video = False

    if is_video and args.input.endswith('.flv'):
        mp4_path = args.input.replace('.flv', '.mp4')
        os.system(f'ffmpeg -i {args.input} -codec copy {mp4_path}')
        args.input = mp4_path

    if args.extract_frame_first and not is_video:
        args.extract_frame_first = False

    run(args,Video_UpScale_check,fix_face_check)
    if args.extract_frame_first:
        tmp_frames_folder = osp.join(args.output, f'{args.video_name}_inp_tmp_frames')
        shutil.rmtree(tmp_frames_folder)

def main():
    Image_input = st.text_input('Input Photo Directory','')
    Image_output = st.text_input('Output Video Directory','')
    drive_video = st.text_input('Resource Video File','')
    Upscale_check = st.checkbox('Video UpScaling')
    if Upscale_check == True:
        with st.container():
            Resize_value = st.slider("Resize", 0,8,1)
            Real_ESRModel0 = st.selectbox(
                'Model',
                ('RealESRGAN_x4plus','RealESRGAN_x2plus','realesr-general-x4v3','RealESRNet_x4plus','RealESRGAN_x4plus_anime_6B')
            )
            Denoise_value = st.slider("Denoising strength", 0.0,10.0,0.1)
        st.markdown("""<style>div[class="st-emotion-cache-0 e1f1d6gn0"]{padding-left:20px;}</style>""",unsafe_allow_html=True)
    face_fix_check = st.checkbox('Fix Face')
    st.markdown("Video Format")
    
    facere = st.radio(label = '', options=['MP4','Sequence'])
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    
    copycmd = "copy"
    import_file_line = Image_input+"\*"
    import_file_line2 = r"C:\Users\emine\Documents\work\LastHouse_PhotoAnimation\Import_image"
    input_file_cmd = copycmd+" "+import_file_line+" "+import_file_line2    
    
    if st.button('Generate'):
        os.system(input_file_cmd)
        path = import_file_line2
        dir_list = os.listdir(path)
        
        if facere == 'MP4':
            for f in range(int(len(dir_list))):
                import_file_name = dir_list[f]
                print(import_file_name)
                print(drive_video)
                dagan_process(import_file_name,f,drive_video)
                #dagan_process(import_file_name,f)
            
                import_file_line3 = r"C:\Users\emine\Documents\work\LastHouse_PhotoAnimation\output_video"
                import_file_line4 = import_file_line3+"\*"
                import_file_line5 = r"C:\Users\emine\Documents\work\LastHouse_PhotoAnimation\inputs_video"
                input_file_cmd1 = copycmd+" "+import_file_line4+" "+import_file_line5
                os.system(input_file_cmd1)

                dir_list1 = os.listdir(import_file_line5)
                print(dir_list1[f])
                Real_ESRGAN_Video_Process(dir_list1[f],f,Upscale_check,face_fix_check,Resize_value,Real_ESRModel0,Denoise_value)

                import_file_line6 = r"C:\Users\emine\Documents\work\LastHouse_PhotoAnimation\save_video\fixed"
                import_file_line7 = import_file_line6+"\*"
                import_file_line8 = Image_output
                input_file_cmd2 = copycmd+" "+import_file_line7+" "+import_file_line8
                os.system(input_file_cmd2)
                
                dir_list2 = os.listdir(import_file_line6)
                Importimgpath = import_file_line2 +"/"+ import_file_name
                os.remove(Importimgpath)
                #Outputvideopath = import_file_line3 +"/"+ dir_list1[f]
                #os.remove(Outputvideopath)
                #Inputsvideopath = import_file_line5 +"/"+ dir_list1[f]
                #os.remove(Inputsvideopath)
                #finalfixvideopath = import_file_line6 +"/"+ dir_list2[f] 
                #os.remove(finalfixvideopath)
            DeleteAllFile(import_file_line3)
            DeleteAllFile(import_file_line5)
            DeleteAllFile(import_file_line6)
                
                # frame version . 2024.01.19 justin
                #import_file_line6 = r"C:\Users\emine\Documents\work\LastHouse_PhotoAnimation\save_video\fixed"
                #dir_list2 = os.listdir(import_file_line6)
                #ffmpegcmd = "ffmpeg -i "+"save_video"+"/"+"fixed"+"/"+dir_list2[f]+" frames/out%03d.png"
                #os.system(ffmpegcmd)
                #import_file_line7 = r"C:\Users\emine\Documents\work\LastHouse_PhotoAnimation\frames"
                #import_file_line8 = import_file_line7+"\*"
                #import_file_line9 = r"C:\Users\emine\Documents\work\LastHouse_PhotoAnimation\inputs\whole_imgs"
                #input_file_cmd2 = copycmd+" "+import_file_line8+" "+import_file_line9
                #os.system(input_file_cmd2)
                #face_fix_process()
                #ffmpegcmd1 = "ffmpeg -framerate "+str(int(20))+" "+"-i"+" "+"results/restored_imgs/out%03d.png"+" "+"-c:v"+" "+"libx264"+" "+"-pix_fmt"+" "+"yuv420p"+" "+"out_high_"+str(f)+".mp4"
                #os.system(ffmpegcmd1)
                #print(ffmpegcmd1)
                #os.system(ffmpegcmd1)
        if facere == 'Sequence':
            for f in range(int(len(dir_list))):
                # sequence file 
                import_file_name = dir_list[f]
                print(import_file_name)
                print(drive_video)
                dagan_process(import_file_name,f,drive_video)
                #dagan_process(import_file_name,f)
            
                import_file_line3 = r"C:\Users\emine\Documents\work\LastHouse_PhotoAnimation\output_video"
                import_file_line4 = import_file_line3+"\*"
                import_file_line5 = r"C:\Users\emine\Documents\work\LastHouse_PhotoAnimation\inputs_video"
                input_file_cmd1 = copycmd+" "+import_file_line4+" "+import_file_line5
                os.system(input_file_cmd1)

                dir_list1 = os.listdir(import_file_line5)
                print(dir_list1[f])
                Real_ESRGAN_Video_Process(dir_list1[f],f,Upscale_check,face_fix_check,Resize_value,Real_ESRModel0,Denoise_value)

                import_file_line6 = r"C:\Users\emine\Documents\work\LastHouse_PhotoAnimation\save_video\fixed"
                import_file_line7 = import_file_line6+"\*"
                import_file_line8 = Image_output
                input_file_cmd2 = copycmd+" "+import_file_line7+" "+import_file_line8
                os.system(input_file_cmd2)    
                
                video_file_out = r"C:\Users\emine\Documents\work\LastHouse_PhotoAnimation\save_video\fixed"
                dir_list3 = os.listdir(video_file_out)
                ffmpegcmd = "ffmpeg -i "+"save_video"+"/"+"fixed"+"/"+ dir_list3[f]+" frames/out%03d.png"
                os.system(ffmpegcmd)
                now = time
                recet_time = now.strftime('%Y-%m-%d_%H%M%S')
                print(recet_time)
                makepath = recet_time
                os.makedirs(makepath)
                import_file_line9 = r"C:\Users\emine\Documents\work\LastHouse_PhotoAnimation\frames"
                import_file_line10 = import_file_line9+"\*"
                import_file_line11 = makepath
                import_file_cmd3 = copycmd+" "+import_file_line10+" "+import_file_line11
                os.system(import_file_cmd3)
                
                #dir_list2 = os.listdir(import_file_line6)
                Importimgpath = import_file_line2 +"/"+ import_file_name
                os.remove(Importimgpath)
            
            DeleteAllFile(import_file_line3)
            DeleteAllFile(import_file_line5)
            DeleteAllFile(import_file_line6)
            DeleteAllFile(import_file_line9)  

if __name__ == "__main__":
    main()