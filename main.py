import numpy as np
import os
import sys
import os.path as osp
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import shutil
from utilities.typing import Frame, Face
from typing import Any, List, Callable
from pathlib import Path
import face_enhancer
import onnxruntime
assert insightface.__version__>='0.7'
import globals
import warnings
import signal
import argparse
from core_process import get_frame_processors_modules
from utilities.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]

def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())

def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]

def pre_check() -> bool:
    if sys.version_info < (3, 9):
        #update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        print('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    """
    if not shutil.which('ffmpeg'):
        update_status('ffmpeg is not installed.')
        return False
    """
    return True


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100))
    program.add_argument('-s', '--source', help='select an source image', dest='source_path')
    program.add_argument('-t', '--target', help='select an target image or video', dest='target_path')
    program.add_argument('-o', '--output', help='select output file or directory', dest='output_path')
    program.add_argument('--frame-processor', help='frame processors (choices: face_swapper, face_enhancer, ...)', dest='frame_processor', default=['face_swapper'], nargs='+')
  # program.add_argument('--keep-fps', help='keep original fps', dest='keep_fps', action='store_true', default=False)
  # program.add_argument('--keep-audio', help='keep original audio', dest='keep_audio', action='store_true', default=True)
  # program.add_argument('--keep-frames', help='keep temporary frames', dest='keep_frames', action='store_true', default=False)
    program.add_argument('--specific-face', help='process an specific face', dest='specific_face', action='store_true', default=False)
    program.add_argument('--target-face_index', help='index position of target face in image', dest='target_face_index', type=int, default=0)
  # program.add_argument('--video-encoder', help='adjust output video encoder', dest='video_encoder', default='libx264', choices=['libx264', 'libx265', 'libvpx-vp9'])
  # program.add_argument('--video-quality', help='adjust output video quality', dest='video_quality', type=int, default=18, choices=range(52), metavar='[0-51]')
  # program.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory', type=int, default=suggest_max_memory())
    program.add_argument('--execution-provider', help='available execution provider (choices: cpu, ...)', dest='execution_provider', default=['cpu'], choices=suggest_execution_providers(), nargs='+')
  # program.add_argument('--execution-threads', help='number of execution threads', dest='execution_threads', type=int, default=suggest_execution_threads())
  # program.add_argument('-v', '--version', action='version', version=f'{roop.metadata.name} {roop.metadata.version}')

    args = program.parse_args()

    globals.source_path = args.source_path
    globals.target_path = args.target_path
    globals.output_path = normalize_output_path(globals.source_path, globals.target_path, args.output_path)
    globals.frame_processors = args.frame_processor
    globals.headless = args.source_path or args.target_path or args.output_path
  # utilities.globals.keep_fps = args.keep_fps
  # utilities.globals.keep_audio = args.keep_audio
  # utilities.globals.keep_frames = args.keep_frames
    globals.specific_face = args.specific_face
    globals.target_face_index = args.target_face_index
  # utilities.globals.video_encoder = args.video_encoder
  # utilities.globals.video_quality = args.video_quality
  # utilities.globals.max_memory = args.max_memory
    globals.execution_providers = decode_execution_providers(args.execution_provider)
  # utilities.globals.execution_threads = args.execution_threads

def destroy() -> None:
    if globals.target_path:
        clean_temp(globals.target_path)
    quit()

"""
def release_resources() -> None:
    if 'CUDAExecutionProvider' in globals.execution_providers:
        torch.cuda.empty_cache()
"""
        
def run() -> None:
    parse_args()
    if not pre_check():
        return
    for frame_processor in get_frame_processors_modules(globals.frame_processors):
        if not frame_processor.pre_check():
            return
    #limit_resources()
    if globals.headless:
        start()
    """
    else:
        window = ui.init(start, destroy)
        window.mainloop()
    """

def start() -> None:
    for frame_processor in get_frame_processors_modules(globals.frame_processors):
        if not frame_processor.pre_start():
            return
    # process image to image
    if has_image_extension(globals.target_path):

        shutil.copy2(globals.target_path, globals.output_path)
        for frame_processor in get_frame_processors_modules(globals.frame_processors):
            #update_status('Progressing...', frame_processor.NAME)
            print('Progressing...',frame_processor.NAME)
            frame_processor.process_image(globals.source_path, globals.output_path, globals.output_path,globals.target_face_index)
            frame_processor.post_process()
            #release_resources()
        if is_image(globals.target_path):
            print('Processing to image succeed!')
            #update_status('Processing to image succeed!')
        else:
            print('Processing to image failed!')
            #update_status('Processing to image failed!')
        return

