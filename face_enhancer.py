from pathlib import Path
import os
import insightface
from insightface.app import FaceAnalysis
import gfpgan
from utilities.typing import Frame, Face
from typing import Any, List, Callable
import threading
import cv2
from face_analyser import get_one_face, get_many_faces_sorted
import globals
from utilities.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_ENHANCER = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
NAME = 'FACE-ENHANCER'

def get_face_enhancer() -> Any:
    global FACE_ENHANCER

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = resolve_relative_path('../models/GFPGANv1.4.pth')
            # todo: set models path https://github.com/TencentARC/GFPGAN/issues/399
            FACE_ENHANCER = gfpgan.GFPGANer(model_path=model_path, upscale=1) # type: ignore[attr-defined]
    return FACE_ENHANCER
    
def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://huggingface.co/henryruhs/roop/resolve/main/GFPGANv1.4.pth'])
    return True

def pre_start() -> bool:
    if not is_image(globals.target_path) and not is_video(globals.target_path):
        print('Select an image or video for target path.')
        #update_status('Select an image or video for target path.', NAME)
        return False
    return True

def process_image(source_path: str, target_path: str, output_path: str,swap_face_index: int) -> None:
    target_frame = cv2.imread(target_path)
    result = process_frame(None, target_frame)
    cv2.imwrite(output_path, result)

def process_frame(source_face: Face, temp_frame: Frame) -> Frame:
    target_face = get_one_face(temp_frame)
    if target_face:
        temp_frame = enhance_face(temp_frame)
    return temp_frame


def enhance_face(temp_frame: Frame) -> Frame:
    with THREAD_SEMAPHORE:
        _, _, temp_frame = get_face_enhancer().enhance(
            temp_frame,
            paste_back=True
        )
    return temp_frame


def post_process() -> None:
    global FACE_ENHANCER

    FACE_ENHANCER = None