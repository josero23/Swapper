import datetime
import numpy as np
import os
import os.path as osp
import glob
import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import threading
import gfpgan
from utilities.typing import Frame, Face
from typing import Any, List, Callable
from pathlib import Path

assert insightface.__version__>='0.7'

FACE_ENHANCER = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()

def get_face_enhancer() -> Any:
    global FACE_ENHANCER

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            path = Path(__file__).parent.absolute()
            link_gfpgan_model=f'{path}\models\GFPGANv1.4.pth'
            model_path = (link_gfpgan_model)
            # todo: set models path https://github.com/TencentARC/GFPGAN/issues/399
            FACE_ENHANCER = gfpgan.GFPGANer(model_path=model_path, upscale=1) # type: ignore[attr-defined]
    return FACE_ENHANCER

def get_oneface(frame: Frame) -> Any:
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    face = app.get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    target_frame = cv2.imread(target_path)
    result = process_frame(None, target_frame)
    cv2.imwrite(output_path, result)
    print('restored.jpg saved')

def process_frame(source_face: Face, temp_frame: Frame) -> Frame:
    target_face = get_oneface(temp_frame)
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

if __name__ == '__main__':
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    path = Path(__file__).parent.absolute()
    link_inswapper_model=f'{path}\models\inswapper_128.onnx'
    swapper = insightface.model_zoo.get_model(link_inswapper_model)



    img = cv2.imread('pic.jpeg')
    img2= cv2.imread('face.jpg')
   # img = ins_get_image('')
    faces = app.get(img)
    faces = sorted(faces, key = lambda x : x.bbox[0])
    faces2 = app.get(img2)
    faces2 = sorted(faces2, key = lambda x : x.bbox[0])
    assert len(faces)==3
    source_face = faces2[0]
    res = img.copy()
    #for face in faces:
    res = swapper.get(res, faces[1], source_face, paste_back=True)
    cv2.imwrite("t1_swapped.jpg", res)
    res = []
    #for face in faces:
    _img, _ = swapper.get(img, faces[1], source_face, paste_back=False)
    res.append(_img)
    res = np.concatenate(res, axis=1)
    cv2.imwrite("1_swapped2.jpg", res)

    process_image('','t1_swapped.jpg','restored.jpg')