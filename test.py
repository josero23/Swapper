import os
from pathlib import Path
from typing import List
import onnxruntime
import utilities.utilities
import core_process
import main
import subprocess
import face_swapper
import face_enhancer
import globals

def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]

def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())

def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]

globals.specific_face=False
#face_swapper.process_image('face.jpg','pic.jpg','restored.jpg',1)
#face_enhancer.process_image('','restored.jpg','restore.jpg',None)
globals.source_path = 'face.jpg'
globals.target_path = 'pic.jpg'
globals.output_path = 'restored.jpg'
globals.frame_processors = ['face_swapper','face_enhancer']
globals.headless = globals.source_path or globals.target_path or globals.output_path 
globals.execution_providers = decode_execution_providers(['cuda'])
#main.run()
#print(suggest_execution_providers())
#print(globals.execution_providers)
ort_session = onnxruntime.InferenceSession(None, providers=['CUDAExecutionProvider'])
print(ort_session.get_providers()) 
# ['CUDAExecutionProvider']
onnxruntime.get_device()
# GPU