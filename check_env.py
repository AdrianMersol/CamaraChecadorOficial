# check_env.py
import sys, struct
print("Python:", sys.version)
print("bits:", struct.calcsize('P')*8)

import numpy as np
print("numpy:", np.__version__)

import cv2
print("opencv:", cv2.__version__)

import onnxruntime as ort
print("onnxruntime:", ort.__version__, "providers:", ort.get_available_providers())

import faiss
print("faiss:", faiss.__version__)

import insightface
from insightface.app import FaceAnalysis
print("insightface:", insightface.__version__)
print("OK")
