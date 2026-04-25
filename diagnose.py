import sys, os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"

print("1 - env set", flush=True)

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication
print("2 - PyQt5 imported", flush=True)

app = QApplication(sys.argv)
print("3 - QApplication created", flush=True)

from ela import convert_to_ela_image
print("4 - ela imported", flush=True)

from prediction import predict_result
print("5 - prediction imported", flush=True)

print("ALL OK", flush=True)
