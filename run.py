from src.main.text_detector import CTPN
from src.utils.detection_utils.detection import show_img
import cv2
import os

net = CTPN(debug=True)
net.detect_text(images_path = 'data/demo/')