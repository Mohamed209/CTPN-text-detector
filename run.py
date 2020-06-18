from main.text_detector import CTPN
from utils.detection_utils.detection import show_img
import cv2

net = CTPN(debug=True)
image = cv2.imread('enter image path here')
# return n numpy arrays as detected lines
_, lines = net.detect_text(image, to_lines=True)
for l in lines:
    show_img(l)
