import cv2
from imwatermark import WatermarkDecoder

bgr = cv2.imread('test_wm.png')

decoder = WatermarkDecoder('bytes', 32)
watermark = decoder.decode(bgr, 'dwtDct')
print(watermark)
print(watermark.decode('utf-8'))
