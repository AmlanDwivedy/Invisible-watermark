import cv2
from imwatermark import WatermarkEncoder

bgr = cv2.imread('test.jpg')
wm = 'test'

encoder = WatermarkEncoder()
encoder.set_watermark('bytes', wm.encode('utf-8'))
bgr_encoded = encoder.encode(bgr, 'dwtDct')

cv2.imwrite('test_wm_1.png', bgr_encoded)
