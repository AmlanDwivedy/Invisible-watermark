import cv2
from imwatermark import WatermarkEncoder, WatermarkDecoder
import os
import time

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def embed(img, wm, file_name):
    bgr = cv2.imread(img)

    encoder = WatermarkEncoder()
    encoder.set_watermark('bytes', wm.encode('utf-8'))
    bgr_encoded = encoder.encode(bgr, 'dwtDctSvd')

    # cv2.imwrite('in.jpg', bgr_encoded)
    cv2.imwrite(file_name, bgr_encoded)

def decode(img):
    bgr = cv2.imread(img)
    decoder = WatermarkDecoder('bytes', 136)
    watermark = decoder.decode(bgr, 'dwtDctSvd')
    # watermark = decoder.decode(bgr, 'rivaGan')
    
    print(watermark.decode('utf-8'))

if __name__ == "__main__":
    WatermarkEncoder.loadModel()
    image_file = "test.png"
    # embed(image_file, "te")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename  = "in_"+timestr+"__.jpg"
    print(filename)
    embed(img=image_file, wm="StableDiffusionV1", file_name=filename)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    decode(filename)
    # image_file = "in.jpg"
    # decode(image_file)