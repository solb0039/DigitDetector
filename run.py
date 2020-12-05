import cv2
import numpy as np
import glob

from DigitDetector import *
from RegionsOfInterest import *

images = glob.glob('./graded_images/unannotated_images/*.png')

for img_path in images:

    img = cv2.imread(img_path)

    img_name = img_path.split('/')[-1]

    digit_detector = DigitDetector('./cnn_models/optimized_vgg16_net.pth', 11)

    roi = RegionsOfInterest(img)
    roi.detect_regions(show=False)
    bboxs = roi.bounding_box

    print(f'Processing image {img_name}')
    print(f'{len(bboxs)} bounding boxes found')

    for box in bboxs:
        ext_img = img[box.st_row:(box.st_row + box.height), box.st_col:(box.st_col + box.width)]

        digit_detector.detect(ext_img)

        if digit_detector.output_val > 11 and digit_detector.output != 11.0: # 11 is the non-digit class
            print(f'Found a number of {digit_detector.output} at location ({box.st_row},{box.st_col}) with confidence {digit_detector.output_val:.2f}')
            cv2.rectangle(img, (box.st_col, box.st_row), (box.st_col + box.width, box.st_row + box.height), (0, 255, 0), thickness=2)
            cv2.putText(img, str(digit_detector.output), (box.st_col, box.st_row),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)

    print('Finished with detections')
    cv2.imwrite(f'./graded_images/{img_name}', img)

print('Done')

