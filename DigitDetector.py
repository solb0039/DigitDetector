import torch
from torchvision import models
import cv2
from torchvision import transforms
import numpy as np


class DigitDetector(object):

    def __init__(self, weight_file, num_classes):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = models.vgg16(pretrained=False)
        self.model.classifier[6].out_features = num_classes
        self.output_val = None
        self.output = None

        if torch.cuda.is_available():
            map_location = lambda storage, loc: storage.cuda()
            self.model.load_state_dict(torch.load(weight_file, map_location=map_location))
        else:
            map_location = 'cpu'
            self.model.load_state_dict(torch.load(weight_file, map_location=map_location))

        self.model.eval()

        self.loader = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    def _classify_digit(self, img):

        image = self.loader(img).float()

        image_T = image.unsqueeze(0)

        with torch.no_grad():
            results = self.model(image_T)
            val, pred_dig = torch.max(results, 1)
            return [val, pred_dig]


    def _average_classification(self, results):

        vals = np.array([x[0].numpy() for x in results])

        digits = np.array([x[1].numpy() for x in results])

        if np.all((digits == digits[0])) or np.all(vals>20.):
            if int(digits[0]) == 10:
                # 10 is 0
                self.output = 0
            else:
                self.output = int(digits[0])
            self.output_val = np.max(vals)
        else:
            # Non-digit regions
            self.output = 11
            self.output_val = 0.


    def detect(self, img):
        """Generate Gaussian pyramid to scale candidate image up and down"""

        img_copy = img.copy()
        scl_up_results = []
        scl_up_results.append(self._classify_digit(img_copy))

        img_copy = cv2.pyrUp(img_copy)
        scl_up_results.append(self._classify_digit(img_copy))

        img_copy = img.copy()
        img_copy = cv2.pyrDown(img_copy)
        scl_up_results.append(self._classify_digit(img_copy))

        self._average_classification([*scl_up_results])
