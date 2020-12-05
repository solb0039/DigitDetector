import cv2
import numpy as np
from collections import namedtuple

Bbox = namedtuple('Bbox', ['st_col', 'st_row', 'width', 'height'])

class RegionsOfInterest(object):
    """
    Identify regions of interest from an image and return
    candidate bounding boxes.
    """

    def __init__(self, img):
        self.bounding_box = []
        self.image = img
        self.gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    def group_rectangles(self):

        cand_rects = [*self.bounding_box, *self.hulls]

        rects, w = cv2.groupRectangles(cand_rects, 1, 0.5)

        rects = rects.tolist()

        rects = [Bbox(x[0], x[1], x[2], x[3]) for x in rects]

        # filter out boxes where width > height
        rects = [x for x in rects if x.width < x.height]

        # filter boxes where 2*width > height
        rects = [x for x in rects if x.height > 1.3 * x.width]

        # filter out boxes that are too tall and skinny
        rects = [x for x in rects if x.height / x.width < 4]

        self.bounding_box = rects


    def _show_bbox(self):
        """Plot images for debugging"""
        for box in  self.bounding_box:
            cv2.rectangle(self.image, (box[0], box[1]), (box[0]+box[2], box[1] + box[3]), (0,255,0))

        cv2.imshow('img', self.image)
        cv2.waitKey(0)


    def detect_regions(self, show=False):
        "Called method to find ROIs"

        mser = cv2.MSER_create()

        regions, bbox = mser.detectRegions(self.gray_image)

        self.bounding_box = bbox.tolist()

        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

        self.hulls = [cv2.boundingRect(hull) for hull in hulls]

        self.group_rectangles()

        if show:
            self._show_bbox()
