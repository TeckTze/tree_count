from dataclasses import dataclass
import cv2
import numpy as np
import scipy
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


@dataclass
class HSVThresholds:
    h_low: int
    h_high: int
    s_low: int
    s_high: int
    v_low: int
    v_high: int

@dataclass
class ContourThresholds:
    min_area: int

class HSVTreeCountDetector:

    @staticmethod
    def predict(img, hsv_thresh, cnt_thresh):
        """
        Predicts number of trees using HSV thresholding
        """

        # Original image
        imgOri = img.copy()

        # Get resize flag and factor
        resize_flag, resize_factor = get_resize_flag_factor(img, threshold = 4000000)
        
        if resize_flag:
            dim = (int(img.shape[1] * resize_factor), int(img.shape[0] * resize_factor))
            img = cv2.resize(img, (dim))

        # HSV implementation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Threshold using ranges
        min_hsv = np.array([hsv_thresh.h_low, hsv_thresh.s_low, hsv_thresh.v_low])
        max_hsv = np.array([hsv_thresh.h_high, hsv_thresh.s_high, hsv_thresh.v_high])
        thresh = cv2.inRange(hsv, min_hsv, max_hsv)

        # Median Blur to remove salt and pepper noise
        thresh = cv2.medianBlur(thresh, 3)

        # Morphology closing
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)), iterations = 3)

        # Get External contours
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # List of contour radius
        r_list = []

        # Create a filled contours image as empty
        img_filled_cnt = np.zeros_like(img)

        # First round of contouring
        for cnt in cnts:
            # Remove contour if it's small
            if cv2.contourArea(cnt) < cnt_thresh.min_area:
                continue

            # Append radius to list
            _, r = cv2.minEnclosingCircle(cnt)
            r_list.append(r)

            # Create fileld image with contours
            cv2.drawContours(img_filled_cnt, [cnt], 0, (255, 255, 255), -1)

        # Convert filled image to grayscale
        img_filled_cnt = cv2.cvtColor(img_filled_cnt, cv2.COLOR_BGR2GRAY)

        # 2.0 Calculate Euclidean Distance from filled contours
        distance = scipy.ndimage.distance_transform_edt(img_filled_cnt)
	
        # Handle case where no contours is shortlisted
        if len(r_list) == 0:
            return 0, imgOri

        # 2.1 Find local maximum
        min_dist_thresh = int(np.median(r_list)) # Calculate minimum distance # 20220914 - change mean to median
        coords = peak_local_max(distance, footprint = np.ones((3,3)), min_distance = min_dist_thresh, labels = img_filled_cnt)

        # 2.2 Fill local maximum into mask
        mask = np.zeros(distance.shape, dtype = bool)
        mask[tuple(coords.T)] = True

        # 2.3 Perform watershed - Generate labels raster
        markers, _ = scipy.ndimage.label(mask)
        labels = watershed(-distance, markers, mask = img_filled_cnt)

        # 2.4 Generate tree count
        tree_count = 0
        box_list = []
        for lab in np.unique(labels):
            if lab == 0:
                continue
            mask = np.zeros_like(labels, dtype = np.uint8)
            mask[labels == lab] = 255
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            c = max(cnts, key = cv2.contourArea)

            if cv2.contourArea(c) > (np.pi * (np.mean(r_list) / 2) ** 2):
                x, y, w, h = cv2.boundingRect(c)
                box_list.append((x, y, w, h))
                # cv2.rectangle(img, (x,y), (x + w, y + h), (0, 0, 255), 3) # Commented 20220914 - Move drawing outside the loop
                tree_count += 1

        # cv2.putText(img, f"Tree Count: {tree_count}",
        #             (10,60), 0, 2, (0, 0, 255), 5)

        for x, y, w, h in box_list:
            if resize_flag:
                x = int(x/ resize_factor)
                y = int(y/ resize_factor)
                w = int(w/ resize_factor)
                h = int(h/ resize_factor)
                x_bottom = min(x + w, imgOri.shape[1])
                y_bottom = min(y + h, imgOri.shape[0])
            
                cv2.rectangle(imgOri, (x, y), (x_bottom, y_bottom), (0, 0, 255), 3)
            else:
                cv2.rectangle(imgOri, (x, y), (x + w, y + h), (0, 0, 255), 3)
        
        # Resize output image - Added 20221004 - Reduce API response time
        if resize_flag:
            imgOri = cv2.resize(imgOri, (dim))

        return tree_count, imgOri

def predict_tree_count(img):
    """
    Predicts the number of tree in image
    """
    # Load Config
    hsv_thresh = HSVThresholds(h_low = 31,
                               h_high = 89,
                               s_low = 70,
                               s_high = 255,
                               v_low = 10,
                               v_high = 245)

    cnt_thresh = ContourThresholds(min_area = 400)

    # Predict tree count
    tree_count, imgOut = HSVTreeCountDetector.predict(img = img,
                                              hsv_thresh = hsv_thresh,
                                              cnt_thresh = cnt_thresh)

    return tree_count, imgOut

def get_resize_flag_factor(img, threshold):
    """
    Resize width and height of image if number of pixels exceeds threshold

    Returns:
        resize_flag: True or False
        threshold: # of pixels
    """
    num_pixel = img.shape[0] * img.shape[1]
    resize_flag = False
    resize_factor = 1

    if num_pixel > threshold:
        resize_flag = True
        resize_factor = 0.6

    return resize_flag, resize_factor
