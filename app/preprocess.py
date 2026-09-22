import cv2
import numpy as np

# Gray-World white balance
def gray_world(bgr):
    b, g, r = cv2.split(bgr)
    mb, mg, mr = b.mean() + 1e-6, g.mean() + 1e-6, r.mean() + 1e-6
    k = (mb + mg + mr) / 3.0
    kb, kg, kr = k/mb, k/mg, k/mr
    b = cv2.multiply(b, kb)
    g = cv2.multiply(g, kg)
    r = cv2.multiply(r, kr)
    out = cv2.merge([b, g, r])
    return np.clip(out, 0, 255).astype(np.uint8)

# CLAHE on L channel (LAB)
def clahe_l(bgr, clip=2.0, tile=(8,8)):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


# Apply ROI polygon mask
def apply_roi(bgr, poly):
    mask = np.zeros(bgr.shape[:2], np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return cv2.bitwise_and(bgr, bgr, mask=mask), mask

class MotionGate:
    def __init__(self, alpha=0.02, th=25, min_area=500):
        self.bg = None
        self.alpha = alpha
        self.th = th
        self.min_area = min_area
        # lưu cường độ motion (số pixel thay đổi) để dùng làm feature ML
        self.last_motion_px = 0

    def check(self, roi_bgr):
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        if self.bg is None:
            self.bg = gray.astype("float")
            self.last_motion_px = 0
            return True  # allow first frames
        cv2.accumulateWeighted(gray, self.bg, self.alpha)
        diff = cv2.absdiff(gray, cv2.convertScaleAbs(self.bg))
        _, bw = cv2.threshold(diff, self.th, 255, cv2.THRESH_BINARY)
        motion_px = int(cv2.countNonZero(bw))
        self.last_motion_px = motion_px
        return motion_px > self.min_area