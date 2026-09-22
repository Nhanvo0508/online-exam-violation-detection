import numpy as np
import torch
from ultralytics import YOLO
class Detector:
    def __init__(self, model_path, conf=0.45, imgsz=640, iou=0.5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        self.model.to(self.device)
        if self.device == "cuda":
            try:
                self.model.model.half()
            except Exception:
                pass
        self.conf = conf
        self.imgsz = imgsz
        self.iou = iou
        torch.backends.cudnn.benchmark = True
        # warmup
        _ = self.model.predict(np.zeros((imgsz, imgsz, 3), dtype=np.uint8),
                                device=self.device, verbose=False)
    def infer(self, frame):
        res = self.model.predict(
            frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
            device=self.device, verbose=False
        )
        # normalize to [x1,y1,x2,y2,score,cls]
        r0 = res[0]
        dets = []
        if hasattr(r0, "boxes") and r0.boxes is not None and len(r0.boxes) >0:
            boxes = r0.boxes.xyxy.cpu().numpy()
            scores = r0.boxes.conf.cpu().numpy()
            clss = r0.boxes.cls.cpu().numpy().astype(int)
            for bb, s, c in zip(boxes, scores, clss):
                dets.append([float(bb[0]), float(bb[1]), float(bb[2]),
    float(bb[3]), float(s), int(c)])
        names = r0.names if hasattr(r0, "names") else {}
        return dets, names
