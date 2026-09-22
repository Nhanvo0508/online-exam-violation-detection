from ultralytics import YOLO

model = YOLO(r"D:\TGMT-01\models\best.pt")

model.predict(
    source=0,
    show=True,
    conf=0.5
)