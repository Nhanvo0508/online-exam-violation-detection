from pathlib import Path
import os
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")
IMG_SIZE = int(os.getenv("IMG_SIZE", 640))
CONF = float(os.getenv("CONF", 0.75))
IOU = float(os.getenv("IOU", 0.5))
#Camera & Render
CAM_INDEX = int(os.getenv("CAM_INDEX", 0))
JPEG_Q = int(os.getenv("JPEG_Q", 80))
TITLE = os.getenv("TITLE", "Nhóm 6")
MAX_FPS = int(os.getenv("MAX_FPS", 24))

DETECT_FPS = float(os.getenv("DETECT_FPS", 24))
PROBE_FPS = float(os.getenv("PROBE_FPS", 8.0))

USE_GRAYWORLD = bool(int(os.getenv("USE_GRAYWORLD", 1)))
USE_CLAHE = bool(int(os.getenv("USE_CLAHE", 1)))

TOP_MARGIN_PX  = int(os.getenv("TOP_MARGIN_PX", 140))
SIDE_MARGIN_PX = int(os.getenv("SIDE_MARGIN_PX", 20))

MOTION_ALPHA = float(os.getenv("MOTION_ALPHA", 0.02))
MOTION_THRESH = int(os.getenv("MOTION_THRESH", 25))
MOTION_MIN_AREA = int(os.getenv("MOTION_MIN_AREA", 500))

#Tracking
TRACKER_TYPE  = "bytetrack"
TRACK_THRESH  = 0.45
MATCH_THRESH  = 0.75
TRACK_BUFFER  = 24
FRAME_RATE    = 30
SMOOTH_ALPHA  = 0.35
TRACK_LOW_THRESH   = 0.05
TRACK_HIGH_THRESH  = 0.45
NEW_TRACK_THRESH   = 0.45
FUSE_SCORE         = True
PROXIMITY_THRESH   = 0.7
INIT_TRACK_THRESH  = 0.45

# Temporal filter
HYST_ON = float(os.getenv("HYST_ON", 0.60))
HYST_OFF = float(os.getenv("HYST_OFF", 0.50))
DWELL_SEC = float(os.getenv("DWELL_SEC", 0.7))

# Event segmentation
MAX_SEG_SEC     = float(os.getenv("MAX_SEG_SEC", 10.0))
INACTIVITY_SEC  = float(os.getenv("INACTIVITY_SEC", 1.0))
SNAPSHOT_DIR    = os.getenv("SNAPSHOT_DIR", "snapshots")
SNAPSHOT_JPEG_Q = int(os.getenv("SNAPSHOT_JPEG_Q", 85))

