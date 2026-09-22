import cv2, base64, numpy as np
COLORS = {
    0:(0,255,0), 1:(255,0,0), 2:(0,0,255), 3:(0,255,255), 4:(255,0,255), 5:(255,255,0)
}

def draw_tracks(frame, tracks, names=None, roi_poly=None):
    out = frame
    if roi_poly is not None:
        cv2.polylines(out, [roi_poly], True, (120,120,120), 2, cv2.LINE_AA)

    for x1, y1, x2, y2, tid, cls, score in tracks:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = COLORS.get(int(cls) % 6, (0,255,0))

        # vẽ bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # LABEL
        name = names.get(int(cls), cls) if names else str(cls)
        pct  = int(round(float(score) * 100))
        label = f"{name} {pct}%"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y1_txt = max(0, y1 - th - 4)
        cv2.rectangle(out, (x1, y1_txt), (x1 + tw + 6, y1_txt + th + 6), color, -1)
        cv2.putText(out, label, (x1 + 3, y1_txt + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
    return out

def encode_jpeg(frame, quality=80):
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")
