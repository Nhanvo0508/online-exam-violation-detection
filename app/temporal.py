# temporal.py
import time
from collections import defaultdict, deque

class TemporalFilter:
    def __init__(self, on=0.5, off=0.4, dwell=0.9,
                 max_seg_sec=None, inactivity_sec=1.0):
        self.ON = on
        self.OFF = off
        self.DWELL = dwell
        self.MAX_SEG = max_seg_sec
        self.INACT = inactivity_sec
        # key = (tid, cls)
        self.state = defaultdict(lambda: {
            "on": False, "t_in": None, "cls": None,
            "scores": deque(maxlen=5),
            "last_seen": None
        })

    def update(self, tracks):
        now = time.time()
        events = []
        seen_keys = set()

        for x1,y1,x2,y2,tid,cls,score in tracks:
            key = (int(tid), int(cls))
            st = self.state[key]
            st["scores"].append(float(score))
            st["last_seen"] = now

            avg = sum(st["scores"]) / max(1, len(st["scores"]))
            if not st["on"] and avg >= self.ON:
                st["on"] = True
                st["t_in"] = now
                st["cls"] = int(cls)
            elif st["on"]:
                # Auto-segment nếu đoạn quá dài
                if self.MAX_SEG and (now - st["t_in"]) >= self.MAX_SEG:
                    if (now - st["t_in"]) >= self.DWELL:
                        events.append({
                            "type":"violation",
                            "id": key[0],
                            "cls": key[1],
                            "t_enter": st["t_in"],
                            "t_leave": now,
                            "score_avg": avg
                        })
                    # mở đoạn mới ngay tại thời điểm hiện tại
                    st["t_in"] = now
                    st["scores"].clear()
            seen_keys.add(key)

        for key, st in list(self.state.items()):
            if st["on"] and key not in seen_keys:
                # Debounce OFF: chỉ kết thúc nếu mất >= INACTIVITY_SEC
                if st["last_seen"] is not None and (now - st["last_seen"]) >= self.INACT:
                    avg = sum(st["scores"]) / max(1, len(st["scores"]))
                    if st["t_in"] and (now - st["t_in"]) >= self.DWELL:
                        events.append({
                            "type":"violation",
                            "id": key[0],
                            "cls": key[1],
                            "t_enter": st["t_in"],
                            "t_leave": now,
                            "score_avg": avg
                        })
                    st["on"] = False
                    st["t_in"] = None
                    st["scores"].clear()

        return events
