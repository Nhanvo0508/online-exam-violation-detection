import os, json, sys, argparse

# Default paths (you can override via CLI)
DEFAULT_EVENTS = r"C:\Users\anbin\PycharmProjects\TGMT-01\MachineLearning\tools\cv_events_with_crops.jsonl"
DEFAULT_OUT = r"C:\Users\anbin\PycharmProjects\TGMT-01\MachineLearning\tools\train_with_labels.json"

# Default violation classes
VIOLATION_CLASSES = {"book", "calculator", "cheat_sheet", "earphone", "notebook", "phone"}

# Default LABEL_0_IDS (you can still override via --ids_file)
DEFAULT_LABEL_0_IDS = {
    "EVT-1763415745511-11-cls4", "EVT-1763416204214-16-cls4", "EVT-1763416205442-17-cls4",
    "EVT-1763416271760-18-cls5", "EVT-1763416279720-19-cls5", "EVT-1763416284571-20-cls5",
    "EVT-1763416292644-20-cls5", "EVT-1763416295473-22-cls5", "EVT-1763416302009-22-cls5",
    "EVT-1763416304346-22-cls5", "EVT-1763416675575-27-cls2", "EVT-1763416682252-27-cls2",
    "EVT-1763416693453-27-cls2", "EVT-1763416709056-33-cls1", "EVT-1763400864317-60-cls4",
    "EVT-1763400919598-73-cls4", "EVT-1763416940118-38-cls5", "EVT-1763416944452-39-cls5",
    "EVT-1763416947357-40-cls5", "EVT-1763416955494-40-cls5", "EVT-1763416966179-43-cls5",
    "EVT-1763416971496-44-cls5", "EVT-1763416204570-16-cls5", "EVT-1763415745953-11-cls2",
    "EVT-1763415703911-6-cls5", "EVT-1763401004723-9-cls4", "EVT-1763400921633-73-cls4",
    "EVT-1763400892553-68-cls4", "EVT-1763400866040-60-cls4", "EVT-1763417461569-47-cls5",
    "EVT-1763417462169-47-cls0", "EVT-1763417679008-51-cls5", "EVT-1763418535492-58-cls3",
}

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for i, ln in enumerate(f, start=1):
            ln = ln.strip()
            if not ln: continue
            try:
                yield json.loads(ln)
            except Exception as e:
                print(f"[WARN] skip line {i} json parse error: {e}", file=sys.stderr)

def load_train_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_ids_file(path):
    s = set()
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            s.add(ln)
    return s

def normalize_clsname(cn: str) -> str:
    if not cn: return ""
    return cn.strip().lower()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--events", default=DEFAULT_EVENTS, help="input events jsonl (cv_events_with_crops.jsonl)")
    p.add_argument("--emb", default=None, help="optional: precomputed train_with_emb.json (list) to keep embeddings")
    p.add_argument("--ids_file", default=None, help="optional: file with event_id per line to override as label=0")
    p.add_argument("--out", default=DEFAULT_OUT, help="output train json (list) with labels (keeps 'emb' if present)")
    args = p.parse_args()

    label0_ids = set(DEFAULT_LABEL_0_IDS)
    if args.ids_file:
        label0_ids |= load_ids_file(args.ids_file)
        print(f"[INFO] loaded {len(label0_ids)} label-0 ids (including defaults and file)")

    # If emb JSON provided, load into a dict by event_id for quick emb lookup
    emb_map = {}
    if args.emb:
        try:
            train_list = load_train_list(args.emb)
            for item in train_list:
                # original embedding may be inside features['emb']
                eid = item.get("event_id") or item.get("features", {}).get("event_id")
                # safer: some train formats don't include event_id - we prefer to match by features snapshot if needed
                # We'll attempt to use event_id field if present in item
                if eid:
                    emb_map[eid] = item.get("features", {}).get("emb")
            print(f"[INFO] loaded embeddings from {args.emb} for {len(emb_map)} event_ids")
        except Exception as e:
            print(f"[WARN] cannot load emb file {args.emb}: {e}", file=sys.stderr)

    out_list = []
    total = 0
    n_label0 = 0
    n_label1 = 0
    n_no_feats = 0
    # read events JSONL
    for rec in load_jsonl(args.events):
        total += 1
        feats = rec.get("features")
        if feats is None:
            n_no_feats += 1
            continue
        eid = rec.get("event_id")
        cls_name = normalize_clsname(rec.get("cls_name", ""))
        # decide label: priority: ids file/default set -> else class mapping -> else 0
        if eid and eid in label0_ids:
            label = 0
        elif cls_name in VIOLATION_CLASSES:
            label = 1
        else:
            label = 0

        # incorporate embedding if present in emb_map (and not already in feats)
        if "emb" not in feats and eid and eid in emb_map and isinstance(emb_map[eid], list):
            feats["emb"] = emb_map[eid]

        out_list.append({"features": feats, "label": int(label)})
        if label == 0:
            n_label0 += 1
        else:
            n_label1 += 1

    # write output
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fo:
        json.dump(out_list, fo, indent=2, ensure_ascii=False)

    print(f"[OK] wrote {len(out_list)} samples to {args.out}  (label1={n_label1}, label0={n_label0}, skipped_feats={n_no_feats})")

if __name__ == "__main__":
    main()
