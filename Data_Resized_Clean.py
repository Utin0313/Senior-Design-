import os, shutil
from collections import defaultdict

SRC = "Data_Resized"
DST = "Data_Resized_Clean"
CLASSES = ["Breast", "Control", "Prostate", "Skin"]

def get_photo_id(fname):
    return fname.split("_masked")[0]  # "Breast_1_Train_10_20251211_144841"

def get_strip_num(fname):
    return fname.split("_")[1]        # "1" from "Breast_1_..."

# Build a clean split:
# Strip 1 → Test, Strip 2 → Val, Strips 3-6 → Train
# AND all 3 variants of a photo stay in the same split
STRIP_SPLIT = {
    "1": "Test",
    "2": "Validation",
    "3": "Train", "4": "Train", "5": "Train", "6": "Train"
}

# Create output dirs
for split in ["Train", "Validation", "Test"]:
    for cls in CLASSES:
        os.makedirs(os.path.join(DST, split, cls), exist_ok=True)

moved = defaultdict(int)

# Read from Train only — it has all 6 strips
for cls in CLASSES:
    src_cls = os.path.join(SRC, "Train", cls)
    for fname in sorted(os.listdir(src_cls)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        strip_num = get_strip_num(fname)
        dest_split = STRIP_SPLIT.get(strip_num)
        if not dest_split:
            continue

        shutil.copy2(
            os.path.join(src_cls, fname),
            os.path.join(DST, dest_split, cls, fname)
        )
        moved[dest_split] += 1

# Summary
print("\n=== Files copied ===")
for split, count in moved.items():
    print(f"  {split}: {count} images total ({count//len(CLASSES)} per class)")

print("\n=== Per class breakdown ===")
for split in ["Train", "Validation", "Test"]:
    for cls in CLASSES:
        n = len(os.listdir(os.path.join(DST, split, cls)))
        print(f"  {split}/{cls}: {n} images")

"""
    Train:      600 images/class  (strips 3 - 6 * 50 photos * 3 variants)
    Validation: 150 images/class  (strip 2 * 50 photos * 3 variants)
    Test:       150 images/class  (strip 1 * 50 photos * 3 variants)
"""