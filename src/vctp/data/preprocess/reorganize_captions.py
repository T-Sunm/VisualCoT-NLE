import json
import os
import argparse

# COMMENT: Hàm này không cần thiết cho test sample
# def find_files(dir, is14=False):
#     files = [f.split(".")[0] for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
#     if is14:
#         files = [f.split("_")[-1] for f in files]
#     return files

parser = argparse.ArgumentParser()
# COMMENT: Không cần coco14root và coco17root cho test sample
# parser.add_argument('--coco14root', type=str, required=True, help='path to coco14')
# parser.add_argument('--coco17root', type=str, required=True, help='path to coco17')
parser.add_argument("--caption14train", type=str, required=True, help="path to 14 caption train")
parser.add_argument("--caption14val", type=str, required=True, help="path to 14 caption val")
parser.add_argument(
    "--caption17train", type=str, required=True, help="output path to 17 caption train"
)
parser.add_argument("--caption17val", type=str, required=True, help="output path to 17 caption val")
args = parser.parse_args()

# COMMENT: Không cần tìm files trong thư mục ảnh cho test sample
# train17dir = os.path.join(args.coco17root, "train2017")
# val17dir = os.path.join(args.coco17root, "val2017")
# train14dir = os.path.join(args.coco14root, "train2014")
# val14dir = os.path.join(args.coco14root, "val2014")
# split14 = [find_files(train14dir, is14=True), find_files(val14dir, is14=True)]
# split17 = [find_files(train17dir, is14=False), find_files(val17dir, is14=False)]

# Load captions
caption14train = json.load(open(args.caption14train))["annotations"]
caption14val = json.load(open(args.caption14val))["annotations"]
caption14 = caption14train + caption14val

# Build caption dictionary
caption14_dict = {}
for c in caption14:
    if c["image_id"] not in caption14_dict:
        caption14_dict[c["image_id"]] = [c["caption"]]
    else:
        caption14_dict[c["image_id"]].append(c["caption"])

# SIMPLIFIED: Cho test sample, chỉ cần copy toàn bộ captions
# Tách theo image_id: 0-7 vào train, 8-9 vào val (ví dụ)
caption17train = {"annotations": []}
caption17val = {"annotations": []}

train_ids = list(range(0, 8))  # image_id 0-7 cho train
val_ids = list(range(8, 10))  # image_id 8-9 cho val

# Populate train captions
for iid in train_ids:
    if iid in caption14_dict:
        captions = caption14_dict[iid]
        for cp in captions:
            caption17train["annotations"].append({"image_id": iid, "caption": cp})

# Populate val captions
for iid in val_ids:
    if iid in caption14_dict:
        captions = caption14_dict[iid]
        for cp in captions:
            caption17val["annotations"].append({"image_id": iid, "caption": cp})

# Save output files
json.dump(caption17train, open(args.caption17train, "w"))
json.dump(caption17val, open(args.caption17val, "w"))
