import json, os, shutil
from tqdm import tqdm

# Load COCO 2014 IDs
coco_2014_ids = set(
    [img['id'] for img in json.load(open("/home/research/workspace/data/raw/coco/annotations/instances_train2014.json"))['images']] +
    [img['id'] for img in json.load(open("/home/research/workspace/data/raw/coco/annotations/instances_val2014.json"))['images']]
)

# Paths
base = "/home/research/workspace/data/raw/scene-graph"
sources = [f"{base}/scene_graph_coco17", f"{base}/scene_graph_coco17_attr"]
targets = [f"{base}/scene_graph_coco14", f"{base}/scene_graph_coco14_attr"]

# Copy files
for src, dst in zip(sources, targets):
    os.makedirs(dst, exist_ok=True)
    print(f"\nCopying {os.path.basename(src)} -> {os.path.basename(dst)}...")
    
    copied = 0
    for img_id in tqdm(coco_2014_ids):
        fname = f"{img_id:012d}.json"
        if os.path.exists(f"{src}/{fname}"):
            shutil.copy2(f"{src}/{fname}", f"{dst}/{fname}")
            copied += 1
    
    print(f"✓ Copied {copied}/{len(coco_2014_ids)} files")

print(f"\n✓ Done! COCO 2014 scene graphs in {base}/scene_graph_coco14/")
