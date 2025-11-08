import os
from PIL import Image
import json
import torch
import numpy as np
from tqdm import tqdm
import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from vctp.utils.clip_manager import get_clip_model


parser = argparse.ArgumentParser()
parser.add_argument("--questions", type=str, required=True, help="path to questions")
parser.add_argument("--images", type=str, required=True, help="path to coco images")
parser.add_argument(
    "--qfeatures", type=str, required=True, help="output features path for questions"
)
parser.add_argument(
    "--ifeatures", type=str, required=True, help="output features path for images"
)
args = parser.parse_args()

dataset = json.load(open(args.questions))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model, processor = get_clip_model(
    model_type="full", device=device, use_safetensors=True
)

text_embeds_list = []
image_embeds_list = []
for q in tqdm(dataset):
    imageID = q["image_id"]
    imageName = q["image_name"]
    if "train2014" in imageName:
        split_folder = "train2014"
    elif "val2014" in imageName:
        split_folder = "val2014"
    else:
        split_folder = "test2014"

    file_name = os.path.join(args.images, split_folder, imageName)
    image = Image.open(file_name)
    inputs = processor(
        text=[q["question"]], images=image, return_tensors="pt", padding=True, truncation=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    text_embeds_list.append(outputs["text_embeds"].detach().cpu())
    image_embeds_list.append(outputs["image_embeds"].detach().cpu())
text_embeds_list = torch.cat(text_embeds_list, dim=0)
image_embeds_list = torch.cat(image_embeds_list, dim=0)
np.save(args.ifeatures, image_embeds_list.numpy())
np.save(args.qfeatures, text_embeds_list.numpy())
