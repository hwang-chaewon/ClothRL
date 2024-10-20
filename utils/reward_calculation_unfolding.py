import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
sys.path.append(current_dir)

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from torchvision.transforms import Compose
from Segment_Anything.segment_anything import SamPredictor, sam_model_registry

from env import cloth_env_unfolding_dualarm

from transformers import pipeline
from PIL import Image

def get_depth_map(img):
    pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")
    img = Image.fromarray(img)
    depth = pipe(img)["depth"]
    depth=np.array(depth)/255.0
    print('      reward 계산: depth 완료    ')
    return depth

# 참고: https://github.com/Victorlouisdg/cloth-competition/blob/main/evaluation-service/backend/main.py
# 참고: https://github.com/facebookresearch/segment-anything
def segment(img):
    torch.cuda.empty_cache()
    DEVICE = 'cpu'
    sam_checkpoint = "/home/hcw/DualRL/utils/Segment_Anything/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=DEVICE)  # "cpu"/"gpu"
    predictor = SamPredictor(sam)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictor.set_image(img)  #이게 시간 소요되는 과정
    input_point = np.array([[200, 150]])
    input_label = np.array([1])
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    max_score=-100
    return_mask=None
    for i, (mask, score) in enumerate(zip(masks, scores)):
        if score>max_score:
            max_score=score
            return_mask=mask
    print('         reward 계산: segment mask 완료       ')
    return return_mask

# 참고: https://github.com/Victorlouisdg/cloth-competition/blob/main/evaluation-service/backend/compute_area.py
def compute_areas_for_image(depth_img, mask_img, fx, fy, cx, cy):
    depth_map = depth_img
    mask = mask_img
    masked_depth_map = np.where(mask > 0, depth_map, 0)
    # print("masked_depth_map: ", masked_depth_map)
    masked_values = masked_depth_map[mask > 0]
    # print("masked_values: ", masked_values)
    mean = np.mean(masked_values)
    # print("mean: ", mean)
    lower_bound = mean - 0.5
    upper_bound = mean + 0.5
    # print("lower_bound,upper_bound : ", lower_bound,upper_bound)
    masked_depth_map = np.where(
        (masked_depth_map > lower_bound) & (masked_depth_map < upper_bound), masked_depth_map, 0
    )
    # print("masked_depth_map2: ", masked_depth_map)
    masked_values = masked_depth_map[(masked_depth_map > lower_bound) & (masked_depth_map < upper_bound)]
    # print("masked_values2: ", masked_values)
    x, y = np.meshgrid(np.arange(masked_depth_map.shape[1]), np.arange(masked_depth_map.shape[0]))
    # print("x, y: ", x,y)
    # print("cx, cy, fx, fy: ",cx, cy, fx, fy)
    X1 = (x - 0.5 - cx) * masked_depth_map / fx
    Y1 = (y - 0.5 - cy) * masked_depth_map / fy
    X2 = (x + 0.5 - cx) * masked_depth_map / fx
    Y2 = (y + 0.5 - cy) * masked_depth_map / fy
    # print("X1, Y1, X2, Y2: ", X1, Y1, X2, Y2)
    pixel_areas = np.abs((X2 - X1) * (Y2 - Y1))
    # print("pixel_areas: ", pixel_areas)
    pixel_areas_masked = np.where(mask > 0, pixel_areas, 0)
    # print("pixel_areas_masked: ", pixel_areas_masked)
    masked_pixel_values = pixel_areas_masked[pixel_areas_masked > 0]
    # print("masked_pixel_values: ", masked_pixel_values)
    # Normalize the pixel areas to the range 0-255
    pixel_areas_normalized = cv2.normalize(
        pixel_areas_masked, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    # print("pixel_areas_normalized: ", pixel_areas_normalized)
    return pixel_areas_normalized, X1, X2, Y1, Y2

def get_area_reward(img, camera_matrix):
    cm=camera_matrix
    mask=segment(img)
    depth_map=get_depth_map(img)
    area, X1,X2,Y1,Y2=compute_areas_for_image(depth_map, mask, cm[0,0], cm[1,1], cm[0,2], cm[1,2])
    return area

def get_unfolding_reward_function():
    def unfolding_reward_function(img, camera_matrix):
        cm=camera_matrix
        mask=segment(img)
        depth_map=get_depth_map(img)
        area, X1,X2,Y1,Y2=compute_areas_for_image(depth_map, mask, cm[0,0], cm[1,1], cm[0,2], cm[1,2])
        return area
    return unfolding_reward_function

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)