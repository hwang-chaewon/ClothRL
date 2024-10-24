import sys
import os
import matplotlib.pyplot as plt
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

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0], pos_points[:, 1], color="green", marker="*", s=marker_size, edgecolor="white", linewidth=1.25
    )
    ax.scatter(
        neg_points[:, 0], neg_points[:, 1], color="red", marker="*", s=marker_size, edgecolor="white", linewidth=1.25
    )
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

    lower_black = np.array([1, 1, 1])
    upper_black = np.array([10, 10, 10])
    black_mask = cv2.inRange(img, lower_black, upper_black)
    input_point = np.column_stack(np.where(black_mask > 0))
    input_label = np.ones(input_point.shape[0])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis("off")
        plt.show()
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
    masked_values = masked_depth_map[mask > 0]
    mean = np.mean(masked_values)
    lower_bound = mean - 0.5
    upper_bound = mean + 0.5
    masked_depth_map = np.where(
        (masked_depth_map > lower_bound) & (masked_depth_map < upper_bound), masked_depth_map, 0
    )
    x, y = np.meshgrid(np.arange(masked_depth_map.shape[1]), np.arange(masked_depth_map.shape[0]))
    X1 = (x - 0.5 - cx) * masked_depth_map / fx
    Y1 = (y - 0.5 - cy) * masked_depth_map / fy
    X2 = (x + 0.5 - cx) * masked_depth_map / fx
    Y2 = (y + 0.5 - cy) * masked_depth_map / fy
    pixel_areas = np.abs((X2 - X1) * (Y2 - Y1))
    pixel_areas_masked = np.where(mask > 0, pixel_areas, 0)
    return np.sum(pixel_areas_masked), X1, X2, Y1, Y2

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

def get_unfolding_with_motion_reward_function(constraints, single_goal_dim, sparse_dense, success_reward, fail_reward, extra_reward):
    constraint_distances = [c['distance'] for c in constraints]
    def unfolding_with_motion_reward_function(img, camera_matrix, achieved_goal, desired_goal, info):
        cm=camera_matrix
        mask=segment(img)
        depth_map=get_depth_map(img)
        area, X1,X2,Y1,Y2=compute_areas_for_image(depth_map, mask, cm[0,0], cm[1,1], cm[0,2], cm[1,2])

        achieved_oks = np.zeros(
            (achieved_goal.shape[0], len(constraint_distances)))
        achieved_distances = np.zeros(
            (achieved_goal.shape[0], len(constraint_distances)))
        for i, constraint_distance in enumerate(constraint_distances):
            achieved = achieved_goal[:, i*single_goal_dim:(i+1)*single_goal_dim]
            desired = desired_goal[:, i*single_goal_dim:(i+1)*single_goal_dim]
            achieved_distances_per_constraint = goal_distance(achieved, desired)
            constraint_ok = achieved_distances_per_constraint < constraint_distance
            achieved_distances[:, i] = achieved_distances_per_constraint
            achieved_oks[:, i] = constraint_ok
        successes = np.all(achieved_oks, axis=1)
        fails = np.invert(successes)
        task_rewards = successes.astype(np.float32).flatten()*success_reward
        if sparse_dense:
            dist_rewards = np.sum((1 - achieved_distances/np.array(constraint_distances)),
                                  axis=1) / len(constraint_distances)
            task_rewards += dist_rewards*extra_reward  # Extra for being closer to the goal
            if "num_future_goals" in info.keys():
                num_future_goals = info['num_future_goals']
                task_rewards[-num_future_goals:] = success_reward
        task_rewards[fails] = fail_reward
        return area+task_rewards
    return unfolding_with_motion_reward_function

def get_folding_reward_function():
    def folding_reward_function(pre_img, pre_cm, post_img, post_cm):
        area_pre, X1_pre,X2_pre,Y1_pre,Y2_pre=get_area_reward(pre_img, pre_cm)
        area_post, X1_post,X2_post,Y1_post,Y2_post=get_area_reward(post_img, post_cm)
        X_m_pre=(X1_pre+X2_pre)/2
        Y_m_pre=(Y1_pre+Y2_pre)/2
        X_m_post=(X1_post+X2_post)/2
        Y_m_post=(Y1_post+Y2_post)/2
        reward_1=abs((area_pre-area_post))/2
        reward_2=(abs(X2_pre-X2_post)+abs(Y2_pre-Y2_post)+abs(X_m_pre-X_m_post)+abs(Y_m_pre-Y_m_post))/4
        return 2/(reward_1+reward_2) #차이를 최소화할 때 reward가 높아지도록
    
    return folding_reward_function