#*************바꾼부분****************#
# unfolding task: 화면에 보이는 넓이 계산
# folding task

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
sys.path.append(current_dir)

# sys.path.append("..")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from torchvision.transforms import Compose
from Segment_Anything.segment_anything import SamPredictor, sam_model_registry
# from Depth_Anything.depth_anything.dpt import DepthAnything
# from Depth_Anything.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

from env import cloth_env_unfolding_dualarm

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'

# 참고: https://github.com/LiheYoung/Depth-Anything
'''
def get_depth_map3(img):
    # load predefined model-http
    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format('vitl')).to(DEVICE).eval()  #'vits', 'vitb', 'vitl'
    # load predefined model-local
    # local_model_dir="/home/hcw/DualRL/utils/Depth_Anything/pretrained"
    # local_model_path="/home/hcw/DualRL/utils/Depth_Anything/pretrained/pytorch_model.bin"
    # local_config_path="/home/hcw/DualRL/utils/Depth_Anything/pretrained/config.json"
    # depth_anything = DepthAnything.from_pretrained(local_model_dir).to(DEVICE).eval()
    
    # image를 model에 입력하기 전 필요한 transformation 설정
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    # model에 입력하기 위해 필요한 transform을 image에 적용
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    h, w = image.shape[:2]
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
    # image를 model에 입력해 depth 예측
    with torch.no_grad():
        depth = depth_anything(image)
    # 원래 image크기로 만들고, [0,1] 범위로 정규화 #[0,255] 범위로 정규화
    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) #* 255.0
    depth = depth.cpu().numpy().astype(np.uint8)

    # 흑백 or 컬러
    # depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO) #colot로 하면 mask와 shape이 안 맞아서 다른 함수에서 에러 발생

    print("depth map 계산 완료===================================")

    return depth
'''

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
    # if depth_map.shape[2] == 3:
    #     depth_map = np.mean(depth_map, axis=2).astype(depth_map.dtype)

    # mask가 0보다 큰 위치에만 depth_map의 값을 유지하고, 나머지는 0으로
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
    # fx: focal length along x
    # fy: focal length along y
    # cx: principal point의 x좌표 (principal point: point that optical axis intersects the image plane)
    # cy: principal point의 y좌표
    # cloth_env.py의 get_camera_matrices()를 이용해야 할듯..
# Camera matrix (mtx) has the form: (GPT 피셜. 확실하지 않음)
# [ fx  0  cx ]
# [  0 fy  cy ]
# [  0  0   1 ]
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