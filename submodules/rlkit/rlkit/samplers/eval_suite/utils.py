import numpy as np
import cv2
import os

def get_obs_preprocessor(observation_key, additional_keys, desired_goal_key):
    def obs_processor(o):
        obs = o[observation_key]
        for additional_key in additional_keys:
            obs = np.hstack((obs, o[additional_key]))
        return np.hstack((obs, o[desired_goal_key]))
    return obs_processor

def create_blank_image_directories(save_folder, epoch):
    eval_blank_path = f"{save_folder}/epochs/{epoch}/eval_images_blank"
    os.makedirs(eval_blank_path, exist_ok=True)

def create_real_corner_image_directories(save_folder, epoch):
    real_corners_path = f"{save_folder}/epochs/{epoch}/real_corners_prediction"
    os.makedirs(real_corners_path, exist_ok=True)

def create_area_image_directories(save_folder, epoch):
    real_corners_path = f"{save_folder}/epochs/{epoch}/area"
    os.makedirs(real_corners_path, exist_ok=True)

def create_real_corner_image_dump_directories(save_folder, prefix, epoch):
    real_corners_path = f"{save_folder}/epochs/{epoch}/real_corners_dump/{prefix}"
    os.makedirs(real_corners_path, exist_ok=True)

def save_blank_images(env, save_folder, epoch, step_number, aux_output):
    corner_image, eval_image, cnn_color_image_full, cnn_color_image, cnn_image = env.capture_images(aux_output)
    cv2.imwrite(f'{save_folder}/epochs/{epoch}/eval_images_blank/{str(step_number).zfill(3)}.png', eval_image)

def create_regular_image_directories(save_folder, prefix, epoch):
    cnn_path = f"{save_folder}/epochs/{epoch}/{prefix}/cnn_images"
    cnn_color_path = f"{save_folder}/epochs/{epoch}/{prefix}/cnn_color_images"
    cnn_color_full_path = f"{save_folder}/epochs/{epoch}/{prefix}/cnn_color_full_images"
    corners_path = f"{save_folder}/epochs/{epoch}/{prefix}/corners_images"
    eval_path = f"{save_folder}/epochs/{epoch}/{prefix}/eval_images"
    os.makedirs(cnn_path, exist_ok=True)
    os.makedirs(cnn_color_path, exist_ok=True)
    os.makedirs(cnn_color_full_path, exist_ok=True)
    os.makedirs(corners_path, exist_ok=True)
    os.makedirs(eval_path, exist_ok=True)


def create_base_epoch_directory(save_folder, epoch):
    base_path = f"{save_folder}/epochs/{epoch}"
    os.makedirs(base_path, exist_ok=True)

def save_regular_images(env, save_folder, prefix, epoch, step_number, aux_output):
    corner_image, eval_image, cnn_color_image_full, cnn_color_image, cnn_image = env.capture_images(aux_output)
    cv2.imwrite(f'{save_folder}/epochs/{epoch}/{prefix}/corners_images/{str(step_number).zfill(3)}.png', corner_image)
    cv2.imwrite(f'{save_folder}/epochs/{epoch}/{prefix}/eval_images/{str(step_number).zfill(3)}.png', eval_image)
    cv2.imwrite(f'{save_folder}/epochs/{epoch}/{prefix}/cnn_images/{str(step_number).zfill(3)}.png', cnn_image)
    #TODO: save also these images
    cv2.imwrite(f'{save_folder}/epochs/{epoch}/{prefix}/cnn_color_images/{str(step_number).zfill(3)}.png', cnn_color_image)
    cv2.imwrite(f'{save_folder}/epochs/{epoch}/{prefix}/cnn_color_full_images/{str(step_number).zfill(3)}.png', cnn_color_image_full)