# 잘못된 구조. policy의 결과로 cloth position이 나오는 게 아님.
import mujoco_py
import osc_binding
import cv2
import numpy as np
import copy
from gym.utils import seeding, EzPickle

#***********바꾼부분***********#
from utils import reward_calculation, reward_calculation_unfolding
import random

import gym
import os
from scipy import spatial
from env.template_renderer import TemplateRenderer
import math
from collections import deque
import copy
from utils import mujoco_model_kwargs
from shutil import copyfile
import psutil
import gc
from xml.dom import minidom
from mujoco_py.utils import remove_empty_lines
import albumentations as A
import pandas as pd
from utils import task_definitions
import logging
import subprocess

#*******바꾼부분: 그랩******************
from collections import defaultdict
from termcolor import colored
from simple_pid import PID
import time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(message)s')


def compute_cosine_distance(vec1, vec2):
    if np.all(vec1 == 0) or np.all(vec2 == 0):
        return 0
    else:
        cosine_dist = spatial.distance.cosine(vec1, vec2)
        if np.isnan(cosine_dist):
            return 0
        else:
            return cosine_dist


class ClothEnv_(object):
    def __init__(
        self,
        timestep,
        sparse_dense,
        success_distance,
        goal_noise_range,
        frame_stack_size,
        output_max,
        success_reward,
        fail_reward,
        extra_reward,
        kp,
        damping_ratio,
        control_frequency,
        ctrl_filter,
        save_folder,
        randomization_kwargs,
        robot_observation,
        max_close_steps,
        model_kwargs_path,
        image_obs_noise_mean=1,
        image_obs_noise_std=0,
        has_viewer=True,
        #****************바꾼부분: 시각화********************#
        image_size=200,  #100
        glfw_window=True,
    ):
        
        #****************바꾼부분: 시각화********************#
        self.glfw_window=glfw_window
        self.original_ld_preload = os.environ.get('LD_PRELOAD')

        #******바꾼부분: 액션**************
        self.action=None

        self.albumentations_transform = A.Compose(
            [
                A.RGBShift(r_shift_limit=15, g_shift_limit=15,
                           b_shift_limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Blur(blur_limit=7, always_apply=False, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2,
                              saturation=0.2, hue=0.2, always_apply=False, p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0,
                               always_apply=False, p=0.5),
            ]
        )

        self.model_kwargs_path = model_kwargs_path
        self.success_distance = success_distance

        self.process = psutil.Process(os.getpid())
        self.randomization_kwargs = randomization_kwargs
        file_dir = os.path.dirname(os.path.abspath(__file__))

        source = os.path.join(file_dir, "mujoco_templates", "arena.xml")
        destination = os.path.join(save_folder, "arena.xml")
        copyfile(source, destination)
        self.template_renderer = TemplateRenderer(save_folder)
        self.save_folder = save_folder
        self.filter = ctrl_filter
        self.kp = kp
        self.damping_ratio = damping_ratio
        self.output_max = output_max
        self.success_reward = success_reward
        self.fail_reward = fail_reward
        self.extra_reward = extra_reward
        self.sparse_dense = sparse_dense
        self.goal_noise_range = goal_noise_range
        self.frame_stack_size = frame_stack_size
        self.image_size = (image_size, image_size)

        self.has_viewer = has_viewer
        self.image_obs_noise_mean = image_obs_noise_mean
        self.image_obs_noise_std = image_obs_noise_std
        self.robot_observation = robot_observation

        self.max_close_steps = max_close_steps
        self.frame_stack = deque([], maxlen=self.frame_stack_size)

        self.limits_min = [-0.35, -0.35, 0.0]
        self.limits_max = [0.05, 0.05, 0.4]

        self.single_goal_dim = 3 #목표 위치가 몇 차원인지
        self.timestep = timestep
        self.control_frequency = control_frequency

        steps_per_second = 1 / self.timestep
        self.substeps = 1 / (self.timestep*self.control_frequency)
        self.between_steps = 1000 / steps_per_second
        self.delta_tau_max = 1000 / steps_per_second

        #*****바꾼부분: 카메라 (1007) *********
        # self.eval_camera = "eval_camera"
        self.eval_camera = "front"

        self.episode_ee_close_steps = 0

        self.seed()

        self.initial_qpos = np.array([0.212422, 0.362907, -0.00733391, -1.9649, -0.0198034, 2.37451, -1.50499])

        self.ee_site_name = 'grip_site'
        self.joints = ["joint1", "joint2", "joint3",
                       "joint4", "joint5", "joint6", "joint7"]
        
        #*******바꾼부분: 양팔(0829)********
        self.initial_qpos_2 = np.array([0.212422, 0.362907, -0.00733391, -1.9649, -0.0198034, 2.37451, -1.50499])
        self.ee_site_name_2 = 'grip_site_2'
        self.joints_2 = ["joint1_2", "joint2_2", "joint3_2",
                       "joint4_2", "joint5_2", "joint6_2", "joint7_2"]

        self.mjpy_model = None
        self.sim = None
        self.viewer = None

        model_kwargs, model_numerical_values = self.build_xml_kwargs_and_numerical_values(
            randomize=self.randomization_kwargs['dynamics_randomization'])
        self.mujoco_model_numerical_values = model_numerical_values

        #*****바꾼부분: 그랩 (1001) 버전2**********
        self.setup_initial_state_and_sim(model_kwargs)
        # self.model_xml=self.setup_initial_state_and_sim(model_kwargs)

        self.dump_xml_models()

        #**********바꾼부분: 모션플래닝*****************
        self.build_motion_distance_and_constraints()

        #*******바꾼부분: 그랩(1019)********
        # self.a2_index=random.choice([0, 8, 72, 80])
        # self.a2_index=0
        # cloth_2=list(self.get_cloth_position_W().items())[self.a2_index] #B0_0과, 그에 해당하는 value
        # self.grasp_cloth_2=cloth_2[0]  #B0_0형태
        # self.action_2 = cloth_2[1] #(x,y,z) 형태
        for index, d in enumerate(list(self.get_cloth_position_W().items())): #B0_0과, 그에 해당하는 value
                if self.singlemocap in d[0]: #B0_0형태
                    self.a2_index=index
                    self.action_2 = d[1] #(x,y,z) 형태

        #*******바꾼부분: 그랩(1019)********
        # self.a_index=random.choice(list(range(81)).remove(self.a2_index))
        # cloth=list(self.get_cloth_position_W().items())[self.a_index] #B0_0과, 그에 해당하는 value
        # self.grasp_cloth=cloth[0]  #B0_0형태
        # self.action_for_show = cloth[1] #(x,y,z) 형태
        for index, d in enumerate(list(self.get_cloth_position_W().items())): #B0_0과, 그에 해당하는 value
                if self.policymocap in d[0]: #B0_0형태
                    self.a_index=index
                    self.action_for_show = d[1] #(x,y,z) 형태

        self.action_space = gym.spaces.Box(-1,1, shape=(3,), dtype='float32')

        self.relative_origin = self.get_ee_position_W()
        
        #*******바꾼부분: 양팔(0829)********
        self.relative_origin_2 = self.get_ee_position_W_2()

        self.goal, self.goal_noise = self.sample_goal_I()

        self.reset_camera()

        #*********바꾼부분: 시각화***********#
        image_obs, initial_image = self.get_image_obs()

        for _ in range(self.frame_stack_size):
            self.frame_stack.append(image_obs)
        obs = self.get_obs()

        self.observation_space = gym.spaces.Dict(dict(
            #*********바꾼부분***********#
            # desired_goal=gym.spaces.Box(-np.inf, np.inf,
            #                             shape=obs['achieved_goal'].shape, dtype='float32'),
            desired_goal=gym.spaces.Box(-np.inf, np.inf,
                                        shape=obs['desired_goal'].shape, dtype='float32'),
                                        
            achieved_goal=gym.spaces.Box(-np.inf, np.inf,
                                         shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=gym.spaces.Box(-np.inf, np.inf,
                                       shape=obs['observation'].shape, dtype='float32'),
            robot_observation=gym.spaces.Box(-np.inf, np.inf,
                                             shape=obs['robot_observation'].shape, dtype='float32'),
            image=gym.spaces.Box(-np.inf, np.inf,
                                 shape=obs['image'].shape, dtype='float32')
        ))

        # ***************바꾼부분: 그랩********************
        self.leftfinger_id=self.sim.model.body_name2id('leftfinger')
        self.rightfinger_id=self.sim.model.body_name2id('rightfinger')
        self.grip_geom_id=self.sim.model.geom_name2id('grip_geom')

        #**********바꾼부분: 양팔 (0831) ***********
        self.leftfinger_id_2=self.sim.model.body_name2id('leftfinger_2')
        self.rightfinger_id_2=self.sim.model.body_name2id('rightfinger_2')
        self.grip_geom_id_2=self.sim.model.geom_name2id('grip_geom_2')

        # ***************바꾼부분: 그랩 (0903)********************
        self.create_lists()
        self.groups = defaultdict(list)
        self.groups["All"] = list(range(7)) #len(self.sim.data.ctrl)
        # self.create_group("Arm", list(range(8)))
        self.create_group("Gripper", [0])
        self.actuated_joint_ids = np.array([i[2] for i in self.actuators])
        self.reached_target = False
        self.current_output = np.zeros(7)  #len(self.sim.data.ctrl)

        #********바꾼부분: 양팔 (0903)********
        self.create_lists_2()
        self.groups_2 = defaultdict(list)
        self.groups_2["All"] = list(range(7))
        # self.create_group_2("Arm_2", list(range(8)))
        self.create_group_2("Gripper_2", [0])
        self.actuated_joint_ids_2 = np.array([i[2] for i in self.actuators_2])
        self.reached_target_2 = False
        self.current_output_2 = np.zeros(7)  #len(self.sim.data.ctrl)

        # ***************바꾼부분: 시각화********************
        cv2.imwrite('/home/hcw/DualRL/data/saved_image.png', initial_image)

        # ***************바꾼부분: 그랩********************
        # self.open_gripper()

        #********바꾼부분: 양팔 (0829)**********
        # self.open_gripper_2()

        # #*******바꾼부분: 시각화 (0902) *****************
        # site_index_1 = self.sim.model.site_name2id("S0_0")
        # self.sim.model.site_rgba[site_index_1] = np.array([1.0, 0.0, 0.0, 1.0])
        # site_index_2 = self.sim.model.site_name2id("S0_8")
        # self.sim.model.site_rgba[site_index_2] = np.array([0.0, 1.0, 0.0, 1.0])
        # site_index_3 = self.sim.model.site_name2id("S8_0")
        # self.sim.model.site_rgba[site_index_3] = np.array([0.0, 0.0, 1.0, 1.0])
        # site_index_4 = self.sim.model.site_name2id("S8_8")
        # self.sim.model.site_rgba[site_index_4] = np.array([1.0, 1.0, 1.0, 1.0])
        # self.sim.forward()

        #****바꾼부분: 양팔 (1003) **********
        # self.step_2()
        # self.goal, self.goal_noise = self.sample_goal_I()

        # #****바꾼부분: 그랩 (1001) 버전1******
        # self.set_cloth_mocapid()

    #*****바꾼부분: 그랩 (1001) 버전1 ***************
    def set_cloth_mocapid(self):
        body_names = self.sim.model.body_names   # (B0_0 형식)
        b_body_names = [name for name in body_names if name.startswith('B')]
        for body in b_body_names:
            body_id=self.sim.model.body_name2id(body)  #body의 id (0~110 숫자)
            # self.sim.data.mocap_pos[body_id-2][:] = self.sim.data.get_body_xpos(body) # body id에 해당하는 mocap pos에 body의 좌표 대입
            self.sim.data.mocap_pos[body_id-2][:]=self.get_cloth_position_W()[body]  # body에 대응하는 cloth 좌표값을 mocap_pos에 대입 


    # ***************바꾼부분: 그랩********************
    def create_group(self, group_name, idx_list):
        try:
            assert len(idx_list) <= len(self.sim.data.ctrl), "Too many joints specified!"
            assert (group_name not in self.groups.keys()), "A group with name {} already exists!".format(group_name)
            # assert np.max(idx_list) <= len(self.sim.data.ctrl), "List contains invalid actuator ID (too high)"
            self.groups[group_name] = idx_list
            print("Created new control group '{}'.".format(group_name))
        except Exception as e:
            print(e)
            print("Could not create a new group.")
    #***** 바꾼부분: 양팔 (0903) ***************
    def create_group_2(self, group_name, idx_list):
        try:
            assert len(idx_list) <= 7, "Too many joints specified!"
            assert (group_name not in self.groups_2.keys()), "A group with name {} already exists!".format(group_name)
            # assert np.max(idx_list) <= len(self.sim.data.ctrl), "List contains invalid actuator ID (too high)"
            self.groups_2[group_name] = idx_list
            print("Created new control group '{}'.".format(group_name))
        except Exception as e:
            print(e)
            print("Could not create a new group.")

    # ***************바꾼부분: 그랩********************
    def create_lists(self):
        self.controller_list = []
        sample_time = 0.0001
        p_scale = 3
        i_scale = 0.0
        i_gripper = 0
        d_scale = 0.1
        self.controller_list.append(  # (1) Shoulder Pan Joint
            PID(7 * p_scale,0.0 * i_scale,1.1 * d_scale,setpoint=0,output_limits=(-2, 2),sample_time=sample_time,))
        self.controller_list.append(  # (2) Shoulder Lift Joint
            PID(10 * p_scale,0.0 * i_scale,1.0 * d_scale,setpoint=-1.57,output_limits=(-2, 2),sample_time=sample_time,))
        self.controller_list.append( # (3) Elbow Joint
            PID(5 * p_scale,0.0 * i_scale,0.5 * d_scale,setpoint=1.57,output_limits=(-2, 2),sample_time=sample_time,))
        self.controller_list.append( # (내가추가) (3.5) Elbow2 Joint
            PID(5 * p_scale,0.0 * i_scale,0.5 * d_scale,setpoint=1.57,output_limits=(-2, 2),sample_time=sample_time,))
        self.controller_list.append( # (4) Wrist 1 Joint
            PID(7 * p_scale,0.0 * i_scale,0.1 * d_scale,setpoint=-1.57,output_limits=(-1, 1),sample_time=sample_time,)) 
        self.controller_list.append( # (5) Wrist 2 Joint
            PID(5 * p_scale,0.0 * i_scale,0.1 * d_scale,setpoint=-1.57,output_limits=(-1, 1),sample_time=sample_time,))
        self.controller_list.append( # (6) Wrist 3 Joint
            PID(5 * p_scale,0.0 * i_scale,0.1 * d_scale,setpoint=0.0,output_limits=(-1, 1),sample_time=sample_time,))
        self.controller_list.append(  # (7) Gripper Joint
            PID(2.5 * p_scale,i_gripper,0.00 * d_scale,setpoint=0.0,output_limits=(-1, 1),sample_time=sample_time,))
        self.current_target_joint_values = [
            self.controller_list[i].setpoint for i in range(7)   #len(self.sim.data.ctrl)
        ]
        self.current_target_joint_values = np.array(self.current_target_joint_values)
        self.current_output = [controller(0) for controller in self.controller_list]
        self.actuators = []
        for i in range(7):   #len(self.sim.data.ctrl)
            item = [i, self.sim.model.actuator_id2name(i)]
            item.append(self.sim.model.actuator_trnid[i][0])
            item.append(self.sim.model.joint_id2name(self.sim.model.actuator_trnid[i][0]))
            item.append(self.controller_list[i])
            self.actuators.append(item)
    # ***************바꾼부분: 양팔 (0903)********************
    def create_lists_2(self):
        self.controller_list_2 = []
        sample_time = 0.0001
        p_scale = 3
        i_scale = 0.0
        i_gripper = 0
        d_scale = 0.1
        self.controller_list_2.append(  # (1) Shoulder Pan Joint
            PID(7 * p_scale,0.0 * i_scale,1.1 * d_scale,setpoint=0,output_limits=(-2, 2),sample_time=sample_time,))
        self.controller_list_2.append(  # (2) Shoulder Lift Joint
            PID(10 * p_scale,0.0 * i_scale,1.0 * d_scale,setpoint=-1.57,output_limits=(-2, 2),sample_time=sample_time,))
        self.controller_list_2.append( # (3) Elbow Joint
            PID(5 * p_scale,0.0 * i_scale,0.5 * d_scale,setpoint=1.57,output_limits=(-2, 2),sample_time=sample_time,))
        self.controller_list_2.append( # (내가추가) (3.5) Elbow2 Joint
            PID(5 * p_scale,0.0 * i_scale,0.5 * d_scale,setpoint=1.57,output_limits=(-2, 2),sample_time=sample_time,))
        self.controller_list_2.append( # (4) Wrist 1 Joint
            PID(7 * p_scale,0.0 * i_scale,0.1 * d_scale,setpoint=-1.57,output_limits=(-1, 1),sample_time=sample_time,)) 
        self.controller_list_2.append( # (5) Wrist 2 Joint
            PID(5 * p_scale,0.0 * i_scale,0.1 * d_scale,setpoint=-1.57,output_limits=(-1, 1),sample_time=sample_time,))
        self.controller_list_2.append( # (6) Wrist 3 Joint
            PID(5 * p_scale,0.0 * i_scale,0.1 * d_scale,setpoint=0.0,output_limits=(-1, 1),sample_time=sample_time,))
        self.controller_list_2.append(  # (7) Gripper Joint
            PID(2.5 * p_scale,i_gripper,0.00 * d_scale,setpoint=0.0,output_limits=(-1, 1),sample_time=sample_time,))
        self.current_target_joint_values_2 = [
            self.controller_list_2[i].setpoint for i in range(7)
        ]
        self.current_target_joint_values_2 = np.array(self.current_target_joint_values_2)
        self.current_output_2 = [controller(0) for controller in self.controller_list_2]
        self.actuators_2 = []
        for i in range(7, len(self.sim.data.ctrl)):
            item = [i, self.sim.model.actuator_id2name(i)]
            item.append(self.sim.model.actuator_trnid[i][0])
            item.append(self.sim.model.joint_id2name(self.sim.model.actuator_trnid[i][0]))
            item.append(self.controller_list_2[i-7])
            self.actuators_2.append(item)

    # ***************바꾼부분: 그랩********************
    def move_group_to_joint_target(
        self,
        group="All",
        target=None,
        tolerance=0.1,
        max_steps=10000,
        marker=False,
        render=True,
        quiet=False,
    ):
        try:
            assert group in self.groups.keys(), "No group with name {} exists!".format(group)
            if target is not None:
                assert len(target) == len(self.groups[group]), "Mismatching target dimensions for group {}!".format(group)
            ids = self.groups[group]
            steps = 1
            result = ""
            self.reached_target = False
            deltas = np.zeros(7)
            if target is not None:
                for i, v in enumerate(ids):
                    self.current_target_joint_values[v] = target[i]  #이제부터 cur_target_joint_val은 그냥 target임!
            for j in range(7):
                self.actuators[j][4].setpoint = self.current_target_joint_values[j]

            while not self.reached_target:
                current_joint_values = self.sim.data.qpos[self.actuated_joint_ids]
                # self.get_image_data(width=200, height=200, show=True)
                for j in range(7):
                    self.current_output[j] = self.actuators[j][4](current_joint_values[j])
                    self.sim.data.ctrl[j] = self.current_output[j]
                for i in ids: 
                    deltas[i] = abs(self.current_target_joint_values[i] - current_joint_values[i])
                if steps % 1000 == 0 and target is not None and not quiet:
                    print("Moving group {} to joint target! Max. delta: {}, Joint: {}".format(
                            group, max(deltas), self.actuators[np.argmax(deltas)][3]))
                if max(deltas) < tolerance:
                    if target is not None and not quiet:
                        print(colored("Joint values for group {} within requested tolerance! ({} steps)".format(
                                    group, steps),color="green",attrs=["bold"],))
                    result = "success"
                    self.reached_target = True
                    # break
                if steps > max_steps:
                    if not quiet:
                        print(colored("Max number of steps reached: {}".format(max_steps),color="red",attrs=["bold"],))
                        print("Deltas: ", deltas)
                    result = "max. steps reached: {}".format(max_steps)
                    break
                self.sim.step()
                if render:
                    camera_id = self.sim.model.camera_name2id(self.train_camera)
                    width = self.randomization_kwargs['camera_config']['width']
                    height = self.randomization_kwargs['camera_config']['height']
                    #************바꾼부분: 시각화******************
                    # self.viewer.render(width, height, camera_id)
                    # self.viewer.render(self.sim)
                    # self.viewer.render(self)
                steps += 1
            self.last_movement_steps = steps
            return result
        except Exception as e:
            print(e)
            print("Could not move to requested joint target.")
    #*********바꾼부분: 양팔 (0903) ********************
    def move_group_to_joint_target_2(
        self,
        group="All",
        target=None,
        tolerance=0.1,
        max_steps=10000,
        marker=False,
        render=True,
        quiet=False,
    ):
        try:
            assert group in self.groups_2.keys(), "No group with name {} exists!".format(group)
            if target is not None:
                assert len(target) == len(self.groups_2[group]), "Mismatching target dimensions for group {}!".format(group)
            ids = self.groups_2[group]
            steps = 1
            result = ""
            self.reached_target_2 = False
            deltas = np.zeros(7)
            if target is not None:
                for i, v in enumerate(ids):
                    self.current_target_joint_values_2[v] = target[i]  #이제부터 cur_target_joint_val은 그냥 target임!
            for j in range(7):
                self.actuators_2[j][4].setpoint = self.current_target_joint_values_2[j]

            while not self.reached_target:
                current_joint_values_2 = self.sim.data.qpos[self.actuated_joint_ids_2]
                for j in range(7):
                    self.current_output_2[j] = self.actuators_2[j][4](current_joint_values_2[j])
                    self.sim.data.ctrl[j+7] = self.current_output_2[j]
                for i in ids: 
                    deltas[i] = abs(self.current_target_joint_values_2[i] - current_joint_values_2[i])
                if steps % 1000 == 0 and target is not None and not quiet:
                    print("Moving group {} to joint target! Max. delta: {}, Joint: {}".format(
                            group, max(deltas), self.actuators_2[np.argmax(deltas)][3]))
                if max(deltas) < tolerance:
                    if target is not None and not quiet:
                        print(colored("Joint values for group {} within requested tolerance! ({} steps)".format(
                                    group, steps),color="green",attrs=["bold"],))
                    result = "success"
                    self.reached_target_2 = True
                    # break
                if steps > max_steps:
                    if not quiet:
                        print(colored("Max number of steps reached: {}".format(max_steps),color="red",attrs=["bold"],))
                        print("Deltas: ", deltas)
                    result = "max. steps reached: {}".format(max_steps)
                    break
                self.sim.step()
                steps += 1
            self.last_movement_steps = steps
            return result
        except Exception as e:
            print(e)
            print("Could not move to requested joint target.")

    #*************바꾼부분: 그랩********************
    def open_gripper(self, **kwargs):
        self.sim.model.body_pos[self.rightfinger_id]=(-0.1, 0, 0.063)
        self.sim.model.body_pos[self.leftfinger_id]=(0.1, 0, 0.063)
        return self.move_group_to_joint_target(group="Gripper", target=[0.2], max_steps=1000, tolerance=0.05, **kwargs)
    def close_gripper(self, **kwargs):
        self.sim.model.body_pos[self.rightfinger_id]=(0, 0, 0.063)
        self.sim.model.body_pos[self.leftfinger_id]=(0, 0, 0.063)
        return self.move_group_to_joint_target(group="Gripper", target=[-0.4], tolerance=0.01, **kwargs)
    
    #*************바꾼부분: 양팔(0829)********************
    def open_gripper_2(self, **kwargs):
        self.sim.model.body_pos[self.rightfinger_id_2]=(-0.1, 0, 0.063)
        self.sim.model.body_pos[self.leftfinger_id_2]=(0.1, 0, 0.063)
        return self.move_group_to_joint_target_2(group="Gripper_2", target=[0.2], max_steps=1000, tolerance=0.05, **kwargs)
    def close_gripper_2(self, **kwargs):
        self.sim.model.body_pos[self.rightfinger_id_2]=(0, 0, 0.063)
        self.sim.model.body_pos[self.leftfinger_id_2]=(0, 0, 0.063)
        return self.move_group_to_joint_target_2(group="Gripper_2", target=[-0.4], tolerance=0.01, **kwargs)

    def get_model_kwargs(self, randomize, rownum=None):
        df = pd.read_csv(self.model_kwargs_path)  # data/model_param.csv

        model_kwargs = copy.deepcopy(mujoco_model_kwargs.BASE_MODEL_KWARGS)

        if randomize:
            choice = np.random.randint(0, df.shape[0] - 1)
            model_kwargs_row = df.iloc[choice]

        if rownum is not None:
            model_kwargs_row = df.iloc[rownum]

        for col in model_kwargs.keys():
            model_kwargs[col] = model_kwargs_row[col]

        return model_kwargs

    def build_xml_kwargs_and_numerical_values(self, randomize, rownum=None):
        model_kwargs = self.get_model_kwargs(
            randomize=randomize, rownum=rownum)

        model_numerical_values = []
        for key in model_kwargs.keys():
            value = model_kwargs[key]
            if type(value) in [int, float]:
                model_numerical_values.append(value)

        model_kwargs['floor_material_name'] = "floor_real_material"
        model_kwargs['table_material_name'] = "table_real_material"
        model_kwargs['cloth_material_name'] = "wipe_real_material"
        if self.randomization_kwargs['materials_randomization']:
            model_kwargs['floor_material_name'] = np.random.choice(
                ["floor_real_material", "floor_material"])
            model_kwargs['table_material_name'] = np.random.choice(
                ["table_real_material", "table_material"])
            model_kwargs['cloth_material_name'] = np.random.choice(["bath_real_material", "bath_2_real_material", "kitchen_real_material", "kitchen_2_real_material",
                                                                   "wipe_real_material", "wipe_2_real_material", "cloth_material", "white_real_material", "blue_real_material", "orange_real_material"])

        # General
        model_kwargs['timestep'] = self.timestep
        model_kwargs['lights_randomization'] = self.randomization_kwargs['lights_randomization']
        model_kwargs['materials_randomization'] = self.randomization_kwargs['materials_randomization']
        model_kwargs['train_camera_fovy'] = (self.randomization_kwargs['camera_config']
                                             ['fovy_range'][0] + self.randomization_kwargs['camera_config']['fovy_range'][1])/2
        model_kwargs['num_lights'] = 1

        model_kwargs['geom_spacing'] = (
            self.randomization_kwargs['cloth_size'] - 2*model_kwargs['geom_size']) / 8
        model_kwargs['offset'] = 4 * model_kwargs['geom_spacing']

        # Appearance
        appearance_choices = mujoco_model_kwargs.appearance_kwarg_choices
        appearance_ranges = mujoco_model_kwargs.appearance_kwarg_ranges
        for key in appearance_choices.keys():
            model_kwargs[key] = np.random.choice(appearance_choices[key])
        for key in appearance_ranges.keys():
            values = appearance_ranges[key]
            model_kwargs[key] = np.random.uniform(values[0], values[1])

        # Camera fovy
        if self.randomization_kwargs['camera_position_randomization']:
            model_kwargs['train_camera_fovy'] = np.random.uniform(
                self.randomization_kwargs['camera_config']['fovy_range'][0], self.randomization_kwargs['camera_config']['fovy_range'][1])

        min_corner = 0
        max_corner = 8
        self.max_corner_name = f"B{max_corner}_{max_corner}"
        
        self.mid_corner_index = 4
        mid = int(max_corner / 2)
        self.corner_index_mapping = {"0": f"S{min_corner}_{max_corner}", "1": f"S{max_corner}_{max_corner}",
                                     "2": f"S{min_corner}_{min_corner}", "3": f"S{max_corner}_{min_corner}"}
        
        #****************바꾼부분: 옷 (1017) ****************#
        min_all = 0
        max_all = 8#19
        mid_all = int(max_all / 2)
        self.cloth_site_names = []
        for i in range(max_all+1):  #[min_all, mid_all, max_all]
            for j in range(max_all+1):   #[min_all, mid_all, max_all]
                # self.cloth_site_names.append(f"S{i}_{j}")  #원래
                self.cloth_site_names.append(f"B{i}_{j}") #0923
        
        #*****************바꾼부분: 보상*****************
        # self.task_reward_function = reward_calculation.get_task_reward_function(
        #     self.constraints, self.single_goal_dim, self.sparse_dense, self.success_reward, self.fail_reward, self.extra_reward)
        self.task_reward_function = reward_calculation_unfolding.get_unfolding_reward_function()
        # self.task_reward_function = reward_calculation_unfolding.get_unfolding_with_motion_reward_function(self.constraints, self.single_goal_dim, self.sparse_dense, self.success_reward, self.fail_reward, self.extra_reward)

        return model_kwargs, model_numerical_values

    #*****************바꾼부분: 모션플래닝*****************
    def build_motion_distance_and_constraints(self):
        cloth_positions_list = list(self.get_cloth_position_W().values())
        self.motion_distance = cloth_positions_list - self.get_ee_position_W()
        #바꾼부분: 옷 (1017) ******
        self.constraints = task_definitions.constraints["random_cloth"](0, 4, 8, self.motion_distance, -0, -0)
        # self.constraints = task_definitions.constraints["random_cloth"](0, 10, 19, self.motion_distance, -0, -0)
    
    #*********바꾼부분: 시각화*******************
    def start_without_ld_preload(self):
        subprocess.run("unset LD_PRELOAD", shell=True, check=True)
        print(os.environ.get('LD_PRELOAD'))
        print("LD_PRELOAD가 해제되었습니다.")
    def return_ld_preload(self):
        command = f"export LD_PRELOAD={self.original_ld_preload}"
        subprocess.run(command, shell=True, check=True)
        print(os.environ.get('LD_PRELOAD'))
        print("LD_PRELOAD가 복구되었습니다.")

    def setup_viewer(self):
        if self.has_viewer:
            if not self.viewer is None:
                del self.viewer
            
            #*****************바꾼부분: 시각화*****************
            if self.glfw_window:
                self.viewer = mujoco_py.MjRenderContextWindow(self.sim)
            else:
                # self.start_without_ld_preload()
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
                # self.return_ld_preload()

            self.viewer.vopt.geomgroup[0] = 0
            self.viewer.vopt.geomgroup[1] = 1

    def dump_xml_models(self):
        with open(f"{self.save_folder}/compiled_mujoco_model_no_inertias.xml", "w") as f:
            self.sim.save(f, format='xml', keep_inertials=False)

        with open(f"{self.save_folder}/compiled_mujoco_model_with_intertias.xml", "w") as f:
            self.sim.save(f, format='xml', keep_inertials=True)

    def reset_camera(self):
        lookat_offset = np.zeros(3)
        #*******바꾼부분: 카메라 (0908) ***************
        # self.train_camera = self.randomization_kwargs['camera_type']
        # if self.train_camera == "all":
        #     self.train_camera = np.random.choice(["up", "front", "side"])
        self.train_camera = "front"

        if self.randomization_kwargs['lookat_position_randomization']:
            radius = self.randomization_kwargs['lookat_position_randomization_radius']
            lookat_offset[0] += np.random.uniform(-radius, radius)
            lookat_offset[1] += np.random.uniform(-radius, radius)

        des_cam_look_pos = self.sim.data.get_body_xpos(
            f"B{self.mid_corner_index}_{self.mid_corner_index}").copy() + lookat_offset
        self.sim.data.set_mocap_pos("lookatbody", des_cam_look_pos)
    
    def add_mocap_to_xml(self, xml):
        dom = minidom.parseString(xml)
        for subelement in dom.getElementsByTagName("body"):

            #******바꾼부분: 그랩(1001)********************
            # ver 0
            # if subelement.getAttribute("name") == self.max_corner_name:
            #     subelement.setAttribute("mocap", "true")
            #     for child_node in subelement.childNodes:
            #         if child_node.nodeType == 1:
            #             if child_node.tagName == "joint":
            #                 subelement.removeChild(child_node)
            # 버전1
            if subelement.getAttribute("name").startswith('B'):
                subelement.setAttribute("mocap", "true")
                # #******바꾼부분: 그랩 (1001) *********
                # subelement.setAttribute("pos", "0 0 0")
                for child_node in subelement.childNodes:
                    if child_node.nodeType == 1:
                        if child_node.tagName == "joint":
                            subelement.removeChild(child_node)

        return remove_empty_lines(dom.toprettyxml(indent=" " * 4))
    #******바꾼부분: 그랩 (1009) 버전2 ******
    def add_new_mocap_to_xml(self, xml, body_name, mocap_value, body_name2=None, mocap_value2=None):
        dom = minidom.parseString(xml)
        for body in dom.getElementsByTagName("body"):
            if body.getAttribute("name") == body_name:
                body.setAttribute("mocap", str(mocap_value).lower())  # True/False로 설정
                for child_node in body.childNodes:
                    if child_node.nodeType == 1:
                        if child_node.tagName == "joint":
                            body.removeChild(child_node)
                break
        if body_name2 != None:
            for body in dom.getElementsByTagName("body"):
                if body.getAttribute("name") == body_name2:
                    body.setAttribute("mocap", str(mocap_value2).lower())  # True/False로 설정
                    for child_node in body.childNodes:
                        if child_node.nodeType == 1:
                            if child_node.tagName == "joint":
                                body.removeChild(child_node)
                    break
        return dom.toxml()
    
    def setup_initial_state_and_sim(self, model_kwargs):
        if not self.mjpy_model is None:
            del self.mjpy_model
        if not self.sim is None:
            del self.sim
        temp_xml_1 = self.template_renderer.render_template(
            "arena.xml", **model_kwargs)
        temp_model = mujoco_py.load_model_from_xml(temp_xml_1)
        temp_xml_2 = copy.deepcopy(temp_model.get_xml())
        del temp_model
        del temp_xml_1

        #*******바꾼부분: 그랩 (1009) 버전1 버전2***********
        # temp_xml_2 = self.add_mocap_to_xml(temp_xml_2)
        # temp_xml_2 = self.add_new_mocap_to_xml(temp_xml_2, "B0_0", "true")
        #*******바꾼부분: 그랩 (1017)**********
        # s_1=random.choice([0, 8])
        # s_2=random.choice([0, 8])
        # self.singlemocap=f"B{s_1}_{s_2}"
        # while 1:
        #     p_1=random.choice(range(0,9))
        #     p_2=random.choice(range(0,9))
        #     self.policymocap=f"B{p_1}_{p_2}"
        #     if self.policymocap!=self.singlemocap:
        #         break
        self.singlemocap="B0_8"
        self.policymocap="B5_4"

        print("single: ", self.singlemocap, " policy: ", self.policymocap)
        temp_xml_2 = self.add_new_mocap_to_xml(temp_xml_2, self.singlemocap, "true", self.policymocap, "true")
        # temp_xml_2 = self.add_new_mocap_to_xml(temp_xml_2, "B0_0", "true")

        self.mjpy_model = mujoco_py.load_model_from_xml(temp_xml_2)

        #*******바꾼부분: 그랩 (1001) 버전2***********
        del temp_xml_2

        gc.collect()
        self.sim = mujoco_py.MjSim(self.mjpy_model)
        self.setup_viewer()

        #바꾼부분: 그랩 (1017)
        # body_id = self.sim.model.body_name2id(self.max_corner_name)
        # self.ee_mocap_id = self.sim.model.body_mocapid[body_id]

        self.joint_indexes = [self.sim.model.joint_name2id(joint) for joint in self.joints]
        self.joint_pos_addr = [self.sim.model.get_joint_qpos_addr(joint) for joint in self.joints]
        self.joint_vel_addr = [self.sim.model.get_joint_qvel_addr(joint) for joint in self.joints]
        self.ee_site_adr = mujoco_py.functions.mj_name2id(  #Get id of object with specified name
            self.sim.model, 6, "grip_site")
        
        #*******바꾼부분: 양팔 (0829)********
        body_id_2 = self.sim.model.body_name2id(self.max_corner_name)
        self.ee_mocap_id_2 = self.sim.model.body_mocapid[body_id_2]
        self.joint_indexes_2 = [self.sim.model.joint_name2id(joint_2) for joint_2 in self.joints_2]
        self.joint_pos_addr_2 = [self.sim.model.get_joint_qpos_addr(joint_2) for joint_2 in self.joints_2]
        self.joint_vel_addr_2 = [self.sim.model.get_joint_qvel_addr(joint_2) for joint_2 in self.joints_2]
        self.ee_site_adr_2 = mujoco_py.functions.mj_name2id(self.sim.model, 6, "grip_site_2")

        # Sets robot to initial qpos and resets osc values
        self.set_robot_initial_joints()
        self.reset_osc_values()

        self.update_osc_values()
        #*******바꾼부분: 양팔 (0829)********
        self.update_osc_values_2()

        for _ in range(30):
            mujoco_py.functions.mj_step1(self.sim.model, self.sim.data)
            self.sim.data.qfrc_applied[self.joint_vel_addr] = self.sim.data.qfrc_bias[self.joint_vel_addr]
            #*******바꾼부분: 양팔 (0829)********
            self.sim.data.qfrc_applied[self.joint_vel_addr_2] = self.sim.data.qfrc_bias[self.joint_vel_addr_2]

            mujoco_py.functions.mj_step2(self.sim.model, self.sim.data)

        #*************바꾼부분***********************
        self.initial_state = copy.deepcopy(self.sim.get_state())
        self.initial_qfrc_applied = self.sim.data.qfrc_applied[self.joint_vel_addr].copy()
        self.initial_qfrc_bias = self.sim.data.qfrc_bias[self.joint_vel_addr].copy()
        #*******바꾼부분: 양팔 (0829)********
        self.initial_qfrc_applied_2 = self.sim.data.qfrc_applied[self.joint_vel_addr_2].copy()
        self.initial_qfrc_bias_2 = self.sim.data.qfrc_bias[self.joint_vel_addr_2].copy()

        # #****바꾼부분: 그랩 (1001) 버전1******
        # self.set_cloth_mocapid()

        #*******바꾼부분: 그랩 (1001) 버전2*******
        # return temp_xml_2

    #******** 바꾼부분: 그랩 (1009) 버전2************
    def update_sim_with_new_mocap(self, xml, state, body_name, mocap_value, body_name2=None, mocap_value2=None, last=False):
        if body_name2 !=None:
            if mocap_value=="true":
                xml = self.add_new_mocap_to_xml(xml, body_name, mocap_value, body_name2, mocap_value2)
        else:
            if mocap_value=="true":
                xml = self.add_new_mocap_to_xml(xml, body_name, mocap_value)
        self.mjpy_model = mujoco_py.load_model_from_xml(xml)
        del xml
        gc.collect()
        self.sim = mujoco_py.MjSim(self.mjpy_model)
        self.setup_viewer()

        #변수들 새롭게 설정
        body_id = self.sim.model.body_name2id(self.max_corner_name)
        self.ee_mocap_id = self.sim.model.body_mocapid[body_id]
        self.joint_indexes = [self.sim.model.joint_name2id(joint) for joint in self.joints]
        self.joint_pos_addr = [self.sim.model.get_joint_qpos_addr(joint) for joint in self.joints]
        self.joint_vel_addr = [self.sim.model.get_joint_qvel_addr(joint) for joint in self.joints]
        self.ee_site_adr = mujoco_py.functions.mj_name2id(  #Get id of object with specified name
            self.sim.model, 6, "grip_site")
        body_id_2 = self.sim.model.body_name2id(self.max_corner_name)
        self.ee_mocap_id_2 = self.sim.model.body_mocapid[body_id_2]
        self.joint_indexes_2 = [self.sim.model.joint_name2id(joint_2) for joint_2 in self.joints_2]
        self.joint_pos_addr_2 = [self.sim.model.get_joint_qpos_addr(joint_2) for joint_2 in self.joints_2]
        self.joint_vel_addr_2 = [self.sim.model.get_joint_qvel_addr(joint_2) for joint_2 in self.joints_2]
        self.ee_site_adr_2 = mujoco_py.functions.mj_name2id(self.sim.model, 6, "grip_site_2")
        self.set_robot_initial_joints()
        self.reset_osc_values()
        self.update_osc_values()
        self.update_osc_values_2()
        for _ in range(30):
            mujoco_py.functions.mj_step1(self.sim.model, self.sim.data)
            self.sim.data.qfrc_applied[self.joint_vel_addr] = self.sim.data.qfrc_bias[self.joint_vel_addr]
            self.sim.data.qfrc_applied[self.joint_vel_addr_2] = self.sim.data.qfrc_bias[self.joint_vel_addr_2]
            mujoco_py.functions.mj_step2(self.sim.model, self.sim.data)
        self.initial_state = copy.deepcopy(self.sim.get_state())
        self.initial_qfrc_applied = self.sim.data.qfrc_applied[self.joint_vel_addr].copy()
        self.initial_qfrc_bias = self.sim.data.qfrc_bias[self.joint_vel_addr].copy()
        self.initial_qfrc_applied_2 = self.sim.data.qfrc_applied[self.joint_vel_addr_2].copy()
        self.initial_qfrc_bias_2 = self.sim.data.qfrc_bias[self.joint_vel_addr_2].copy()

        # # 이거 실행시 index 254 is out of bounds for axis 0 with size 254, Could not move to requested joint target.
        # self.open_gripper()
        # self.open_gripper_2()

        # 바꾼부분: 그랩 (1003) 직전 state를 전달
        if not last:
            self.sim.set_state(state)
            self.sim.forward()
    
    def set_robot_initial_joints(self):
        for j, joint in enumerate(self.joints):
            self.sim.data.set_joint_qpos(joint, self.initial_qpos[j])
        
        #*******바꾼부분: 양팔 (0829)***********
        for j_2, joint_2 in enumerate(self.joints_2):
            self.sim.data.set_joint_qpos(joint_2, self.initial_qpos_2[j_2])

        for _ in range(10):
            self.sim.forward()

    def run_controller(self):
        tau = osc_binding.step_controller(self.initial_O_T_EE,
                                          self.O_T_EE,
                                          self.initial_joint_osc,
                                          self.joint_pos_osc,
                                          self.joint_vel_osc,
                                          self.mass_matrix_osc,
                                          self.jac_osc,
                                          np.zeros(7),
                                          self.tau_J_d_osc,
                                          self.desired_pos_ctrl_W,
                                          np.zeros(3),
                                          self.delta_tau_max,
                                          self.kp,
                                          self.kp,
                                          self.damping_ratio
                                          )
        torques = tau.flatten()
        return torques
    #******바꾼부분: 양팔 (0829) *********
    def run_controller_2(self):
        tau = osc_binding.step_controller(self.initial_O_T_EE_2,
                                          self.O_T_EE_2,
                                          self.initial_joint_osc_2,
                                          self.joint_pos_osc_2,
                                          self.joint_vel_osc_2,
                                          self.mass_matrix_osc_2,
                                          self.jac_osc_2,
                                          np.zeros(7),
                                          self.tau_J_d_osc_2,
                                          self.desired_pos_ctrl_W_2,
                                          np.zeros(3),
                                          self.delta_tau_max,
                                          self.kp,
                                          self.kp,
                                          self.damping_ratio
                                          )
        torques = tau.flatten()
        return torques
    
    #********바꾼부분: 모션플래닝 (0909) ******
    def plan_trajectory(self, start_pos, end_pos, num_points):
        start = np.array(start_pos)
        end = np.array(end_pos)
        t = np.linspace(0, 1, num_points)
        path = np.outer(1 - t, start) + np.outer(t, end)
        return path
    #********바꾼부분: 모션플래닝 (0920) ************8
    def plan_joint_trajectory(self, start_joint, end_joint, num_points):
        start = np.array(start_joint)
        end = np.array(end_joint)
        t = np.linspace(0, 1, num_points)
        path = np.outer(1 - t, start) + np.outer(t, end)
        return path
    
    def step_env(self, q_pos=None):  #********바꾼부분: 양팔 (0909)**********
        #********바꾼부분: 모션플래닝 (0909)**********
        # self.sim.data.mocap_pos[self.ee_mocap_id][:] = self.sim.data.get_geom_xpos(
        #     "grip_geom")
        mujoco_py.functions.mj_step1(self.sim.model, self.sim.data)
        #********바꾼부분: 양팔 (0909)**********
        if q_pos is not None:
            self.update_osc_values(q_pos)
        else:
            self.update_osc_values()
        tau = self.run_controller()
        self.sim.data.qfrc_applied[self.joint_vel_addr] = self.sim.data.qfrc_bias[self.joint_vel_addr] + tau
        mujoco_py.functions.mj_step2(self.sim.model, self.sim.data)

    #********바꾼부분: 양팔 (0829)**********
    def step_env_2(self, q_pos=None):  #********바꾼부분: 양팔 (0909)**********
        # self.sim.data.mocap_pos[self.ee_mocap_id_2][:] = self.sim.data.get_geom_xpos("grip_geom_2")
        mujoco_py.functions.mj_step1(self.sim.model, self.sim.data)
        #********바꾼부분: 양팔 (0909)**********
        if q_pos is not None:
            self.update_osc_values_2(q_pos)
        else:
            self.update_osc_values_2()
        tau_2 = self.run_controller_2()
        self.sim.data.qfrc_applied[self.joint_vel_addr_2] = self.sim.data.qfrc_bias[self.joint_vel_addr_2] + tau_2
        mujoco_py.functions.mj_step2(self.sim.model, self.sim.data)
    
    #*******바꾼부분: 모션플래닝 (0909) ***********
    def set_joint_positions(self, q_pos):
        for j, joint in enumerate(self.joints):
            self.sim.data.set_joint_qpos(joint, q_pos[j])
        # for _ in range(10):
        #     self.sim.forward()
    def set_joint_positions_2(self, q_pos):
        for j_2, joint_2 in enumerate(self.joints_2):
            self.sim.data.set_joint_qpos(joint_2, q_pos[j_2])
        # for _ in range(10):
        #     self.sim.forward()

    #****바꾼부분: 그랩 (1003) 버전2 ********
    def remove_body_qposvel(self, state, body_id):
        qpos_start = self.sim.model.jnt_qposadr[self.sim.model.body_jntadr[body_id]]
        qpos_end = qpos_start + 3
        new_qpos = np.delete(state.qpos, slice(qpos_start, qpos_end))
        new_qvel = np.delete(state.qvel, slice(qpos_start, qpos_end))
        new_state = mujoco_py.MjSimState(state.time, new_qpos, new_qvel, state.act, state.udd_state)
        return new_state, qpos_start, qpos_end
    #****바꾼부분: 그랩 (1003) 버전2 ********
    def add_body_qposvel(self, receiver, giver, s_idx, e_idx, pos):
        new_qpos=receiver.qpos
        new_qvel=receiver.qvel
        new_qpos[:s_idx] = giver.qpos[:s_idx]
        new_qpos[s_idx:e_idx] = pos
        new_qpos[e_idx:] = giver.qpos[s_idx:]
        new_qvel[:s_idx] = giver.qvel[:s_idx]
        new_qvel[e_idx:] = giver.qvel[s_idx:]
        new_state = mujoco_py.MjSimState(giver.time, new_qpos, new_qvel, giver.act, giver.udd_state)
        return new_state
    
    #*****그랩 (1017)**********
    def get_surroundings(self, grasp_cloth):
        first=int(grasp_cloth.split('_')[0][1:]) #B4_4에서 첫번째 4
        second=int(grasp_cloth.split('_')[1]) #B4_4에서 두번째 4
        valid_range = range(0, 9)
        candidates = [
            (first - 1, second),   # 왼쪽
            (first + 1, second),   # 오른쪽
            (first, second - 1),   # 아래쪽
            (first, second + 1),   # 위쪽
            (first - 1, second - 1), # 왼쪽 아래 대각선
            (first - 1, second + 1), # 왼쪽 위 대각선
            (first + 1, second - 1), # 오른쪽 아래 대각선
            (first + 1, second + 1)  # 오른쪽 위 대각선
        ]
        return [f"B{f}_{s}" for f, s in candidates if f in valid_range and s in valid_range]
    def get_avg_position(self, surrounding_cloths):
        surrounding_positions = [
            self.get_cloth_position_W()[cloth] for cloth in surrounding_cloths 
        ]
        return np.mean(surrounding_positions, axis=0)

    #*******바꾼부분: 양팔 (0920) ************
    def step_2(self):
        camera_id = self.sim.model.camera_name2id(
            self.train_camera)
        width = self.randomization_kwargs['camera_config']['width']
        height = self.randomization_kwargs['camera_config']['height']
        # #reset
        # self.sim.reset()
        # self.sim.set_state(self.initial_state)

        # self.sim.data.qfrc_applied[self.joint_vel_addr] = self.initial_qfrc_applied
        # self.sim.data.qfrc_bias[self.joint_vel_addr] = self.initial_qfrc_bias
        # #*******바꾼부분: 양팔 (0829)************
        # self.sim.data.qfrc_applied[self.joint_vel_addr_2] = self.initial_qfrc_applied_2
        # self.sim.data.qfrc_bias[self.joint_vel_addr_2] = self.initial_qfrc_bias_2

        # self.sim.forward()  # BODY POSITIONS CORRECT
        # self.reset_camera()
        # self.sim.forward()  # CAMERA CHANGES CORRECT
        # self.reset_osc_values()

        # self.update_osc_values()
        # self.relative_origin = self.get_ee_position_W()
        # #********바꾼부분: 양팔 (0829)*********
        # self.update_osc_values_2()
        # self.relative_origin_2 = self.get_ee_position_W_2()

        # self.goal, self.goal_noise = self.sample_goal_I()
        # if not self.viewer is None:
        #     del self.viewer._markers[:]
        # self.episode_ee_close_steps = 0

        # self.open_gripper()
        # #********바꾼부분: 양팔 (0829)*********
        # self.open_gripper_2()

        # #****바꾼부분: 그랩 (1001) 버전1******
        # self.set_cloth_mocapid()

        #*****바꾼부분: 그랩 (1009)*******
        # self.grasp_cloth_2=list(self.get_cloth_position_W().items())[self.a2_index][0] #B0_0 형식
        body_id = self.sim.model.body_name2id(self.singlemocap) #0~110 숫자

        #*******바꾼부분: 그랩 (1009) 버전2*******
        # state_0=self.sim.get_state()
        # state, startidx, endidx =self.remove_body_qposvel(state_0, body_id)
        # self.update_sim_with_new_mocap(self.model_xml, state, self.grasp_cloth_2, "true")
        self.ee_mocap_id_2 = self.sim.model.body_mocapid[body_id]

        #******바꾼부분: 그랩 (1001) 버전3********
        # self.sim.data.set_mocap_pos("singlemocap", self.action_2.copy())
        # singlemocap_id = self.sim.model.body_name2id("singlemocap")
        #******바꾼부분: 그랩 (1009) 버전3********
        # self.cloth_site_names.append("singlemocap")
        # ver (3)-1
        # self.sim.model.body_mocapid[body_id]=self.sim.model.body_mocapid[singlemocap_id]
        # self.sim.data.mocap_pos[self.sim.model.body_mocapid[body_id]] = self.sim.data.get_body_xpos(self.grasp_cloth_2)
        # self.sim.data.mocap_quat[self.sim.model.body_mocapid[body_id]] = self.sim.data.get_body_xquat(self.grasp_cloth_2)
        # self.ee_mocap_id_2 = self.sim.model.body_mocapid[body_id]
        # ver (3)-2
        # self.sim.data.mocap_pos[self.sim.model.body_mocapid[singlemocap_id]] = self.sim.data.get_body_xpos(self.grasp_cloth_2)
        # self.sim.data.mocap_quat[self.sim.model.body_mocapid[singlemocap_id]] = self.sim.data.get_body_xquat(self.grasp_cloth_2)
        # self.ee_mocap_id_2 = self.sim.model.body_mocapid[singlemocap_id]
        print("self.ee_mocap_id_2: ", self.ee_mocap_id_2)

        #*******바꾼부분: 그랩 (1016) ******
        body_i = self.sim.model.body_name2id(self.policymocap)
        ee_mocap_id = self.sim.model.body_mocapid[body_i]
        sur=self.get_surroundings(self.policymocap)
        print("sur: ", sur)

        #*******바꾼부분: 양팔 (0829)********
        #**********바꾼부분: 모션플래닝 (1019) ************
        # goal_2=self.action_2.copy()
        goal_2=list(self.get_cloth_position_W().items())[self.a2_index][1]
        j_pos_1=self.get_joint_positions() #np.array(positions)
        pos_1=self.get_ee_position_W()
        num_points=200
        trajectory_2=self.plan_trajectory(self.get_ee_position_W_2(), goal_2, num_points)
        for i in range(num_points):
            self.desired_pos_ctrl_W_2=trajectory_2[i]
            # self.set_joint_positions(j_pos_1) # 바꾼부분: 0920
            self.step_env_2()

            #******바꾼부분: 그랩 (0920) **********
            # if i>=num_points-1:
            #     #******바꾼부분: 그랩 (1010) 버전1 **********
            #     self.sim.data.mocap_pos[self.ee_mocap_id_2][:] = self.sim.data.get_geom_xpos("grip_geom_2")
            #     # for site in self.cloth_site_names:  #B0_0 형식
            #     #     if site==self.grasp_cloth_2:
            #     #         self.sim.data.mocap_pos[self.ee_mocap_id_2][:] = self.sim.data.get_geom_xpos("grip_geom_2")
            #     #     else:
            #     #         body_id = self.sim.model.body_name2id(site) #0~110 숫자
            #     #         mocap_id = self.sim.model.body_mocapid[body_id]
            #     #         self.sim.data.mocap_pos[mocap_id][:] = self.sim.data.get_body_xpos(site)
            #     #*****바꾼부분: 그랩 (1016)****
            #     self.sim.data.mocap_pos[ee_mocap_id][:] = self.get_avg_position(sur)
            if self.glfw_window:
                self.viewer.render(self.sim)
            else:
                self.viewer.render(width, height, camera_id)
        
        #****바꾼부분: 그랩 (1002) 이거 실행이 안됨******
        # self.close_gripper_2()
        #******바꾼부분: 그랩 (1010) 버전1 **********
        self.sim.data.mocap_pos[self.ee_mocap_id_2][:] = self.sim.data.get_geom_xpos("grip_geom_2")
        # for site in self.cloth_site_names:  #B0_0 형식
        #     if site==self.grasp_cloth_2:
        #         self.sim.data.mocap_pos[self.ee_mocap_id_2][:] = self.sim.data.get_geom_xpos("grip_geom_2")
        #     else:
        #         body_id = self.sim.model.body_name2id(site) #0~110 숫자
        #         mocap_id = self.sim.model.body_mocapid[body_id]
        #         self.sim.data.mocap_pos[mocap_id][:] = self.sim.data.get_body_xpos(site)
        #*****바꾼부분: 그랩 (1016)****
        self.sim.data.mocap_pos[ee_mocap_id][:] = self.get_avg_position(sur)
        if self.glfw_window:
            self.viewer.render(self.sim)
        else:
            self.viewer.render(width, height, camera_id)

        trajectory_2_2=self.plan_trajectory(self.get_ee_position_W_2(), goal_2+(0,0,0.4), num_points)
        for i in range(num_points):
            diff=trajectory_2_2[i]-trajectory_2_2[i-1]
            self.desired_pos_ctrl_W_2=trajectory_2_2[i]
            # self.set_joint_positions(j_pos_1) # 바꾼부분: 0920
            self.step_env_2()
            # self.set_joint_positions(j_pos_1) # 바꾼부분: 0920
            #******바꾼부분: 그랩 (1010) 버전1 **********
            self.sim.data.mocap_pos[self.ee_mocap_id_2][:] = self.sim.data.get_geom_xpos("grip_geom_2")
            # for site in self.cloth_site_names:  #B0_0 형식
            #     if site==self.grasp_cloth_2:
            #         self.sim.data.mocap_pos[self.ee_mocap_id_2][:] = self.sim.data.get_geom_xpos("grip_geom_2")
            #     else:
            #         body_id = self.sim.model.body_name2id(site) #0~110 숫자
            #         mocap_id = self.sim.model.body_mocapid[body_id]
            #         self.sim.data.mocap_pos[mocap_id][:] = self.sim.data.get_body_xpos(site)+diff
            #*****바꾼부분: 그랩 (1016)****
            self.sim.data.mocap_pos[ee_mocap_id][:] = self.get_avg_position(sur)
            if self.glfw_window:
                self.viewer.render(self.sim)
            else:
                self.viewer.render(width, height, camera_id)


        #*******바꾼부분: 그랩 (1003) 버전2*******
        # state_1=self.sim.get_state()
        # point_pos=self.sim.data.mocap_pos[self.ee_mocap_id_2][:]
        # state_3=self.add_body_qposvel(state_0, state_1, startidx, endidx, point_pos)
        # self.update_sim_with_new_mocap(self.model_xml, state_3, self.grasp_cloth_2, "false")

        self.goal, self.goal_noise = self.sample_goal_I()
        obs = self.get_obs()
        # print("이거 잘되나? get_xml")
        # self.xml_step2 = self.sim.model.get_xml()
        # print("이거나옴 성공. get_xml")
        return obs
    
    #********바꾼부분: 구조 (0902)************
    def step(self, action):
        camera_id = self.sim.model.camera_name2id(
            self.train_camera)
        width = self.randomization_kwargs['camera_config']['width']
        height = self.randomization_kwargs['camera_config']['height']
        num_points=200

        #******바꾼부분: 모션플래닝**************
        self.action=action.copy()

        raw_action = action.copy()
        action = raw_action*self.output_max

        image_obs_substep_idx_mean = self.image_obs_noise_mean * \
            (self.substeps-1)
        image_obs_substep_idx = int(np.random.normal(
            image_obs_substep_idx_mean, self.image_obs_noise_std))
        image_obs_substep_idx = np.clip(
            image_obs_substep_idx, 0, self.substeps-1)
        cosine_distance = compute_cosine_distance(
            self.previous_raw_action, raw_action)
        self.previous_raw_action = raw_action
        previous_desired_pos_step_W = self.desired_pos_step_W.copy()
        desired_pos_step_W = previous_desired_pos_step_W + action
        self.desired_pos_step_W = np.clip(desired_pos_step_W, self.min_absolute_W, self.max_absolute_W)

        #********바꾼부분: 모션플래닝 (0909) *********
        j_pos_2=self.get_joint_positions_2()

        # ******바꾼부분: 모션플래닝 (1003) *****
        def pre_goal():  # collision에 의해 막히는 걸 방지하기 위해, ee를 위로 올린 상태로 만들기
            goal_1=self.get_ee_position_W()+(-0.2,0,0.4)
            trajectory_pre=self.plan_trajectory(self.get_ee_position_W(), goal_1, num_points)
            for i in range(num_points):
                self.desired_pos_ctrl_W=trajectory_pre[i]
                self.step_env()
                if self.glfw_window:
                    self.viewer.render(self.sim)
                else:
                    self.viewer.render(width, height, camera_id)
            self.desired_pos_ctrl_W=goal_1
            self.step_env()
            print("잘 갔는가?1 self.get_ee_position_W(): ", self.get_ee_position_W(), "goal: ", goal_1)
        def goto_goal(goal, grip): # goal(action)로 가기
            trajectory=self.plan_trajectory(self.get_ee_position_W(), goal, num_points)
            for i in range(num_points):
                self.desired_pos_ctrl_W=trajectory[i]
                self.step_env()
                if i>=num_points-1:
                    if grip:
                        self.sim.data.mocap_pos[self.ee_mocap_id][:] = self.sim.data.get_geom_xpos("grip_geom")
                if self.glfw_window:
                    self.viewer.render(self.sim)
                else:
                    self.viewer.render(width, height, camera_id)
            self.desired_pos_ctrl_W=goal
            self.step_env()
            print("잘 갔는가?2 self.get_ee_position_W(): ", self.get_ee_position_W(), ", goal: ", goal)

        # ******바꾼부분: 액션 (1003) *****
        for idx, pos in enumerate(list(self.get_cloth_position_W().values())):
            distance = np.linalg.norm(np.array(self.action) - np.array(pos))
            if distance < 0.01:
                action_in_cloth=idx
            else:
                # action_in_cloth=None
                action_in_cloth=14
        if action_in_cloth==None:
            pre_goal()
            goto_goal(self.action, grip=False)
            print("FAIL")
        else:
            # **바꾼부분: 그랩 (1010) ********
            # grasp_cloth=list(self.get_cloth_position_W().items())[action_in_cloth][0] #B0_0 형식
            body_id = self.sim.model.body_name2id(self.policymocap) #0~110 숫자
            for d in enumerate(list(self.get_cloth_position_W().items())):
                if self.policymocap in d[1][0]:
                    cloth_pos = d[1][1]
            cloth_pos = list(self.get_cloth_position_W().items())[self.a_index][1]
            print("cloth_pos: ", cloth_pos)
            #****바꾼부분: 그랩 *****
            # state_0=self.sim.get_state()
            # state, startidx, endidx =self.remove_body_qposvel(state_0, body_id)
            # self.update_sim_with_new_mocap(self.model_xml, state, grasp_cloth, "true")
            self.ee_mocap_id = self.sim.model.body_mocapid[body_id]
            print("self.ee_mocap_id: ", self.ee_mocap_id)
            pre_goal()
            goto_goal(cloth_pos, grip=True)  # goto_goal(self.action, grip=True)
            #****바꾼부분: 그랩 (1002) 이거 실행이 안됨******
            # self.close_gripper()
            self.sim.data.mocap_pos[self.ee_mocap_id][:] = self.sim.data.get_geom_xpos("grip_geom")

            # trajectory=self.plan_trajectory(self.get_ee_position_W(), self.action+(0,0,0.2), num_points)
            cloth_pos = list(self.get_cloth_position_W().items())[self.a_index][1]
            trajectory=self.plan_trajectory(self.get_ee_position_W(), cloth_pos+(-0.1,-0.4,0.4), num_points)
            for i in range(num_points):
                self.desired_pos_ctrl_W=trajectory[i]
                self.step_env()
                self.sim.data.mocap_pos[self.ee_mocap_id][:] = self.sim.data.get_geom_xpos("grip_geom")
                if self.glfw_window:
                    self.viewer.render(self.sim)
                else:
                    self.viewer.render(width, height, camera_id)
        obs = self.get_obs()
        reward, done, info = self.post_action(obs, raw_action, cosine_distance)
        # info['corner_positions'] = flattened_corners

        # ******바꾼부분: 그랩 (1003) *****
        # self.update_sim_with_new_mocap(self.model_xml, state, grasp_cloth, "false", last=True)

        return obs, reward, done, info

    #**********바꾼부분: 보상***********#
    # def compute_task_reward(self, achieved_goal, desired_goal, info):
    #     return self.task_reward_function(achieved_goal, desired_goal, info)
    def compute_task_reward(self, img, cm):
        return self.task_reward_function(img, cm)
    # def compute_task_reward(self, img, cm, achieved_goal, desired_goal, info):
    #     return self.task_reward_function(img, cm, achieved_goal, desired_goal, info)

    def post_action_image_capture(self):
        camera_matrix, camera_transformation = self.get_camera_matrices(
            self.train_camera, self.image_size[0], self.image_size[1])
        corners_in_image = self.get_corner_image_positions(
            self.image_size[0], self.image_size[0], camera_matrix, camera_transformation)
        flattened_corners = []
        for corner in corners_in_image:
            flattened_corners.append(corner[0]/self.image_size[0])
            flattened_corners.append(corner[1]/self.image_size[1])
        flattened_corners = np.array(flattened_corners)

        return flattened_corners

    def get_corner_constraint_distances(self):
        inv_corner_index_mapping = {v: k for k,
                                    v in self.corner_index_mapping.items()}
        distances = {"0": 0, "1": 0, "2": 0, "3": 0}
        for i, constraint in enumerate(self.constraints):
            if constraint['origin'] in inv_corner_index_mapping.keys():

                #*****바꾼부분: 옷 (0923) **********
                # origin_pos = self.sim.data.get_site_xpos(constraint['origin']).copy() - self.relative_origin
                origin_pos = self.sim.data.get_body_xpos(constraint['origin']).copy() - self.relative_origin
                
                target_pos = self.goal[i *
                                       self.single_goal_dim:(i+1)*self.single_goal_dim]
                distances[inv_corner_index_mapping[constraint['origin']]
                          ] = np.linalg.norm(origin_pos-target_pos)
        return distances

    def post_action(self, obs, raw_action, cosine_distance): #cosine_distance = compute_cosine_distance(self.previous_raw_action, raw_action)
        #***********바꾼부분: 보상******************#
        # reward = self.compute_task_reward(np.reshape(
        #     obs['achieved_goal'], (1, -1)), np.reshape(self.goal, (1, -1)), dict())[0]
        camera_matrix, camera_transformation = self.get_camera_matrices(
            self.train_camera, self.image_size[0], self.image_size[1])
        image_obs, image = self.get_image_obs()
        reward = self.compute_task_reward(image, camera_matrix)
        # reward = self.compute_task_reward(image, camera_matrix, 
        #                                   np.reshape(obs['achieved_goal'], (1, -1)), np.reshape(self.goal, (1, -1)), dict())[0]

        is_success = reward > self.fail_reward

        delta_size = np.linalg.norm(raw_action)
        ctrl_error = np.linalg.norm(self.desired_pos_ctrl_W - self.get_ee_position_W())
        
        #*********바꾼부분**********#
        if np.any(is_success) and self.episode_ee_close_steps == 0: #if is_success and self.episode_ee_close_steps == 0:
            logger.debug(
                # f"Successful fold, reward: {np.round(reward, decimals=3)}")
                f"Successful fold")

        env_memory_usage = self.process.memory_info().rss
        info = {
            "reward": reward,
            'is_success': is_success,
            "delta_size": delta_size,
            "ctrl_error": ctrl_error,
            "env_memory_usage": env_memory_usage,
            "corner_sum_error": 0
        }

        # ***** 바꾼부분: (1003) 임시방편. ********
        # constraint_distances = self.get_corner_constraint_distances()
        constraint_distances = {"0": 0, "1": 0, "2": 0, "3": 0}

        print("여기? 22")

        for key in constraint_distances.keys():
            info[f"corner_{key}"] = constraint_distances[key]
            info["corner_sum_error"] += constraint_distances[key]

        done = False

         #***************바꾼부분: 모션플래닝******************
        # if constraint_distances["1"] < self.success_distance:
        #     self.episode_ee_close_steps += 1
        # else:
        #     self.episode_ee_close_steps = 0
        self.episode_ee_close_steps += 1
            
        if self.episode_ee_close_steps >= self.max_close_steps:
            done = True
        print("여기? 33")

        return reward, done, info

    def update_osc_values(self, q_pos=None): #********바꾼부분: 양팔 (0909)**********
        self.joint_pos_osc = np.ndarray(shape=(7,), dtype=np.float64)
        self.joint_vel_osc = np.ndarray(shape=(7,), dtype=np.float64)
        self.O_T_EE = np.ndarray(shape=(16,), dtype=np.float64)
        self.jac_osc = np.ndarray(shape=(42,), dtype=np.float64)
        self.mass_matrix_osc = np.ndarray(shape=(49,), dtype=np.float64)
        self.tau_J_d_osc = self.sim.data.qfrc_applied[self.joint_vel_addr] - \
            self.sim.data.qfrc_bias[self.joint_vel_addr]

        L = len(self.sim.data.qvel)

        #***********바꾼부분: 모션플래닝 (0909) ***********************
        # if self.action is None:
        #     p = random.choice(list(self.get_cloth_position_W().values()))
        # else:
        #     p = self.action
        p = self.sim.data.site_xpos[self.ee_site_adr]
        
        R = self.sim.data.site_xmat[self.ee_site_adr].reshape(
            [3, 3]).T  # SAATANA

        self.O_T_EE[0] = R[0, 0]
        self.O_T_EE[1] = R[0, 1]
        self.O_T_EE[2] = R[0, 2]
        self.O_T_EE[3] = 0.0

        self.O_T_EE[4] = R[1, 0]
        self.O_T_EE[5] = R[1, 1]
        self.O_T_EE[6] = R[1, 2]
        self.O_T_EE[7] = 0.0

        self.O_T_EE[8] = R[2, 0]
        self.O_T_EE[9] = R[2, 1]
        self.O_T_EE[10] = R[2, 2]
        self.O_T_EE[11] = 0.0

        self.O_T_EE[12] = p[0]
        self.O_T_EE[13] = p[1]
        self.O_T_EE[14] = p[2]
        self.O_T_EE[15] = 1.0

        #********바꾼부분: 양팔 (0909)**********
        if q_pos is not None:
            self.joint_pos_osc=q_pos
            self.joint_vel_osc=[0,0,0,0,0,0,0]
        else:
            for j in range(7):
                self.joint_pos_osc[j] = self.sim.data.qpos[self.joint_pos_addr[j]].copy(
                )
                self.joint_vel_osc[j] = self.sim.data.qvel[self.joint_vel_addr[j]].copy(
                )

        jac_pos_osc = np.ndarray(shape=(L*3,), dtype=np.float64)
        jac_rot_osc = np.ndarray(shape=(L*3,), dtype=np.float64)
        mujoco_py.functions.mj_jacSite(
            self.sim.model, self.sim.data, jac_pos_osc, jac_rot_osc, self.ee_site_adr)

        for j in range(7):
            for r in range(6):
                if (r < 3):
                    value = jac_pos_osc[L*r + self.joint_pos_addr[j]]
                else:
                    value = jac_rot_osc[L*(r-3) + self.joint_pos_addr[j]]
                self.jac_osc[j*6 + r] = value

        mass_array_osc = np.ndarray(shape=(L ** 2,), dtype=np.float64)
        mujoco_py.cymj._mj_fullM(
            self.sim.model, mass_array_osc, self.sim.data.qM)

        for c in range(7):
            for r in range(7):
                self.mass_matrix_osc[c*7 + r] = mass_array_osc[self.joint_pos_addr[r]
                                                               * L + self.joint_pos_addr[c]]

        if self.initial_O_T_EE is None:
            self.initial_O_T_EE = self.O_T_EE.copy()
        if self.initial_joint_osc is None:
            self.initial_joint_osc = self.joint_pos_osc.copy()
        if self.desired_pos_ctrl_W is None:
            self.desired_pos_ctrl_W = p.copy()
        if self.desired_pos_step_W is None:
            self.desired_pos_step_W = p.copy()
        if self.initial_ee_p_W is None:
            self.initial_ee_p_W = p.copy()
            self.min_absolute_W = self.initial_ee_p_W + self.limits_min
            self.max_absolute_W = self.initial_ee_p_W + self.limits_max
    
    #******바꾼부분: 양팔(0829)********
    def update_osc_values_2(self, q_pos=None):  #******바꾼부분: 양팔(0909)********
        self.joint_pos_osc_2 = np.ndarray(shape=(7,), dtype=np.float64)
        self.joint_vel_osc_2 = np.ndarray(shape=(7,), dtype=np.float64)
        self.O_T_EE_2 = np.ndarray(shape=(16,), dtype=np.float64)
        self.jac_osc_2 = np.ndarray(shape=(42,), dtype=np.float64)
        self.mass_matrix_osc_2 = np.ndarray(shape=(49,), dtype=np.float64)
        self.tau_J_d_osc_2 = self.sim.data.qfrc_applied[self.joint_vel_addr_2] - \
            self.sim.data.qfrc_bias[self.joint_vel_addr_2]

        L = len(self.sim.data.qvel) #이게 맞나???

        p_2 = self.sim.data.site_xpos[self.ee_site_adr_2]   #이게 맞나???
        R_2 = self.sim.data.site_xmat[self.ee_site_adr_2].reshape([3, 3]).T
        self.O_T_EE_2[0] = R_2[0, 0]
        self.O_T_EE_2[1] = R_2[0, 1]
        self.O_T_EE_2[2] = R_2[0, 2]
        self.O_T_EE_2[3] = 0.0
        self.O_T_EE_2[4] = R_2[1, 0]
        self.O_T_EE_2[5] = R_2[1, 1]
        self.O_T_EE_2[6] = R_2[1, 2]
        self.O_T_EE_2[7] = 0.0
        self.O_T_EE_2[8] = R_2[2, 0]
        self.O_T_EE_2[9] = R_2[2, 1]
        self.O_T_EE_2[10] = R_2[2, 2]
        self.O_T_EE_2[11] = 0.0
        self.O_T_EE_2[12] = p_2[0]
        self.O_T_EE_2[13] = p_2[1]
        self.O_T_EE_2[14] = p_2[2]
        self.O_T_EE_2[15] = 1.0

        #********바꾼부분: 양팔 (0909)**********
        if q_pos is not None:
            self.joint_pos_osc_2=q_pos
            self.joint_vel_osc_2=[0,0,0,0,0,0,0]
        else:
            for j in range(7):
                self.joint_pos_osc_2[j] = self.sim.data.qpos[self.joint_pos_addr_2[j]].copy(
                )
                self.joint_vel_osc_2[j] = self.sim.data.qvel[self.joint_vel_addr_2[j]].copy(
                )

        jac_pos_osc_2 = np.ndarray(shape=(L*3,), dtype=np.float64)
        jac_rot_osc_2 = np.ndarray(shape=(L*3,), dtype=np.float64)
        mujoco_py.functions.mj_jacSite(
            self.sim.model, self.sim.data, jac_pos_osc_2, jac_rot_osc_2, self.ee_site_adr_2)

        for j in range(7):
            for r in range(6):
                if (r < 3):
                    value_2 = jac_pos_osc_2[L*r + self.joint_pos_addr_2[j]]
                else:
                    value_2 = jac_rot_osc_2[L*(r-3) + self.joint_pos_addr_2[j]]
                self.jac_osc_2[j*6 + r] = value_2

        mass_array_osc_2 = np.ndarray(shape=(L ** 2,), dtype=np.float64)
        mujoco_py.cymj._mj_fullM(
            self.sim.model, mass_array_osc_2, self.sim.data.qM)

        for c in range(7):
            for r in range(7):
                self.mass_matrix_osc_2[c*7 + r] = mass_array_osc_2[self.joint_pos_addr_2[r]
                                                               * L + self.joint_pos_addr_2[c]]
        if self.initial_O_T_EE_2 is None:
            self.initial_O_T_EE_2 = self.O_T_EE_2.copy()
        if self.initial_joint_osc_2 is None:
            self.initial_joint_osc_2 = self.joint_pos_osc_2.copy()
        if self.desired_pos_ctrl_W_2 is None:
            self.desired_pos_ctrl_W_2 = p_2.copy()
        if self.desired_pos_step_W_2 is None:
            self.desired_pos_step_W_2 = p_2.copy()
        if self.initial_ee_p_W_2 is None:
            self.initial_ee_p_W_2 = p_2.copy()
            self.min_absolute_W_2 = self.initial_ee_p_W_2 + self.limits_min
            self.max_absolute_W_2 = self.initial_ee_p_W_2 + self.limits_max

    def get_trajectory_log_entry(self):
        entry = {
            'origin': self.relative_origin,
            'output_max': self.output_max,
            'desired_pos_step_I': self.desired_pos_step_W - self.relative_origin,
            'desired_pos_ctrl_I': self.desired_pos_ctrl_W - self.relative_origin,
            'ee_position_I': self.get_ee_position_I(),
            'raw_action': self.previous_raw_action,
            'substeps': self.substeps,
            'timestep': self.timestep,
            'goal_noise': self.goal_noise
        }
        return entry

    def get_ee_position_W(self): #절대 위치. world position
        return self.sim.data.get_site_xpos(self.ee_site_name).copy()
    def get_ee_position_I(self): #상대 위치. 기준점으로부터 ee가 얼마나 떨어져 있는지
        return self.sim.data.get_site_xpos(self.ee_site_name).copy() - self.relative_origin
    def get_joint_positions(self):
        positions = [self.sim.data.get_joint_qpos(
            joint).copy() for joint in self.joints]
        return np.array(positions)
    def get_joint_velocities(self):
        velocities = [self.sim.data.get_joint_qvel(
            joint).copy() for joint in self.joints]
        return np.array(velocities)
    def get_ee_velocity(self):
        return self.sim.data.get_site_xvelp(self.ee_site_name).copy()
    def get_cloth_position_I(self):
        positions = dict()
        for site in self.cloth_site_names:
            #********바꾼부분: 옷 (0923) *********
            # positions[site] = self.sim.data.get_site_xpos(site).copy() - self.relative_origin
            positions[site] = self.sim.data.get_body_xpos(site).copy() - self.relative_origin
        return positions

    #************바꾼부분: 양팔 (0829)**********
    def get_ee_position_W_2(self): #절대 위치. world position
        return self.sim.data.get_site_xpos(self.ee_site_name_2).copy()
    def get_ee_position_I_2(self): #상대 위치. 기준점으로부터 ee가 얼마나 떨어져 있는지
        return self.sim.data.get_site_xpos(self.ee_site_name_2).copy() - self.relative_origin_2
    def get_joint_positions_2(self):
        positions = [self.sim.data.get_joint_qpos(joint).copy() for joint in self.joints_2]
        return np.array(positions)
    def get_joint_velocities_2(self):
        velocities = [self.sim.data.get_joint_qvel(joint).copy() for joint in self.joints_2]
        return np.array(velocities)
    def get_ee_velocity_2(self):
        return self.sim.data.get_site_xvelp(self.ee_site_name_2).copy()
    def get_cloth_position_I_2(self):
        positions = dict()
        for site in self.cloth_site_names:
            #******바꾼부분: 옷 (0923) *************
            # positions[site] = self.sim.data.get_site_xpos(site).copy() - self.relative_origin_2
            positions[site] = self.sim.data.get_body_xpos(site).copy() - self.relative_origin_2
        return positions

    def get_cloth_position_W(self):
        positions = dict()
        for site in self.cloth_site_names:
            #*****바꾼부분: 옷 (0923) **********
            # positions[site] = self.sim.data.get_site_xpos(site).copy()
            positions[site] = self.sim.data.get_body_xpos(site).copy()
        return positions

    # 바꾼부분: 옷 (1017)
    def get_cloth_edge_positions_W(self):
        positions = dict()
        for i in range(9): #20
            for j in range(9): #20
                if (i in [0, 8]) or (j in [0, 8]):   #[0,19]
                    #*****바꾼부분: 옷(0923) ***********
                    # site_name = f"S{i}_{j}"
                    # positions[site_name] = self.sim.data.get_site_xpos(site_name).copy()
                    site_name = f"B{i}_{j}"
                    positions[site_name] = self.sim.data.get_body_xpos(site_name).copy()
        return positions
    
    def get_cloth_velocity(self):
        velocities = dict()
        for site in self.cloth_site_names:
            #********바꾼부분: 옷 (0923) ***********
            # velocities[site] = self.sim.data.get_site_xvelp(site).copy()
            velocities[site] = self.sim.data.get_body_xvelp(site).copy()
        return velocities

    def get_image_obs(self):
        camera_id = self.sim.model.camera_name2id(
            self.train_camera)
        width = self.randomization_kwargs['camera_config']['width']
        height = self.randomization_kwargs['camera_config']['height']

        #*****************바꾼부분: 시각화*****************
        if self.glfw_window:
            self.viewer.render(self.sim)
        else:
            # self.start_without_ld_preload()
            self.viewer.render(width, height, camera_id)
            # self.return_ld_preload()

        #*****************바꾼부분*****************
        image_obs2 = copy.deepcopy(
            self.viewer.read_pixels(width, height, depth=False))
        # image_obs2 = np.float32(image_obs[::-1, :, :]).copy()

        image_obs = image_obs2[::-1, :, :]

        height_start = int(image_obs.shape[0]/2 - self.image_size[1]/2)
        height_end = height_start + self.image_size[1]

        width_start = int(image_obs.shape[1]/2 - self.image_size[0]/2)
        width_end = width_start + self.image_size[0]
        image_obs = image_obs[height_start:height_end,
                              width_start:width_end, :]

        if self.randomization_kwargs['albumentations_randomization']:
            image_obs = cv2.cvtColor(image_obs, cv2.COLOR_BGR2RGB)
            image_obs = self.albumentations_transform(image=image_obs)["image"]
            image_obs = cv2.cvtColor(image_obs, cv2.COLOR_RGB2GRAY)
        else:
            image_obs = cv2.cvtColor(image_obs, cv2.COLOR_BGR2GRAY)

        #*********************바꾼부분***********#
        return (image_obs / 255).flatten().copy(), image_obs2#data

    def get_obs(self):
        achieved_goal_I = np.zeros(self.single_goal_dim*len(self.constraints))
        for i, constraint in enumerate(self.constraints):
            origin = constraint['origin']
            #*********바꾼부분: 모션플래닝 (1003) *************
            # achieved_goal_I[i*self.single_goal_dim:(i+1)*self.single_goal_dim] = self.sim.data.get_site_xpos(origin).copy() - self.relative_origin
            # achieved_goal_I[i*self.single_goal_dim:(i+1)*self.single_goal_dim] = self.get_ee_position_I()
            achieved_goal_I[i*self.single_goal_dim:(i+1)*self.single_goal_dim] = (0,0,0)

        cloth_position = np.array(list(self.get_cloth_position_I().values()))
        cloth_velocity = np.array(list(self.get_cloth_velocity().values()))

        cloth_observation = np.concatenate(
            [cloth_position.flatten(), cloth_velocity.flatten()])

        desired_pos_ctrl_I = self.desired_pos_ctrl_W - self.relative_origin

        #****바꾼부분: 액션 (1003) **********
        self.goal, self.goal_noise = self.sample_goal_I()
        full_observation = {'achieved_goal': achieved_goal_I.copy(), 'desired_goal': self.goal.copy()}
        # full_observation = {'achieved_goal': achieved_goal_I.copy(), 'desired_goal': np.concatenate(self.get_cloth_position_W().values()).copy()}

        if self.robot_observation == "ee":
            robot_observation = np.concatenate(
                [self.get_ee_position_I(), self.get_ee_velocity(), desired_pos_ctrl_I])
        elif self.robot_observation == "ctrl":
            robot_observation = np.concatenate(
                [self.previous_raw_action, np.zeros(6)])
        elif self.robot_observation == "none":
            robot_observation = np.zeros(9)
        #****************바꾼부분*********************#
        full_observation['initial_image'] = self.frame_stack[0]
        
        full_observation['image'] = np.array(
            [image for image in self.frame_stack]).flatten()
        if self.randomization_kwargs["dynamics_randomization"]:
            full_observation['observation'] = np.concatenate(
                [cloth_observation.copy(), np.array(self.mujoco_model_numerical_values)])
        else:
            full_observation['observation'] = cloth_observation.copy(
            ).flatten()
        full_observation['robot_observation'] = robot_observation.flatten(
        ).copy()

        return full_observation

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # def sample_goal_I(self):
    #     goal = np.zeros(self.single_goal_dim*len(self.constraints))
    #     noise = self.np_random.uniform(self.goal_noise_range[0],
    #                                    self.goal_noise_range[1])
    #     for i, constraint in enumerate(self.constraints):
    #         target = constraint['target']
    #         target_pos = self.sim.data.get_site_xpos(target).copy()
    #         offset = np.zeros(self.single_goal_dim)
    #         if 'noise_directions' in constraint.keys():
    #             for idx, offset_dir in enumerate(constraint['noise_directions']):
    #                 offset[idx] = offset_dir*noise
    #         goal[i*self.single_goal_dim: (i+1) *
    #              self.single_goal_dim] = target_pos + offset - self.relative_origin
    #     return goal.copy(), noise

    #******바꾼부분: 액션 (0922)*********
    def sample_goal_I(self):

        #****바꾼부분: 액션 (1003) *******
        goal = list(self.get_cloth_position_W().values())
        # goal = list(self.get_cloth_position_W().values())-self.get_ee_position_W()

        goal=np.concatenate(goal)
        noise = self.np_random.uniform(self.goal_noise_range[0],
                                       self.goal_noise_range[1])
        return goal.copy(), noise

    def reset_osc_values(self):
        self.initial_O_T_EE = None
        self.initial_joint_osc = None
        self.initial_ee_p_W = None
        self.desired_pos_step_W = None
        self.desired_pos_ctrl_W = None
        self.previous_raw_action = np.zeros(3)
        self.raw_action = None
        #****바꾼부분: 양팔 (0829) **********
        self.initial_O_T_EE_2 = None
        self.initial_joint_osc_2 = None
        self.initial_ee_p_W_2 = None
        self.desired_pos_step_W_2 = None
        self.desired_pos_ctrl_W_2 = None

    def setup_xml_model(self, randomize, rownum=None):
        model_kwargs, model_numerical_values = self.build_xml_kwargs_and_numerical_values(
            randomize=randomize, rownum=rownum)
        self.mujoco_model_numerical_values = model_numerical_values

        #******바꾼부분: 그랩 (1001) 버전2*********
        # self.setup_initial_state_and_sim(model_kwargs)
        self.model_xml=self.setup_initial_state_and_sim(model_kwargs)

    def reset(self):
        self.sim.reset()
        self.sim.set_state(self.initial_state)
        self.sim.data.qfrc_applied[self.joint_vel_addr] = self.initial_qfrc_applied
        self.sim.data.qfrc_bias[self.joint_vel_addr] = self.initial_qfrc_bias

        self.sim.forward()  # BODY POSITIONS CORRECT

        self.reset_camera()
        
        self.sim.forward()  # CAMERA CHANGES CORRECT

        self.reset_osc_values()

        self.update_osc_values()
        self.relative_origin = self.get_ee_position_W()

        #********바꾼부분: 양팔 (0829)************
        self.update_osc_values_2()
        self.relative_origin_2 = self.get_ee_position_W_2()

        # self.goal, self.goal_noise = self.sample_goal_I()


        if not self.viewer is None:
            del self.viewer._markers[:]

        self.episode_ee_close_steps = 0


        #****바꾼부분: 그랩 (1002) 이거 실행이 안됨******
        # self.open_gripper()

        #********바꾼부분: 양팔 (0829)**************
        #****바꾼부분: 그랩 (1002) 이거 실행이 안됨******
        # self.open_gripper_2()

        #**********바꾼부분*************#
        image_obs, image_ = self.get_image_obs()
        for _ in range(self.frame_stack_size):
            self.frame_stack.append(image_obs)

        q_ok = np.allclose(self.initial_qpos, self.get_joint_positions(), rtol=0.01, atol=0.01)

        #******바꾼부분: 양팔 (0829)********
        q_ok_2 = np.allclose(self.initial_qpos_2, self.get_joint_positions(), rtol=0.01, atol=0.01)

        #****바꾼부분: 양팔 (1003) **********
        # self.step_2()
        obs = self.step_2()

        self.goal, self.goal_noise = self.sample_goal_I()

        #****바꾼부분: 양팔 (1003) **********
        # return self.get_obs()
        return obs

    def get_corner_image_positions(self, w, h, camera_matrix, camera_transformation):
        corners = []
        cloth_positions = self.get_cloth_position_W()
        for site in self.corner_index_mapping.values():
            corner_in_image = np.ones(4)
            corner_in_image[:3] = cloth_positions[site]
            corner = (camera_matrix @ camera_transformation) @ corner_in_image
            u_c, v_c, _ = corner/corner[2]
            corner = [w-u_c, v_c]
            corners.append(corner)
        return corners

    def get_edge_image_positions(self, w, h, camera_matrix, camera_transformation):
        edges = []
        cloth_edge_positions = self.get_cloth_edge_positions_W()
        for site in cloth_edge_positions.keys():
            edge_in_image = np.ones(4) #homogeneous 좌표를 저장하는 [x,y,z,1] 형태
            edge_in_image[:3] = cloth_edge_positions[site] # [x,y,z] 부분 채우기: 위에서 가져온 get_cloth_edge_position으로 정하기
            edge = (camera_matrix @ camera_transformation) @ edge_in_image
            u_c, v_c, _ = edge/edge[2]
            edge = [w-u_c, v_c]
            edges.append(edge)
        return edges
    
    #*****************바꾼부분*****************
    def get_all_image_positions(self, w, h, camera_matrix, camera_transformation):
        alls = []
        cloth_all_positions = self.get_cloth_position_W()
        for site in cloth_all_positions.keys():
            all_in_image = np.ones(4)
            all_in_image[:3] = cloth_all_positions[site]
            all = (camera_matrix @ camera_transformation) @ all_in_image
            u_c, v_c, _ = all/all[2]
            all = [w-u_c, v_c]
            alls.append(all)
        return alls

    def get_camera_matrices(self, camera_name, w, h):
        camera_id = self.sim.model.camera_name2id(camera_name)
        fovy = self.sim.model.cam_fovy[camera_id]
        f = 0.5 * h / math.tan(fovy * math.pi / 360)
        camera_matrix = np.array(((f, 0, w / 2), (0, f, h / 2), (0, 0, 1)))
        xmat = self.sim.data.get_camera_xmat(camera_name)
        xpos = self.sim.data.get_camera_xpos(camera_name)

        camera_transformation = np.eye(4)
        camera_transformation[:3, :3] = xmat
        camera_transformation[:3, 3] = xpos
        camera_transformation = np.linalg.inv(camera_transformation)[:3, :]

        return camera_matrix, camera_transformation

    def get_masked_image(self, camera, width, height, ee_in_image, aux_output, point_size, greyscale=False, mask_type=None):
        camera_matrix, camera_transformation = self.get_camera_matrices(
            camera, width, height)
        camera_id = self.sim.model.camera_name2id(camera)

        #*****************바꾼부분: 시각화*****************
        if self.glfw_window:
            self.viewer.render(self.sim)
        else:
            # self.start_without_ld_preload()
            self.viewer.render(width, height, camera_id)
            # self.return_ld_preload()

        data = np.float32(self.viewer.read_pixels(
            width, height, depth=False)).copy()
        data = np.float32(data[::-1, :, :]).copy()
        data = np.float32(data)
        if greyscale:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        # ee = camera_matrix @ camera_transformation @ ee_in_image
        # u_ee, v_ee, _ = ee/ee[2]
        # cv2.circle(data, (width-int(u_ee), int(v_ee)), point_size, (0, 0, 0), -1)

        if mask_type == "corners":
            mask = self.get_corner_image_positions(
                width, height, camera_matrix, camera_transformation)
        elif mask_type == "edges":
            mask = self.get_edge_image_positions(
                width, height, camera_matrix, camera_transformation)
        #*****************바꾼부분*****************
        elif mask_type == "alls":
            mask = self.get_all_image_positions(
                width, height, camera_matrix, camera_transformation)
        else:
            mask = []

        for point in mask:
            u = int(point[0])
            v = int(point[1])
            cv2.circle(data, (u, v), point_size, (255, 0, 0), -1)

        #*****************바꾼부분*****************
        # if not aux_output is None:
        #     for aux_idx in range(4):
        #         aux_u = int(aux_output.flatten()[aux_idx*2]*width)
        #         aux_v = int(aux_output.flatten()[aux_idx*2+1]*height)
        #         cv2.circle(data, (aux_u, aux_v), point_size, (0, 255, 0), -1)

        return data
    
    #*****************바꾼부분*****************
    def capture_images(self, aux_output=None, mask_type="alls"):  # "corners"

        #*****************바꾼부분*****************
        w_eval, h_eval = 1000, 1000   #500,500
        w_corners, h_corners = 1000, 1000   #500,500
        w_cnn, h_cnn = 1000,1000 #self.image_size
        # w_cnn_full, h_cnn_full = self.randomization_kwargs['camera_config'][
        #     'width'], self.randomization_kwargs['camera_config']['height']
        w_cnn_full, h_cnn_full = self.randomization_kwargs['camera_config'][
            'width']*2, self.randomization_kwargs['camera_config']['height']*2

        ee_in_image = np.ones(4)
        ee_pos = self.get_ee_position_W()
        ee_in_image[:3] = ee_pos

        corner_image = self.get_masked_image(
            self.train_camera, w_corners, h_corners, ee_in_image, aux_output, 6, greyscale=False, mask_type=mask_type)  #8
        eval_image = self.get_masked_image(
            self.eval_camera, w_eval, h_eval, ee_in_image, None, 3, greyscale=False, mask_type=mask_type) #4
        cnn_color_image_full = self.get_masked_image(
            self.train_camera, w_cnn_full, h_cnn_full, ee_in_image, aux_output, 2, mask_type=mask_type) #2
        cnn_color_image = self.get_masked_image(
            self.train_camera, w_cnn, h_cnn, ee_in_image, aux_output, 2, mask_type=mask_type) #2
        cnn_image = self.get_masked_image(
            self.train_camera, w_cnn, h_cnn, ee_in_image, aux_output, 2, greyscale=True, mask_type=mask_type) #2

        return corner_image, eval_image, cnn_color_image_full, cnn_color_image, cnn_image

class ClothEnv(ClothEnv_, EzPickle):
    def __init__(self, **kwargs):
        ClothEnv_.__init__(self, **kwargs)
        EzPickle.__init__(self)