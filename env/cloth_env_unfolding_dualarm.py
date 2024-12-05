import mujoco_py
import osc_binding
import cv2
import numpy as np
import copy
from gym.utils import seeding, EzPickle

from utils import reward_calculation_unfolding
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
        image_size=200,  #100
        glfw_window=True,
    ):
        
        self.glfw_window=glfw_window
        self.original_ld_preload = os.environ.get('LD_PRELOAD')

        self.num_cloth=9 # 9     #<composite type="grid" count="num num 1"  

        self.action=None
        self.R = None

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
*
        self.eval_camera = "front"

        self.episode_ee_close_steps = 0

        self.seed()

        self.initial_qpos = np.array([0.212422, 0.362907, -0.00733391, -1.9649, -0.0198034, 2.37451, -1.50499])

        self.ee_site_name = 'grip_site'
        self.joints = ["joint1", "joint2", "joint3",
                       "joint4", "joint5", "joint6", "joint7"]
        
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

        self.setup_initial_state_and_sim(model_kwargs)

        self.dump_xml_models()

        self.build_motion_distance_and_constraints()


        for index, d in enumerate(list(self.get_cloth_position_W().items())): #B0_0과, 그에 해당하는 value
                if self.singlemocap in d[0]: #B0_0형태
                    self.a2_index=index
        for index, d in enumerate(list(self.get_cloth_position_W().items())): #B0_0과, 그에 해당하는 value
                if self.policymocap in d[0]: #B0_0형태
                    self.a_index=index

        self.action_space = gym.spaces.Box(-1,1, shape=(3,), dtype='float32')

        self.relative_origin = self.get_ee_position_W()
        
        self.relative_origin_2 = self.get_ee_position_W_2()

        self.goal, self.goal_noise = self.sample_goal_I()

        self.reset_camera()

        image_obs, initial_image = self.get_image_obs()

        for _ in range(self.frame_stack_size):
            self.frame_stack.append(image_obs)
        obs = self.get_obs()

        self.observation_space = gym.spaces.Dict(dict(
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

    def quat_2_rotmat(self,Q):
        q0 = Q[0]
        q1 = Q[1]
        q2 = Q[2]
        q3 = Q[3]
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1
        rot_matrix = np.array([[r00, r01, r02],
                            [r10, r11, r12],
                            [r20, r21, r22]])    
        return rot_matrix
    def euler_2_rotmat(self, roll, pitch, yaw):
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        R = Rz @ Ry @ Rx
        return R
    def quaternion_to_euler(self,q):
        w, x, y, z = q
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.sign(sinp) * (np.pi / 2)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw    
    
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

        model_kwargs['timestep'] = self.timestep
        model_kwargs['lights_randomization'] = self.randomization_kwargs['lights_randomization']
        model_kwargs['materials_randomization'] = self.randomization_kwargs['materials_randomization']
        model_kwargs['train_camera_fovy'] = (self.randomization_kwargs['camera_config']
                                             ['fovy_range'][0] + self.randomization_kwargs['camera_config']['fovy_range'][1])/2
        model_kwargs['num_lights'] = 1

        model_kwargs['geom_spacing'] = (self.randomization_kwargs['cloth_size'] - 2*model_kwargs['geom_size']) / (self.num_cloth-1)
        model_kwargs['offset'] = (self.num_cloth-1)/2 * model_kwargs['geom_spacing']

        appearance_choices = mujoco_model_kwargs.appearance_kwarg_choices
        appearance_ranges = mujoco_model_kwargs.appearance_kwarg_ranges
        for key in appearance_choices.keys():
            model_kwargs[key] = appearance_choices[key][1]
        for key in appearance_ranges.keys():
            values = appearance_ranges[key]
            model_kwargs[key] = values[1]

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
        
        min_all = 0
        max_all = self.num_cloth-1
        mid_all = int(max_all / 2)
        self.cloth_site_names = []
        for i in range(max_all+1):  #[min_all, mid_all, max_all]
            for j in range(max_all+1):   #[min_all, mid_all, max_all]
                # self.cloth_site_names.append(f"S{i}_{j}")  #원래
                self.cloth_site_names.append(f"B{i}_{j}") #0923
        
        self.task_reward_function = reward_calculation_unfolding.get_unfolding_reward_function()

        return model_kwargs, model_numerical_values

    def build_motion_distance_and_constraints(self):
        cloth_positions_list = list(self.get_cloth_position_W().values())
        self.motion_distance = cloth_positions_list - self.get_ee_position_W()
        self.constraints = task_definitions.constraints["random_cloth"](0, (self.num_cloth-1)/2, self.num_cloth-1, self.motion_distance, -0, -0)
    
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
            
            if self.glfw_window:
                self.viewer = mujoco_py.MjRenderContextWindow(self.sim)
            else:
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)

            self.viewer.cam.fixedcamid = self.sim.model.camera_name2id('front')
            self.viewer.cam.type = mujoco_py.generated.const.CAMERA_FIXED
            
            self.viewer.vopt.geomgroup[0] = 0
            self.viewer.vopt.geomgroup[1] = 1

    def dump_xml_models(self):
        with open(f"{self.save_folder}/compiled_mujoco_model_no_inertias.xml", "w") as f:
            self.sim.save(f, format='xml', keep_inertials=False)

        with open(f"{self.save_folder}/compiled_mujoco_model_with_intertias.xml", "w") as f:
            self.sim.save(f, format='xml', keep_inertials=True)

    def reset_camera(self):
        lookat_offset = np.zeros(3)
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

            if subelement.getAttribute("name").startswith('B'):
                subelement.setAttribute("mocap", "true")
                for child_node in subelement.childNodes:
                    if child_node.nodeType == 1:
                        if child_node.tagName == "joint":
                            subelement.removeChild(child_node)

        return remove_empty_lines(dom.toprettyxml(indent=" " * 4))
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

        self.singlemocap="B0_8" **
        self.policymocap="B1_1"
        temp_xml_2 = self.add_new_mocap_to_xml(temp_xml_2, self.singlemocap, "true", self.policymocap, "true")

        self.mjpy_model = mujoco_py.load_model_from_xml(temp_xml_2)

        del temp_xml_2

        gc.collect()
        self.sim = mujoco_py.MjSim(self.mjpy_model)
        self.setup_viewer()

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

        if not last:
            self.sim.set_state(state)
            self.sim.forward()
    
    def set_robot_initial_joints(self):
        for j, joint in enumerate(self.joints):
            self.sim.data.set_joint_qpos(joint, self.initial_qpos[j])
        
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
    
    def plan_trajectory(self, start_pos, end_pos, num_points):
        start = np.array(start_pos)
        end = np.array(end_pos)
        t = np.linspace(0, 1, num_points)
        path = np.outer(1 - t, start) + np.outer(t, end)
        return path
    def plan_joint_trajectory(self, start_joint, end_joint, num_points):
        start = np.array(start_joint)
        end = np.array(end_joint)
        t = np.linspace(0, 1, num_points)
        path = np.outer(1 - t, start) + np.outer(t, end)
        return path
    
    def step_env(self, q_pos=None, action=None):
        mujoco_py.functions.mj_step1(self.sim.model, self.sim.data)
        if q_pos is not None:
            if action is not None:
                self.update_osc_values(q_pos=q_pos, action=True)
            else:
                self.update_osc_values(q_pos=q_pos)
        else:
            if action is not None:
                self.update_osc_values(action=True)
            else:
                self.update_osc_values()
        tau = self.run_controller()
        self.sim.data.qfrc_applied[self.joint_vel_addr] = self.sim.data.qfrc_bias[self.joint_vel_addr] + tau
        mujoco_py.functions.mj_step2(self.sim.model, self.sim.data)

    def step_env_2(self, q_pos=None): 
        mujoco_py.functions.mj_step1(self.sim.model, self.sim.data)
        if q_pos is not None:
            self.update_osc_values_2(q_pos)
        else:
            self.update_osc_values_2()
        tau_2 = self.run_controller_2()
        self.sim.data.qfrc_applied[self.joint_vel_addr_2] = self.sim.data.qfrc_bias[self.joint_vel_addr_2] + tau_2
        mujoco_py.functions.mj_step2(self.sim.model, self.sim.data)
    
    def set_joint_positions(self, q_pos):
        for j, joint in enumerate(self.joints):
            self.sim.data.set_joint_qpos(joint, q_pos[j])
    def set_joint_positions_2(self, q_pos):
        for j_2, joint_2 in enumerate(self.joints_2):
            self.sim.data.set_joint_qpos(joint_2, q_pos[j_2])

    def remove_body_qposvel(self, state, body_id):
        qpos_start = self.sim.model.jnt_qposadr[self.sim.model.body_jntadr[body_id]]
        qpos_end = qpos_start + 3
        new_qpos = np.delete(state.qpos, slice(qpos_start, qpos_end))
        new_qvel = np.delete(state.qvel, slice(qpos_start, qpos_end))
        new_state = mujoco_py.MjSimState(state.time, new_qpos, new_qvel, state.act, state.udd_state)
        return new_state, qpos_start, qpos_end
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
    
    def get_surroundings(self, grasp_cloth):
        first=int(grasp_cloth.split('_')[0][1:])
        second=int(grasp_cloth.split('_')[1])
        valid_range = range(0, self.num_cloth)
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

    def step_2(self):
        camera_id = self.sim.model.camera_name2id(
            self.train_camera)
        width = self.randomization_kwargs['camera_config']['width']
        height = self.randomization_kwargs['camera_config']['height']

        body_id = self.sim.model.body_name2id(self.singlemocap) #0~110 숫자

        self.ee_mocap_id_2 = self.sim.model.body_mocapid[body_id]

        body_i = self.sim.model.body_name2id(self.policymocap)
        ee_mocap_id = self.sim.model.body_mocapid[body_i]
        sur=self.get_surroundings(self.policymocap)

        goal_2=list(self.get_cloth_position_W().items())[self.a2_index][1]

        policy_pos = list(self.get_cloth_position_W().items())[self.a_index][1]
        self.initial_dist= math.sqrt(sum((a - b) ** 2 for a, b in zip(goal_2, policy_pos)))

        j_pos_1=self.get_joint_positions()
        pos_1=self.get_ee_position_W()
        num_points=200
        trajectory_2=self.plan_trajectory(self.get_ee_position_W_2(), goal_2, num_points)
        for i in range(num_points):
            self.desired_pos_ctrl_W_2=trajectory_2[i]
            self.step_env_2()
            if self.glfw_window:
                self.viewer.render(self.sim)
            else:
                self.viewer.render(width, height, camera_id)
        
        self.sim.data.mocap_pos[ee_mocap_id][:] = self.get_avg_position(sur)
        if self.glfw_window:
            self.viewer.render(self.sim)
        else:
            self.viewer.render(width, height, camera_id)

        trajectory_2_2=self.plan_trajectory(self.get_ee_position_W_2(), (0.41, 0.12, 0.55), num_points)
        for i in range(num_points):
            self.desired_pos_ctrl_W_2=trajectory_2_2[i]
            self.step_env_2()
            self.sim.data.mocap_pos[self.ee_mocap_id_2][:] = self.sim.data.get_geom_xpos("grip_geom_2")
            self.sim.data.mocap_pos[ee_mocap_id][:] = self.get_avg_position(sur)
            if self.glfw_window:
                self.viewer.render(self.sim)
            else:
                self.viewer.render(width, height, camera_id)
        
        self.post_goal_2=self.get_ee_position_W_2()
        self.last_single_joint = self.get_joint_positions_2()

        self.goal, self.goal_noise = self.sample_goal_I()
        obs = self.get_obs()
        return obs
    
    def step(self, action):
        camera_id = self.sim.model.camera_name2id(
            self.train_camera)
        width = self.randomization_kwargs['camera_config']['width']
        height = self.randomization_kwargs['camera_config']['height']
        num_points=200

        self.action=action.copy()
        print("action: ", self.action)
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

        def pre_goal(): 
            goal_1=self.get_ee_position_W()+(0,0,0.3)
            trajectory_pre=self.plan_trajectory(self.get_ee_position_W(), goal_1, num_points)
            for i in range(num_points):
                self.desired_pos_ctrl_W=trajectory_pre[i]

                self.set_joint_positions_2(self.last_single_joint)

                self.step_env()
                if self.glfw_window:
                    self.viewer.render(self.sim)
                else:
                    self.viewer.render(width, height, camera_id)
            self.desired_pos_ctrl_W=goal_1
            self.step_env()
        def goto_goal(goal, grip): 
            trajectory=self.plan_trajectory(self.get_ee_position_W(), goal, num_points)
            for i in range(num_points):
                self.desired_pos_ctrl_W=trajectory[i]
                self.set_joint_positions_2(self.last_single_joint)
                if i>=num_points-1:
                    if grip:
                        self.sim.data.mocap_pos[self.ee_mocap_id][:] = self.sim.data.get_geom_xpos("grip_geom")
                    self.R=self.euler_2_rotmat(self.action[0],self.action[1],self.action[2])
                    self.step_env(action=True)
                else:
                    self.step_env()
                if self.glfw_window:
                    self.viewer.render(self.sim)
                else:
                    self.viewer.render(width, height, camera_id)
            self.desired_pos_ctrl_W=goal
            self.step_env()

        body_id = self.sim.model.body_name2id(self.policymocap) #0~110 숫자
        self.ee_mocap_id = self.sim.model.body_mocapid[body_id]
        
        cloth_pos = list(self.get_cloth_position_W().items())[self.a_index][1]
        goto_goal(cloth_pos, grip=True)

        cloth_pos = list(self.get_cloth_position_W().items())[self.a_index][1]
        trajectory=self.plan_trajectory(self.get_ee_position_W(), cloth_pos+(-0.05,-0.05,0.1), num_points)
        for i in range(num_points):
            self.desired_pos_ctrl_W=trajectory[i]
            self.set_joint_positions_2(self.last_single_joint)

            self.sim.data.mocap_pos[self.ee_mocap_id][:] = self.sim.data.get_geom_xpos("grip_geom")
            self.step_env()
            if self.glfw_window:
                self.viewer.render(self.sim)
            else:
                self.viewer.render(width, height, camera_id)
            print("11")
        
        cloth_pos = list(self.get_cloth_position_W().items())[self.a_index][1]
        trajectory=self.plan_trajectory(self.get_ee_position_W(), (self.post_goal_2[0], 0,self.post_goal_2[2]), num_points)
        for i in range(num_points*2):
            self.desired_pos_ctrl_W=trajectory[i]
            self.set_joint_positions_2(self.last_single_joint)
            self.step_env()
            self.sim.data.mocap_pos[self.ee_mocap_id][:] = self.sim.data.get_geom_xpos("grip_geom")
            if self.glfw_window:
                self.viewer.render(self.sim)
            else:
                self.viewer.render(width, height, camera_id)
            print("22")
    
        cloth_pos = list(self.get_cloth_position_W().items())[self.a_index][1]
        trajectory=self.plan_trajectory(self.get_ee_position_W(), cloth_pos+(0,-1,0), num_points)
        for i in range(num_points):
            single_pos=list(self.get_cloth_position_W().items())[self.a2_index][1]
            policy_pos = list(self.get_cloth_position_W().items())[self.a_index][1]
            dist= math.sqrt(sum((a - b) ** 2 for a, b in zip(single_pos, policy_pos)))
            if (dist >= self.initial_dist):
                break
            self.desired_pos_ctrl_W=trajectory[i]
            self.set_joint_positions_2(self.last_single_joint)

            self.step_env()
            self.sim.data.mocap_pos[self.ee_mocap_id][:] = self.sim.data.get_geom_xpos("grip_geom")
            if self.glfw_window:
                self.viewer.render(self.sim)
            else:
                self.viewer.render(width, height, camera_id)
            print("33")
        
        obs = self.get_obs()
        reward, done, info = self.post_action(obs, raw_action, cosine_distance)

        info['corner_positions'] = np.zeros((8))

        return obs, reward, done, info

    def compute_task_reward(self, img, cm):
        return self.task_reward_function(img, cm)

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

                origin_pos = self.sim.data.get_body_xpos(constraint['origin']).copy() - self.relative_origin
                
                target_pos = self.goal[i *
                                       self.single_goal_dim:(i+1)*self.single_goal_dim]
                distances[inv_corner_index_mapping[constraint['origin']]
                          ] = np.linalg.norm(origin_pos-target_pos)
        return distances

    def post_action(self, obs, raw_action, cosine_distance): #cosine_distance = compute_cosine_distance(self.previous_raw_action, raw_action)
        camera_matrix, camera_transformation = self.get_camera_matrices(
            self.train_camera, self.image_size[0], self.image_size[1])
        image_obs, image = self.get_image_obs()
        reward = self.compute_task_reward(image, camera_matrix)
        is_success = reward > self.fail_reward

        delta_size = np.linalg.norm(raw_action)
        ctrl_error = np.linalg.norm(self.desired_pos_ctrl_W - self.get_ee_position_W())

        if np.any(is_success) and self.episode_ee_close_steps == 0: #if is_success and self.episode_ee_close_steps == 0:
            logger.debug(
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

        constraint_distances = {"0": 0, "1": 0, "2": 0, "3": 0}

        for key in constraint_distances.keys():
            info[f"corner_{key}"] = constraint_distances[key]
            info["corner_sum_error"] += constraint_distances[key]

        done = False

        self.episode_ee_close_steps += 1
            
        if self.episode_ee_close_steps >= self.max_close_steps:
            done = True
            
        return reward, done, info

    def update_osc_values(self, q_pos=None, action=None):
        self.joint_pos_osc = np.ndarray(shape=(7,), dtype=np.float64)
        self.joint_vel_osc = np.ndarray(shape=(7,), dtype=np.float64)
        self.O_T_EE = np.ndarray(shape=(16,), dtype=np.float64)
        self.jac_osc = np.ndarray(shape=(42,), dtype=np.float64)
        self.mass_matrix_osc = np.ndarray(shape=(49,), dtype=np.float64)
        self.tau_J_d_osc = self.sim.data.qfrc_applied[self.joint_vel_addr] - \
            self.sim.data.qfrc_bias[self.joint_vel_addr]

        L = len(self.sim.data.qvel)

        p = self.sim.data.site_xpos[self.ee_site_adr]
        
        if action is None:
            R = self.sim.data.site_xmat[self.ee_site_adr].reshape([3, 3]).T  # SAATANA
        else:
            R = self.R

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
    
    def update_osc_values_2(self, q_pos=None):
        self.joint_pos_osc_2 = np.ndarray(shape=(7,), dtype=np.float64)
        self.joint_vel_osc_2 = np.ndarray(shape=(7,), dtype=np.float64)
        self.O_T_EE_2 = np.ndarray(shape=(16,), dtype=np.float64)
        self.jac_osc_2 = np.ndarray(shape=(42,), dtype=np.float64)
        self.mass_matrix_osc_2 = np.ndarray(shape=(49,), dtype=np.float64)
        self.tau_J_d_osc_2 = self.sim.data.qfrc_applied[self.joint_vel_addr_2] - \
            self.sim.data.qfrc_bias[self.joint_vel_addr_2]

        L = len(self.sim.data.qvel)

        p_2 = self.sim.data.site_xpos[self.ee_site_adr_2]
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

    def get_ee_position_W(self):
        return self.sim.data.get_site_xpos(self.ee_site_name).copy()
    def get_ee_position_I(self):
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
            positions[site] = self.sim.data.get_body_xpos(site).copy() - self.relative_origin
        return positions

    def get_ee_position_W_2(self):
        return self.sim.data.get_site_xpos(self.ee_site_name_2).copy()
    def get_ee_position_I_2(self):
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
            positions[site] = self.sim.data.get_body_xpos(site).copy() - self.relative_origin_2
        return positions

    def get_cloth_position_W(self):
        positions = dict()
        for site in self.cloth_site_names:
            positions[site] = self.sim.data.get_body_xpos(site).copy()
        return positions

    def get_cloth_quat_W(self):
        quats = dict()
        for site in self.cloth_site_names:
            quats[site] = self.sim.data.get_body_xquat(site).copy()
        return quats

    def get_cloth_edge_positions_W(self):
        positions = dict()
        for i in range(self.num_cloth): #9
            for j in range(self.num_cloth): #9
                if (i in [0, self.num_cloth-1]) or (j in [0, self.num_cloth-1]):   #[0,8]
                    site_name = f"B{i}_{j}"
                    positions[site_name] = self.sim.data.get_body_xpos(site_name).copy()
        return positions
    
    def get_cloth_velocity(self):
        velocities = dict()
        for site in self.cloth_site_names:
            velocities[site] = self.sim.data.get_body_xvelp(site).copy()
        return velocities

    def get_image_obs(self):
        camera_id = self.sim.model.camera_name2id(
            self.train_camera)
        width = self.randomization_kwargs['camera_config']['width']
        height = self.randomization_kwargs['camera_config']['height']

        if self.glfw_window:
            self.viewer.render(self.sim)
        else:
            self.viewer.render(width, height, camera_id)

        image_obs2 = copy.deepcopy(
            self.viewer.read_pixels(width, height, depth=False))

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

        return (image_obs / 255).flatten().copy(), image_obs2#data

    def get_obs(self):
        achieved_goal_I = np.zeros(self.single_goal_dim*len(self.constraints))
        for i, constraint in enumerate(self.constraints):
            origin = constraint['origin']
            achieved_goal_I[i*self.single_goal_dim:(i+1)*self.single_goal_dim] = (0,0,0)

        cloth_position = np.array(list(self.get_cloth_position_I().values()))
        cloth_velocity = np.array(list(self.get_cloth_velocity().values()))

        cloth_observation = np.concatenate(
            [cloth_position.flatten(), cloth_velocity.flatten()])

        desired_pos_ctrl_I = self.desired_pos_ctrl_W - self.relative_origin

        self.goal, self.goal_noise = self.sample_goal_I()
        full_observation = {'achieved_goal': achieved_goal_I.copy(), 'desired_goal': self.goal.copy()}

        if self.robot_observation == "ee": robot_observation = np.concatenate([self.get_ee_position_I(), self.get_ee_velocity(), desired_pos_ctrl_I])
        elif self.robot_observation == "ctrl":
            robot_observation = np.concatenate(
                [self.previous_raw_action, np.zeros(6)])
        elif self.robot_observation == "none":
            robot_observation = np.zeros(9)
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

    def sample_goal_I(self):
        goal_value = self.quaternion_to_euler(self.sim.model.site_quat[self.ee_site_adr])
        goal = [goal_value for _ in self.get_cloth_quat_W().values()] # cloth_quat 개수만큼 반복

        goal = np.concatenate(goal)
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
        self.initial_O_T_EE_2 = None
        self.initial_joint_osc_2 = None
        self.initial_ee_p_W_2 = None
        self.desired_pos_step_W_2 = None
        self.desired_pos_ctrl_W_2 = None

    def setup_xml_model(self, randomize, rownum=None):
        model_kwargs, model_numerical_values = self.build_xml_kwargs_and_numerical_values(
            randomize=randomize, rownum=rownum)
        self.mujoco_model_numerical_values = model_numerical_values

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

        self.update_osc_values_2()
        self.relative_origin_2 = self.get_ee_position_W_2()

        if not self.viewer is None:
            del self.viewer._markers[:]

        self.episode_ee_close_steps = 0

        image_obs, image_ = self.get_image_obs()
        for _ in range(self.frame_stack_size):
            self.frame_stack.append(image_obs)

        q_ok = np.allclose(self.initial_qpos, self.get_joint_positions(), rtol=0.01, atol=0.01)

        q_ok_2 = np.allclose(self.initial_qpos_2, self.get_joint_positions(), rtol=0.01, atol=0.01)

        obs = self.step_2()

        self.goal, self.goal_noise = self.sample_goal_I()

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
            edge_in_image = np.ones(4)
            edge_in_image[:3] = cloth_edge_positions[site]
            edge = (camera_matrix @ camera_transformation) @ edge_in_image
            u_c, v_c, _ = edge/edge[2]
            edge = [w-u_c, v_c]
            edges.append(edge)
        return edges
    
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

        if self.glfw_window:
            self.viewer.render(self.sim)
        else:
            self.viewer.render(width, height, camera_id)

        data = np.float32(self.viewer.read_pixels(
            width, height, depth=False)).copy()
        data = np.float32(data[::-1, :, :]).copy()
        data = np.float32(data)
        if greyscale:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        if mask_type == "corners":
            mask = self.get_corner_image_positions(
                width, height, camera_matrix, camera_transformation)
        elif mask_type == "edges":
            mask = self.get_edge_image_positions(
                width, height, camera_matrix, camera_transformation)
        elif mask_type == "alls":
            mask = self.get_all_image_positions(
                width, height, camera_matrix, camera_transformation)
        else:
            mask = []

        return data
    
    def capture_images(self, aux_output=None, mask_type=None): #alls # "corners"

        w_eval, h_eval = 1000, 1000   #500,500
        w_corners, h_corners = 1000, 1000   #500,500
        w_cnn, h_cnn = 1000,1000 #self.image_size
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
