import numpy as np
import sys
import copy
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../mover_library/')
from conveyor_belt_problem import two_tables_through_door
from problem_environment import TwoArmProblemEnvironment #, DynamicEnvironmentStateSaverWithCurrObj
from utils import compute_occ_vec, set_robot_config, remove_drawn_configs, \
    draw_configs, clean_pose_data, draw_robot_at_conf, \
    pick_obj, place_obj, check_collision_except, release_obj, grab_obj


from motion_planner import collision_fn, base_extend_fn, base_sample_fn, base_distance_fn, smooth_path, rrt_connect


from utils import *
from operator_utils.grasp_utils import solveTwoArmIKs, compute_two_arm_grasp

from openravepy import *
import time


class ConveyorBelt(TwoArmProblemEnvironment):
    def __init__(self):
        TwoArmProblemEnvironment.__init__(self)
        self.problem = two_tables_through_door(self.env)
        self.objects = self.problem['objects']
        self.init_base_conf = np.array([0, 1.05, 0])
        self.robot_region = self.problem['all_region']
        self.obj_region = self.problem['loading_region']
        self.robot = self.problem['env'].GetRobots()[0]
        self.infeasible_reward = -2
        self.optimal_score = 5

        self.curr_obj = self.objects[0]
        self.curr_state = self.get_state()
        self.objs_to_move = self.objects

        #self.init_saver = DynamicEnvironmentStateSaverWithCurrObj(self.env, self.get_placements(), self.curr_obj,
        #                                                          self.which_operator())
        self.is_init_pick_node = True

    def apply_pick_action(self, action, obj=None, do_check_reachability=False):
        leftarm_manip = self.robot.GetManipulator('leftarm')
        rightarm_torso_manip = self.robot.GetManipulator('rightarm_torso')

        if obj is None:
            obj_to_pick = self.curr_obj
        else:
            obj_to_pick = obj
        pick_base_pose, grasp_params = action
        set_robot_config(pick_base_pose, self.robot)
        grasps = compute_two_arm_grasp(depth_portion=grasp_params[2],
                                       height_portion=grasp_params[1],
                                       theta=grasp_params[0],
                                       obj=obj_to_pick,
                                       robot=self.robot)
        g_config = solveTwoArmIKs(self.env, self.robot, obj_to_pick, grasps)
        assert g_config is not None

        pick_obj(obj_to_pick, self.robot, g_config, leftarm_manip, rightarm_torso_manip)
        set_robot_config(self.init_base_conf, self.robot)
        curr_state = self.get_state()
        reward = 0
        pick_path = None
        return curr_state, reward, g_config, pick_path

    def update_next_obj_to_pick(self, place_action):
        self.placements.append(place_action)
        if len(self.placements) < len(self.objects):
            self.curr_obj = self.objs_to_move[len(self.placements)]  # update the next object to be picked

    def apply_place_action(self, action, do_check_reachability=True):
        # todo should this function tell you that it is a terminal state?
        path, is_action_feasible = self.check_action_feasible(action, do_check_reachability)
        if is_action_feasible:
            place_robot_pose = action[0, :]
            self.place_object(place_robot_pose)
            self.curr_state = self.get_state()
            self.update_next_obj_to_pick(action)
            reward = 1

            is_goal_state = len(self.placements) == len(self.objects)
            if is_goal_state:
                return self.curr_state, reward, path
            return self.curr_state, reward, path
        else:
            return self.curr_state, self.infeasible_reward, path

    def reset_to_init_state(self):
        self.init_saver.Restore()
        self.curr_state = self.get_state()
        self.placements = copy.deepcopy(self.initial_placements)

        self.curr_obj = self.objs_to_move[len(self.placements)] # todo change it for NAMO domain
        if not self.init_which_opreator != 'two_arm_pick':
            grab_obj(self.robot, self.curr_obj)
        self.robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])

    def is_goal_reached(self):
        return len(self.placements) == 5

    def set_init_state(self, saver):
        self.init_saver = saver
        self.initial_placements = copy.deepcopy(saver.placements)
        # I think you should restore first?
        self.init_which_opreator = saver.which_operator
        self.objs_to_move = self.objects[len(saver.placements):]
