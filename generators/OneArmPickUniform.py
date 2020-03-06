import sys
import numpy as np
from manipulation.bodies.bodies import set_config

sys.path.append('../mover_library/')
from samplers import sample_pick, sample_grasp_parameters, sample_ir, sample_ir_multiple_regions, sample_one_arm_grasp_parameters

from utils import compute_occ_vec, set_robot_config, remove_drawn_configs, \
    draw_configs, clean_pose_data, draw_robot_at_conf, \
    pick_obj, place_obj, check_collision_except, one_arm_pick_obj, release_obj, one_arm_place_obj, set_config

sys.path.append('../mover_library/')
from operator_utils.grasp_utils import solveIK, compute_one_arm_grasp
from openravepy import IkFilterOptions, IkReturnAction

def check_collision_except_obj(obj, robot, env):
    in_collision = (check_collision_except(obj, robot, env)) \
                   or (check_collision_except(robot, obj, env))
    return in_collision


class PickUnif(object):
    def __init__(self, problem_env):
        self.env = problem_env.env
        self.robot = self.env.GetRobots()[0]

    def predict(self, obj, region):
        raise NotImplementedError


class OneArmPickUnif(PickUnif):
    def __init__(self, env):
        PickUnif.__init__(self, env)

    def predict(self, obj, region):
        rightarm_torso_manip = self.robot.GetManipulator('rightarm_torso')

        for iter in range(1000):
            pick_base_pose = None
            while pick_base_pose is None:
                with self.robot:
                    pick_base_pose = sample_ir(obj, self.robot, self.env, region)
            theta, height_portion, depth_portion = sample_one_arm_grasp_parameters()
            grasp_params = [theta[0], height_portion[0], depth_portion[0]]

            with self.robot:
                set_robot_config(pick_base_pose, self.robot)
                grasps = compute_one_arm_grasp(depth_portion=grasp_params[2],
                                               height_portion=grasp_params[1],
                                               theta=grasp_params[0],
                                               obj=obj,
                                               robot=self.robot)
                for g in grasps:
                    g_config = rightarm_torso_manip.FindIKSolution(g, 0)
                    if g_config is not None:
                        set_config(self.robot, g_config, self.robot.GetManipulator('rightarm_torso').GetArmIndices())
                        if not check_collision_except(self.robot, obj, self.env):
                            pick_params = {'operator_name': 'one_arm_pick', 'base_pose': pick_base_pose, 'grasp_params': grasp_params, 'g_config':g_config}
                            return pick_params
                        one_arm_place_obj(obj, self.robot)

        pick_params = {'operator_name': 'one_arm_pick', 'base_pose': pick_base_pose, 'grasp_params': grasp_params, 'g_config': g_config}
        return pick_params



