import sys
import numpy as np
import pickle

sys.path.append('../mover_library/')
from samplers import *
from utils import *
from operator_utils.grasp_utils import solveIK, compute_one_arm_grasp

def generate_rand(min, max):
    return np.random.rand() * (max - min) + min



class OneArmPlaceUnif:
    def __init__(self, problem_env):
        self.env = problem_env.env
        self.problem_env = problem_env
        self.robot = self.env.GetRobots()[0]

    def place_robot_on_floor(self):
        FLOOR_Z = 0.13918
        trans = self.robot.GetTransform()
        trans[2, -1] = FLOOR_Z
        self.robot.SetTransform(trans)

    def predict(self, grasp_params, obj, obj_region, robot_region):
        if robot_region.name == 'entire_region':
            import pdb;pdb.set_trace()

        original_trans = self.robot.GetTransform()

        target_obj_region = obj_region
        target_robot_region = robot_region

        # use the same grasp parameters?

        T_r_wrt_o = np.dot(np.linalg.inv(obj.GetTransform()), self.robot.GetTransform())
        for _ in range(1000):
            release_obj(self.robot, obj)

            with self.robot:
                obj_xytheta = randomly_place_in_region(self.env, obj, target_obj_region)  # randomly place obj
                new_T_robot = np.dot(obj.GetTransform(), T_r_wrt_o)
                self.robot.SetTransform(new_T_robot)
                self.place_robot_on_floor()
                place_base_pose = get_body_xytheta(self.robot)
                if self.env.CheckCollision(self.robot):
                    continue

                # find an IK solution with grasp parameters; because I am maintaining my relative base pose, it should
                # be the same one as before
                grasps = compute_one_arm_grasp(depth_portion=grasp_params[2],
                                               height_portion=grasp_params[1],
                                               theta=grasp_params[0],
                                               obj=obj,
                                               robot=self.robot)

                for g in grasps:
                    g_config = self.robot.GetManipulator('rightarm_torso').FindIKSolution(g, 0)
                    if g_config is not None:
                        set_config(self.robot, g_config, self.robot.GetManipulator('rightarm_torso').GetArmIndices())
                        if not check_collision_except(self.robot, obj, self.env):
                            grab_obj(self.robot, obj) # put the robot state back to what it was before
                            return {'base_pose': place_base_pose, 'g_config': g_config}

        import pdb;pdb.set_trace()
        return {'base_pose': None, 'g_config': None}


