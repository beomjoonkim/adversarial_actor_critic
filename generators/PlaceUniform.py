import sys
import numpy as np
import pickle

sys.path.append('../mover_library/')
from samplers import *
from utils import *


def generate_rand(min, max):
    return np.random.rand() * (max - min) + min


class PlaceUnif:
    def __init__(self, problem_env):
        self.problem_env = problem_env
        self.env = problem_env.env
        self.robot = self.env.GetRobots()[0]
        self.robot_region = self.problem_env.regions['entire_region']

    def predict(self, obj, obj_region):
        #todo does it even enter here?
        original_trans = self.robot.GetTransform()
        T_r_wrt_o = np.dot(np.linalg.inv(obj.GetTransform()), self.robot.GetTransform())

        target_obj_region = obj_region
        target_robot_region = self.robot_region
        original = get_body_xytheta(self.robot)
        for _ in range(1000):
            release_obj(self.robot, obj)
            self.robot.SetTransform(original_trans)
            with self.robot:
                obj_xytheta = randomly_place_in_region(self.env, obj, target_obj_region)  # randomly place obj

                # compute the resulting robot transform
                new_T_robot = np.dot(obj.GetTransform(), T_r_wrt_o)
                self.robot.SetTransform(new_T_robot)
                self.robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
                robot_xytheta = self.robot.GetActiveDOFValues()
                set_robot_config(robot_xytheta, self.robot)
                new_T = self.robot.GetTransform()
                assert (np.all(np.isclose(new_T, new_T_robot)))
                if not (check_collision_except(obj, self.env)) \
                        and (target_robot_region.contains(self.robot.ComputeAABB())):
                    grab_obj(self.robot, obj)
                    return {'operator_name': 'two_arm_place', 'base_pose': robot_xytheta}
        import pdb;pdb.set_trace()
        return None


