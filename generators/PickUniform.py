import sys
import numpy as np

sys.path.append('../mover_library/')
from mover_library.samplers import sample_pick, sample_grasp_parameters, sample_ir, sample_ir_multiple_regions

from mover_library.utils import compute_occ_vec, set_robot_config, remove_drawn_configs, \
    draw_configs, clean_pose_data, draw_robot_at_conf, \
    pick_obj, place_obj, check_collision_except

sys.path.append('../mover_library/')
from mover_library.operator_utils.grasp_utils import solveTwoArmIKs, compute_two_arm_grasp


#def check_collision_except_obj(obj, robot, env):
#    in_collision = (check_collision_except(obj, robot, env)) \
#                   or (check_collision_except(robot, obj, env))
#    return in_collision


class PickUnif(object):
    def __init__(self, problem_env):
        self.problem_env = problem_env
        self.env = problem_env.env
        self.robot = self.env.GetRobots()[0]

    def predict(self, obj, region):
        raise NotImplementedError


class PickWithoutBaseUnif(PickUnif):
    def __init__(self, problem_env, robot, init_base_conf):
        PickUnif.__init__(self, problem_env, robot)
        self.init_base_conf = init_base_conf

    def predict(self, obj, region):
        leftarm_manip = self.robot.GetManipulator('leftarm')
        rightarm_manip = self.robot.GetManipulator('rightarm')
        rightarm_torso_manip = self.robot.GetManipulator('rightarm_torso')

        for iter in range(1000):
            pick_base_pose = None
            while pick_base_pose is None:
                with self.robot:
                    pick_base_pose = sample_ir(obj, self.robot, self.env, region)
            theta, height_portion, depth_portion = sample_grasp_parameters()
            grasp_params = [theta[0], height_portion[0], depth_portion[0]]

            with self.robot:
                set_robot_config(pick_base_pose, self.robot)
                grasps = compute_two_arm_grasp(depth_portion=grasp_params[2],
                                               height_portion=grasp_params[1],
                                               theta=grasp_params[0],
                                               obj=obj,
                                               robot=self.robot)

                g_config = solveTwoArmIKs(self.env, self.robot, obj, grasps)
                if g_config is not None:
                    pick_obj(obj, self.robot, g_config, leftarm_manip, rightarm_torso_manip)
                    set_robot_config(self.init_base_conf, self.robot)
                    if not check_collision_except(obj, self.env):
                        set_robot_config(pick_base_pose, self.robot)
                        place_obj(obj, self.robot, leftarm_manip, rightarm_manip)
                        action = (pick_base_pose, grasp_params)
                        return action

                    set_robot_config(pick_base_pose, self.robot)
                    place_obj(obj, self.robot, leftarm_manip, rightarm_manip)
                else:
                    print 'gconfig none'

        return None


class PickWithBaseUnif(PickUnif):
    def __init__(self, problem_env):
        PickUnif.__init__(self, problem_env)

    def compute_grasp_config(self, obj, pick_base_pose, grasp_params):
        set_robot_config(pick_base_pose, self.robot)
        grasps = compute_two_arm_grasp(depth_portion=grasp_params[2],
                                       height_portion=grasp_params[1],
                                       theta=grasp_params[0],
                                       obj=obj,
                                       robot=self.robot)

        g_config = solveTwoArmIKs(self.env, self.robot, obj, grasps)
        return g_config

    def compute_grasp_action(self, obj, region, n_iter=100):
        leftarm_manip = self.robot.GetManipulator('leftarm')
        rightarm_manip = self.robot.GetManipulator('rightarm')
        rightarm_torso_manip = self.robot.GetManipulator('rightarm_torso')

        for iter in range(n_iter):
            pick_base_pose = None
            while pick_base_pose is None:
                with self.robot:
                    pick_base_pose = sample_ir(obj, self.robot, self.env, region)
            theta, height_portion, depth_portion = sample_grasp_parameters()
            grasp_params = np.array([theta[0], height_portion[0], depth_portion[0]])

            with self.robot:
                g_config = self.compute_grasp_config(obj, pick_base_pose, grasp_params)
                if g_config is not None:
                    pick_obj(obj, self.robot, g_config, leftarm_manip, rightarm_torso_manip)
                    if not check_collision_except(obj, self.env):
                        set_robot_config(pick_base_pose, self.robot)
                        place_obj(obj, self.robot, leftarm_manip, rightarm_manip)
                        pick_params = {'operator_name': 'two_arm_pick', 'base_pose': pick_base_pose, 'grasp_params': grasp_params, 'g_config': g_config}
                        return pick_params

                    set_robot_config(pick_base_pose, self.robot)
                    place_obj(obj, self.robot, leftarm_manip, rightarm_manip)

        print "Sampling pick failed"
        pick_params = {'operator_name': 'two_arm_pick', 'base_pose': pick_base_pose, 'grasp_params': grasp_params, 'g_config': g_config}
        return pick_params

    def predict(self, obj, region):
        if self.problem_env.is_solving_ramo:
            pick_params = self.compute_grasp_action(obj, region, n_iter=100)
        else:
            pick_params = self.compute_grasp_action(obj, region, n_iter=10)

        #pick_params = {}
        #pick_params['g_config'] = None
        if self.problem_env.is_solving_fetching and pick_params['g_config'] is None:
            self.problem_env.disable_objects_in_region(region.name)
            obj.Enable(True)
            pick_params = self.compute_grasp_action(obj, region, n_iter=100)
            self.problem_env.enable_objects_in_region(region.name)
        #if pick_params['g_config'] is None:
        #    import pdb;pdb.set_trace()
        return pick_params





