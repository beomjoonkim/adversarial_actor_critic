import numpy as np
import sys
import socket
import os
import random
import copy

from openravepy import DOFAffine

## NAMO problem environment
from problem_environment import TwoArmProblemEnvironment #, DynamicEnvironmentStateSaverWithCurrObj
from NAMO_problem import NAMO_problem

## openrave_wrapper imports
from manipulation.constants import PARALLEL_LEFT_ARM, REST_LEFT_ARM, HOLDING_LEFT_ARM, FOLDED_LEFT_ARM, \
    FAR_HOLDING_LEFT_ARM, LOWER_TOP_HOLDING_LEFT_ARM, REGION_Z_OFFSET
from manipulation.primitives.utils import mirror_arm_config
from manipulation.bodies.bodies import set_color
from manipulation.primitives.transforms import set_config

## mover library utility functions
sys.path.append('../mover_library/')
from utils import set_robot_config, get_body_xytheta, pick_obj, check_collision_except, place_obj, grab_obj, \
    simulate_path
from operator_utils.grasp_utils import solveTwoArmIKs, compute_two_arm_grasp

OBJECT_ORIGINAL_COLOR = (0, 0, 0)
COLLIDING_OBJ_COLOR = (0, 1, 1)
TARGET_OBJ_COLOR = (1, 0, 0)


class NAMO(TwoArmProblemEnvironment):
    def __init__(self, problem_idx):
        TwoArmProblemEnvironment.__init__(self)

        self.manual_pinst = self.get_p_inst(problem_idx)

        while self.problem is None:
            self.problem = NAMO_problem(self.env, self.manual_pinst)
        self.robot = self.env.GetRobots()[0]


        self.objects = self.problem['objects']
        for obj in self.objects:
            set_color(obj, OBJECT_ORIGINAL_COLOR)

        self.target_obj = self.problem['target_obj']
        set_color(self.target_obj, TARGET_OBJ_COLOR)

        self.objs_to_move = self.compute_obj_collisions()
        for obj_name in self.objs_to_move:
            set_color(self.env.GetKinBody(obj_name), COLLIDING_OBJ_COLOR)
        self.init_objs_to_move = copy.deepcopy(self.objs_to_move)
        self.curr_obj = self.env.GetKinBody(self.objs_to_move[0])

        self.robot_region = self.problem['all_region']
        self.obj_region = self.problem['all_region']
        self.init_base_conf = np.array([-1, 1, 0])
        self.infeasible_reward = -2
        self.optimal_score = len(self.init_objs_to_move)

        #self.init_saver = DynamicEnvironmentStateSaverWithCurrObj(self.env, self.placements, self.curr_obj,
        #                                                          self.is_pick_time())
        #self.original_init_saver = self.init_saver
        self.is_init_pick_node = True

    def reset_to_init_state(self):
        self.init_saver.Restore()
        self.curr_state = self.get_state()

        self.objs_to_move = self.init_objs_to_move
        self.curr_obj = self.env.GetKinBody(self.objs_to_move[0])
        if not self.is_init_pick_node:
            grab_obj(self.robot, self.curr_obj)
        self.robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
        set_robot_config(self.init_base_conf, self.robot)

    def compute_obj_collisions(self):
        robot = self.robot

        obj_names = []
        leftarm_manip = robot.GetManipulator('leftarm')
        rightarm_manip = robot.GetManipulator('rightarm')

        env = self.env
        path = self.problem['original_path']
        objs = self.objects

        assert len(robot.GetGrabbed()) == 0, "Robot must not hold anything to check obj collisions"

        path_reduced = path
        while len(path_reduced) > 500:
            path_reduced = path_reduced[0:len(path_reduced)-1:2]  # halves the path length

        with robot:
            set_config(robot, FOLDED_LEFT_ARM, leftarm_manip.GetArmIndices())
            set_config(robot, mirror_arm_config(FOLDED_LEFT_ARM),
                       rightarm_manip.GetArmIndices())
            for p in path_reduced:
                # This would not work if the robot is holding the object
                set_robot_config(p, robot)
                for obj in objs:
                    if env.CheckCollision(robot, obj) and not (obj.GetName() in obj_names):
                        obj_names.append(obj.GetName())
                        # set_color(obj, COLLIDING_OBJ_COLOR  )
                    # else:
                    # import pdb;pdb.set_tracce()
                    # set_color(obj,OBJECT_ORIGINAL_COLOR)
        print 'Collisions: ', obj_names
        return obj_names

    def get_p_inst(self, problem_idx):
        if socket.gethostname() == 'dell-XPS-15-9560':
            self.problem_dir = '../AdvActorCriticNAMOResults/problems/'
            n_probs_threshold = 900
        else:
            self.problem_dir = '/data/public/rw/pass.port//NAMO/problems/'
            n_probs_threshold = 8500

        if problem_idx is None:
            probs_created = [f for f in os.listdir(self.problem_dir) if f.find('.pkl') != -1]
            n_probs_created = len(probs_created)
            is_need_more_problems = n_probs_threshold > n_probs_created
            if not is_need_more_problems:
                print "Using problem ", self.problem_dir + probs_created[random.randint(1, n_probs_created - 1)]
                return self.problem_dir + probs_created[np.random.randint(n_probs_threshold - 1)]
            else:
                return self.problem_dir + 'prob_inst_' + str(n_probs_created) + '.pkl'
        else:
            # for allowing us to check the solution
            pfiles = [pfile for pfile in os.listdir(self.problem_dir) if pfile.find('.pkl') != -1]
            pfile_idxs = [int((pfile.split('_')[-1]).split('.pkl')[0]) for pfile in pfiles if pfile.find('.pkl')!=-1]
            pfiles = np.array(pfiles)[np.argsort(pfile_idxs)]
            pfile_name = pfiles[problem_idx]
            return self.problem_dir + pfile_name

    def apply_pick_action(self, action, obj=None, do_check_reachability=True):
        if action is None:
            curr_state = self.get_state()
            return curr_state, self.infeasible_reward, None, None

        leftarm_manip = self.robot.GetManipulator('leftarm')
        rightarm_torso_manip = self.robot.GetManipulator('rightarm_torso')

        if obj is None:
            obj_to_pick = self.curr_obj
        else:
            obj_to_pick = obj
        pick_base_pose, grasp_params = action
        path, is_action_feasible = self.check_action_feasible(pick_base_pose, do_check_reachability)
        if is_action_feasible:
            set_robot_config(pick_base_pose, self.robot)
            grasps = compute_two_arm_grasp(depth_portion=grasp_params[2],
                                           height_portion=grasp_params[1],
                                           theta=grasp_params[0],
                                           obj=obj_to_pick,
                                           robot=self.robot)
            g_config = solveTwoArmIKs(self.env, self.robot, obj_to_pick, grasps)
            if g_config is None:
                curr_state = self.get_state()
                return curr_state, self.infeasible_reward, None, None

            pick_obj(obj_to_pick, self.robot, g_config, leftarm_manip, rightarm_torso_manip)
            curr_state = self.get_state()
            reward = 0
            return curr_state, reward, g_config, path
        else:
            curr_state = self.get_state()
            return curr_state, self.infeasible_reward, None, None

    def apply_place_action(self, action, do_check_reachability=True):
        #  todo make the namo domain work
        path, is_action_feasible = self.check_action_feasible(action, do_check_reachability)
        if is_action_feasible:
            place_robot_pose = action[0, :]
            prev_objs_to_move = copy.deepcopy(self.objs_to_move)
            self.place_object(place_robot_pose)
            self.update_next_obj_to_pick(action)
            n_moved = len(prev_objs_to_move) - len(self.objs_to_move)
            if n_moved == 0:
                reward = -1
            else:
                reward = 1
            return self.curr_state, reward, path
        else:
            return self.curr_state, self.infeasible_reward, path

    def update_next_obj_to_pick(self, place_action):
        self.placements.append(place_action) # should I only save the ones that moved
        self.objs_to_move = self.compute_obj_collisions()
        if len(self.objs_to_move) > 0:
            self.curr_obj = self.env.GetKinBody(self.objs_to_move[0])
        else:
            self.curr_obj = None

    def is_goal_reached(self):
        return len(self.objs_to_move) == 0

    def set_init_state(self, saver):
        self.init_saver = saver
        self.initial_placements = copy.deepcopy(saver.placements)
        self.is_init_pick_node = saver.is_pick_node
        self.init_saver.Restore()
        self.init_base_conf = get_body_xytheta(self.robot).squeeze()
        self.objs_to_move = self.compute_obj_collisions()
        self.init_objs_to_move = copy.deepcopy(self.objs_to_move)

    def replay_plan(self, plan):
        self.original_init_saver.Restore()
        leftarm_manip = self.robot.GetManipulator('leftarm')
        rightarm_torso_manip = self.robot.GetManipulator('rightarm_torso')

        objs_to_move = self.compute_obj_collisions()
        curr_obj = self.env.GetKinBody(objs_to_move[0])

        for plan_step in plan:
            if len(plan_step)==0:
                break
            is_pick_step = len(plan_step[0]) == 2
            if is_pick_step:
                grasp_and_path = plan_step[1]
                g_config = grasp_and_path[0]
                pick_path = grasp_and_path[1]
                simulate_path(self.robot, pick_path, 0.1)
                pick_obj(curr_obj, self.robot, g_config, leftarm_manip, rightarm_torso_manip)
            else:
                place_path = plan_step[1]
                simulate_path(self.robot, place_path, 0.1)
                self.place_object(place_path[-1])
                objs_to_move = self.compute_obj_collisions()
                curr_obj = self.env.GetKinBody(objs_to_move[0])

            raw_input('Press a key to see the next step')








