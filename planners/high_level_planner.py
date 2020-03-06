from planners.mcts import MCTS
from planners.constrained_mcts import ConstrainedMCTS
from generators.OneArmPickUniform import OneArmPickUnif
from generators.OneArmPlaceUniform import OneArmPlaceUnif

import sys
sys.path.append('../mover_library/')
from utils import draw_robot_at_conf, remove_drawn_configs, set_robot_config, grab_obj, pick_obj, \
    get_body_xytheta, one_arm_pick_obj, set_active_dof_conf
from utils  import *
import pickle
import numpy as np
import time



def pklsave(obj, name=''):
    pickle.dump(obj, open('tmp'+str(name)+'.pkl', 'wb'))


def pklload(name=''):
    return pickle.load(open('tmp'+str(name)+'.pkl', 'r'))


class HighLevelPlanner:
    def __init__(self, task_plan, problem_env, domain_name):
        self.task_plan = task_plan
        self.problem_env = problem_env
        self.domain_name = domain_name
        self.widening_parameter = None
        self.uct_parameter = None
        self.sampling_stategy = None

    def set_mcts_parameters(self, widening_parameter, uct_parameter, sampling_stategy):
        self.widening_parameter = widening_parameter
        self.uct_parameter = uct_parameter
        self.sampling_stategy = sampling_stategy

    def get_connecting_region(self, target_region):
        if target_region.name == 'home_region' or target_region.name == 'loading_region':
            return self.problem_env.regions['bridge_region']
        else:
            raise NotImplementedError

    def compute_object_ordering(self, objects):
        return objects

    def setup_fetch_problem(self, objects, target_region):
        # get a connecting region
        connecting_region = self.get_connecting_region(target_region)
        starting_region = self.problem_env.get_region_containing(objects[0])
        return starting_region, connecting_region

    @staticmethod
    def get_initial_pick_configs_from_fetch_plan(fetch_plan):
        initial_pick_configs = []
        for plan_step in fetch_plan:
            full_pick_config = plan_step[-2]['path'][0]
            obj_name = plan_step[-2]['obj_name']
            pick_base_config = plan_step[-1]['action']
            initial_pick_configs.append( {'obj_name':obj_name, 'pick_config':full_pick_config,
                                     'pick_base_config':pick_base_config})
        return initial_pick_configs

    def setup_packing_problem(self, fetch_plan):
        initial_pick_configs = self.get_initial_pick_configs_from_fetch_plan(fetch_plan)
        return initial_pick_configs

    def solve_fetching_single_obj(self, obj, target_region):
        # plans a pick and place for fetching obj to the target region
        # todo later, change the reward function to take account of the number of objects to clear
        self.problem_env.is_solving_fetching = True
        fetching_problem = [{'region': target_region, 'objects': [obj]}]
        mcts = MCTS(self.widening_parameter, self.uct_parameter, self.sampling_stategy,
                    self.problem_env, self.domain_name, fetching_problem)
        search_time_to_reward, fetch_plan, optimal_score_achieved = mcts.search(n_optimal_iter=10)
        self.problem_env.is_solving_fetching = False
        return fetch_plan

    def solve_ramo_for_single_obj(self, obj, pick_region, target_region):
        ramo_plan = None
        while ramo_plan is None:
            fetch_plan = self.solve_fetching_single_obj(obj, target_region)
            #import pdb;pdb.set_trace()
            #pklsave(fetch_plan, 'fetchplan')
            #fetch_plan = pklload('fetchplan')

            obstacles_in_way = self.problem_env.compute_new_namo_objects(obj, fetch_plan[0], fetch_plan[1])
            if len(obstacles_in_way) > 0:
                self.problem_env.initialize_ramo_problem(fetch_plan, obj, obstacles_in_way, pick_region, target_region)
                ramo_problem_for_obj = [{'region': pick_region, 'objects': obstacles_in_way}]
                self.problem_env.is_solving_ramo = True
                mcts = MCTS(self.widening_parameter, self.uct_parameter, self.sampling_stategy,
                            self.problem_env, self.domain_name, ramo_problem_for_obj)
                search_time_to_reward, ramo_plan, optimal_score_achieved = mcts.search(n_iter=30)
                self.problem_env.is_solving_ramo = False
            else:
                ramo_plan = []
        #import pdb;pdb.set_trace()
        ramo_plan = ramo_plan + fetch_plan
        return ramo_plan

    def solve_ramo(self, obj_plan, pick_region, connecting_region):
        # fetch first object
        ramo_plan = []
        stime = time.time()
        #ramo_plan = pklload()
        for idx, obj in enumerate(obj_plan):
            if idx < len(ramo_plan):
                obj_ramo_plan = ramo_plan[idx]
            else:
                obj_ramo_plan = self.solve_ramo_for_single_obj(obj, pick_region, connecting_region)
                ramo_plan.append(obj_ramo_plan)
                pklsave(ramo_plan)
            self.problem_env.apply_plan(obj_ramo_plan)
            obj.Enable(False)
            print "Time elapsed %.2f" % (time.time()-stime)
        #import pdb;pdb.set_trace()
        print "Done solving RAMO"
        return ramo_plan

    def make_packing_plan(self, objects, target_region, starting_configurations):
        subtask_plan = [{'region': target_region, 'objects': objects}]
        self.problem_env.objs_to_pack = objects
        self.problem_env.is_solving_packing = True
        mcts = ConstrainedMCTS(self.widening_parameter, self.uct_parameter, self.sampling_stategy,
                               self.problem_env, self.domain_name, subtask_plan, starting_configurations)
        search_time_to_reward, plan, optimal_score_achieved = mcts.search()
        return plan

    def apply_fetch_plan(self, plan):
        for p in plan:
            self.problem_env.apply_plan(p)

    def apply_packing_plan(self, fetch_plan, packing_plan):
        assert len(fetch_plan) == len(packing_plan),  'Length of packing and fetching should be equal'

        for fetch_step, pack_step in zip(fetch_plan, packing_plan):
            pick_full_conf = fetch_step[-2]['path'][0]
            obj_name = fetch_step[-2]['obj_name']
            pick_base_conf = fetch_step[-1]['action']
            self.problem_env.apply_pick_constraint(obj_name, pick_full_conf, pick_base_conf)
            self.problem_env.apply_plan([pack_step])

    def visualize_picking_and_packing_plans(self, fetch_plan, packing_plan):
        assert len(fetch_plan) == len(packing_plan),  'Length of packing and fetching should be equal'

        for fetch_step, pack_step in zip(fetch_plan, packing_plan):
            self.problem_env.visualize_plan(fetch_step)
            pick_full_conf = fetch_step[-2]['path'][0]
            obj_name = fetch_step[-2]['obj_name']
            pick_base_conf = fetch_step[-1]['action']
            self.problem_env.apply_pick_constraint(obj_name, pick_full_conf, pick_base_conf)
            self.problem_env.visualize_plan([pack_step])
            import pdb;pdb.set_trace()

    def search(self):
        for plan_step in self.task_plan:
            # get the first region
            target_packing_region = plan_step['region']

            # get the list of objects to be packed
            objects = plan_step['objects']

            # get the object ordering
            obj_plan = self.compute_object_ordering(objects)

            # setup fetch problem
            pick_region, connecting_region = self.setup_fetch_problem(objects, target_packing_region)

            # search the fetch problem
            ramo_plan = self.solve_ramo(obj_plan, pick_region, connecting_region)
            print "Done solving box packing problem"
            sys.exit(-1)
            #pklsave(ramo_plan)

            # perform fetch
            self.apply_fetch_plan(fetch_plan)
            #import pdb;pdb.set_trace()

            # setup the packing problem
            #starting_configs = self.setup_packing_problem(fetch_plan)
            #packing_plan = self.make_packing_plan(objects, target_packing_region, starting_configs)
            #pklsave(packing_plan, 'packing')
            packing_plan = pklload('packing')

            #self.visualize_picking_and_packing_plans(fetch_plan, packing_plan)
            self.apply_packing_plan(fetch_plan, packing_plan)

            # todo call these everytime we move a box
            self.problem_env.box_regions['rectangular_packing_box4'].draw(self.problem_env.env)
            self.problem_env.box_regions['rectangular_packing_box1'].draw(self.problem_env.env)
            self.problem_env.box_regions['rectangular_packing_box2'].draw(self.problem_env.env)
            self.problem_env.box_regions['rectangular_packing_box3'].draw(self.problem_env.env)

            self.problem_env.box_regions['square_packing_box4'].draw(self.problem_env.env)
            self.problem_env.box_regions['square_packing_box1'].draw(self.problem_env.env)
            self.problem_env.box_regions['square_packing_box2'].draw(self.problem_env.env)
            self.problem_env.box_regions['square_packing_box3'].draw(self.problem_env.env)

            target_obj = self.problem_env.shelf_objs['left_top'][0]
            for obj in self.problem_env.shelf_objs['left_top']:
                if obj != target_obj:
                    obj.Enable(False)
            for obj in self.problem_env.packing_boxes: obj.Enable(False)

            region = self.problem_env.regions['home_region']
            one_arm_pick_pi = OneArmPickUnif(self.problem_env.env, self.problem_env.robot)
            pick_action = one_arm_pick_pi.predict(target_obj, region)
            self.problem_env.apply_one_arm_pick_action(pick_action, target_obj, region, False, None)
            one_arm_place_pi = OneArmPlaceUnif(self.problem_env.env, self.problem_env.robot)
            place_action = one_arm_place_pi.predict(pick_action['grasp_params'], target_obj,
                                                    self.problem_env.box_regions['rectangular_packing_box1'],
                                                    self.problem_env.regions['home_region'])
            state, reward, plan = self.problem_env.apply_one_arm_place_action(place_action, target_obj, region, True,
                                                                              None)
            import pdb;pdb.set_trace()


            #plan, status = self.problem_env.get_base_motion_plan(pick_action['base_pose'])
            self.robot = self.problem_env.robot
            pick_action = {'base_pose': np.array([ 3.96807094,  0.90672245, -0.30283194]), 'g_config': np.array([ 0.14806586,  0.05496191, -0.33972887,  0.        , -0.52787472,
       -2.14329968, -1.39698209,  2.55163376]), 'grasp_params': [1.7855677087892705, 0.5310030378042822, 0.941896718400509]}


            # these three go in to the feasibility check for one-arm motion
            plan, status = self.problem_env.get_base_motion_plan(pick_action['base_pose'])
            set_robot_config(pick_action['base_pose'], self.problem_env.robot)
            plan, status = self.problem_env.get_arm_motion_plan(pick_action['g_config'])
            visualize_path(self.robot, plan)
            import pdb;pdb.set_trace()
            # todo plan an arm motion


            import pdb;pdb.set_trace()
            # todo plan a path

            one_arm_pick_obj(target_obj, self.problem_env.robot, pick_action[-1])


            import pdb;pdb.set_trace()






