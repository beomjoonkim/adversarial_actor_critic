import copy
import numpy as np

from manipulation.constants import PARALLEL_LEFT_ARM, REST_LEFT_ARM, HOLDING_LEFT_ARM, FOLDED_LEFT_ARM, FAR_HOLDING_LEFT_ARM, LOWER_TOP_HOLDING_LEFT_ARM, REGION_Z_OFFSET
from manipulation.primitives.utils import mirror_arm_config

from problem_environment import ProblemEnvironment #, DynamicEnvironmentStateSaverWithCurrObj
from mover_problem import mover_problem
from openravepy import DOFAffine
from utils import draw_robot_at_conf, remove_drawn_configs, set_robot_config, grab_obj, pick_obj, get_body_xytheta, \
    visualize_path, check_collision_except, place_obj, one_arm_pick_obj, set_active_dof_conf, release_obj, one_arm_place_obj
from operator_utils.grasp_utils import solveTwoArmIKs, compute_two_arm_grasp
from constraint_removal_planners.constraint_removal_planner import ConstraintRemovalPlanner
from manipulation.regions import create_region, AARegion

from manipulation.primitives.savers import DynamicEnvironmentStateSaver


class Mover(ProblemEnvironment):
    def __init__(self):
        ProblemEnvironment.__init__(self)
        self.problem_config = mover_problem(self.env)
        self.infeasible_reward = -2

        self.regions = {}
        self.regions['home_region'] = self.problem_config['home_region']
        self.regions['loading_region'] = self.problem_config['loading_region']
        self.regions['entire_region'] = self.problem_config['entire_region']
        self.regions['bridge_region'] = self.problem_config['bridge_region']

        self.box_regions = self.problem_config['box_regions']
        self.shelf_objs = self.problem_config['shelf_objects']
        self.small_objs = self.problem_config['objects_to_pack']
        self.big_objs = self.problem_config['big_objects_to_pack']
        self.packing_boxes = self.problem_config['packing_boxes']

        #for p in self.packing_boxes[1:]:
        #    self.env.Remove(p)

        # related to constraint-removal subproblem
        self.is_solving_ramo = False
        self.curr_namo_object_names = []
        self.init_namo_object_names = []

        # related to packing subproblem
        self.is_solving_packing = False
        self.is_solving_fetching = False
        self.objs_to_pack = []
        self.fetch_base_config = None

        self.robot = self.env.GetRobots()[0]
        self.is_boxes_in_home = False
        self.is_big_objs_in_truck = False
        self.is_small_objs_in_boxes = False
        self.is_shelf_objs_in_boxes = False
        self.constraint_removal_planner = ConstraintRemovalPlanner(self)

    def update_box_region(self, box):
        box_region = AARegion.create_on_body(box)
        box_region.color = (1., 1., 0., 0.25)
        self.box_regions[box.GetName()] = box_region

    def reset_to_init_state(self, node):
        saver = node.state_saver
        saver.Restore()
        obj = node.obj
        if node.operator.find('pick') == -1 and not self.is_solving_packing:
            grab_obj(self.robot, obj)

        # todo
        #   I need to reset the namo object list
        if self.is_solving_ramo:
            self.curr_namo_object_names = copy.deepcopy(self.init_namo_object_names)
        if self.is_solving_packing:
            for obj in self.objs_to_pack:
                obj.Enable(False)
        self.robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])

    def which_operator(self, obj):
        # todo put assert statement on the obj type
        if obj in self.big_objs + self.packing_boxes:
            if self.is_pick_time():
                return 'two_arm_pick'
            else:
                return 'two_arm_place'
        else:
            if self.is_pick_time():
                return 'one_arm_pick'
            else:
                return 'one_arm_place'

    def apply_two_arm_pick_action(self, action, obj, region, check_feasibility, parent_motion):
        base_pose = action['base_pose']
        grasp_params = action['grasp_params']
        g_config = action['g_config']

        if g_config is None:
            curr_state = self.get_state()
            return curr_state, self.infeasible_reward, None

        leftarm_manip = self.robot.GetManipulator('leftarm')
        rightarm_torso_manip = self.robot.GetManipulator('rightarm_torso')

        obj_to_pick = obj
        if check_feasibility:
            motion, status = self.check_base_motion_plan_exists(base_pose, obj, region)
        else:
            motion = parent_motion
            status = 'HasSolution'

        if status == 'HasSolution':
            if self.is_solving_fetching:
                reward = np.exp(-len(self.get_objs_in_collision(motion, region.name)))
            else:
                reward = 0

            set_robot_config(base_pose, self.robot)
            pick_obj(obj_to_pick, self.robot, g_config, leftarm_manip, rightarm_torso_manip)
            curr_state = self.get_state()
            return curr_state, reward, motion
        else:
            curr_state = self.get_state()
            return curr_state, self.infeasible_reward, None

    def initialize_ramo_problem(self, fetch_plan, target_obj, obstacles_in_way, fetching_region, target_region):
        self.is_solving_ramo = True  # this controls the transition model and the reward of the environment
        self.prefetching_robot_config = get_body_xytheta(self.robot).squeeze()
        self.curr_namo_object_names = [obj.GetName() for obj in obstacles_in_way]
        self.init_namo_object_names = [obj.GetName() for obj in obstacles_in_way]

        self.namo_pick_op_instance = fetch_plan[0]
        self.namo_place_op_instance = fetch_plan[1]
        self.namo_target_obj = target_obj
        self.fetching_region = fetching_region
        self.target_region = target_region

        """
        self.pick_path = fetch_plan[0]['path']
        self.place_path = fetch_plan[1]['path']
        self.pick_region = pick_region
        self.target_region = target_region

        self.namo_target_obj = target_obj
        self.pick_config = pick_op_instance['action']['base_pose']
        self.pick_g_config = pick_op_instance['action']['g_config']
        """

    def two_arm_pick_obj(self, obj, pick_action):
        base_pose = pick_action['base_pose']
        g_config = pick_action['g_config']
        set_robot_config(base_pose, self.robot)
        leftarm_manip = self.robot.GetManipulator('leftarm')
        rightarm_torso_manip = self.robot.GetManipulator('rightarm_torso')
        pick_obj(obj, self.robot, g_config, leftarm_manip, rightarm_torso_manip)

    def compute_new_namo_objects(self, target_obj, pick_op_instance, place_op_instance):
        #was_target_obj_held = len(self.robot.GetGrabbed()) != 0 and self.robot.GetGrabbed()[0] == self.namo_target_obj
        held_obj = self.robot.GetGrabbed()
        picking_region = self.get_region_containing(target_obj)
        curr_base_pose = get_body_xytheta(self.robot)
        curr_robot_config = self.robot.GetDOFValues()

        # did I move
        self.two_arm_pick_obj(target_obj, pick_op_instance['action'])
        place_motion_collisions = self.get_objs_in_collision(place_op_instance['path'], picking_region.name)

        self.place_object(pick_op_instance['action']['base_pose'])
        pick_motion_collisions = self.get_objs_in_collision(pick_op_instance['path'], picking_region.name)

        # restore previous robot state
        set_robot_config(curr_base_pose, self.robot)
        self.robot.SetDOFValues(curr_robot_config)
        if len(held_obj) > 0:
            grab_obj(self.robot, held_obj[0])

        # todo do not add duplicates
        # collisions = place_motion_collisions + [p for p in pick_motion_collisions if p not in place_motion_collisions]
        collisions = [p for p in pick_motion_collisions if p not in place_motion_collisions] + place_motion_collisions
        return collisions

    def apply_two_arm_place_action(self, action, obj, target_region, check_feasibility, parent_motion):
        base_pose = action['base_pose']
        pick_conf = self.robot.GetDOFValues()
        curr_state = self.get_state()
        fetching_region = self.get_region_containing(obj)
        pick_base_pose = get_body_xytheta(self.robot)
        #if fetching_region is None or target_region is None or self.regions['entire_region'] is None:
        #    import pdb;pdb.set_trace()

        if check_feasibility:
            if self.is_solving_ramo:
                prev_objs_to_move = copy.deepcopy(self.curr_namo_object_names)
                self.place_object(base_pose)
                collided_objs = self.compute_new_namo_objects(self.namo_target_obj,
                                                              self.namo_pick_op_instance,
                                                              self.namo_place_op_instance)
                self.curr_namo_object_names = [col_obj.GetName() for col_obj in collided_objs]

                n_moved = len(prev_objs_to_move) - len(self.curr_namo_object_names)
                assert n_moved >= 0, 'Number of objects cannot be negative'
                self.robot.SetDOFValues(pick_conf)
                grab_obj(self.robot, obj)
                set_robot_config(pick_base_pose, self.robot)
                if n_moved == 0:
                    print "NAMO, has solution, but not cleared"
                    return curr_state, self.infeasible_reward, None
            plan, status = self.check_base_motion_plan_exists(base_pose, obj, target_region)

            if self.is_solving_ramo:
                plan = {'plan_to_place': plan}

                if status == 'HasSolution':
                    with self.robot:
                        self.place_object(base_pose)
                        plan_to_prefetch, status_for_prefetch = self.check_base_motion_plan_exists(
                            self.prefetching_robot_config, obj, self.regions['entire_region'])
                        self.robot.SetDOFValues(pick_conf)
                        grab_obj(self.robot, obj)
                    plan['plan_to_prefetch'] = plan_to_prefetch
                    status = status_for_prefetch
        else:
            status = 'HasSolution'
            plan = parent_motion

        if status == 'HasSolution':
            if self.is_solving_fetching:
                reward = np.exp(-len(self.get_objs_in_collision(plan, fetching_region.name)))
            else:
                reward = 1
            self.place_object(base_pose, obj)
            return curr_state, reward, plan
        else:
            if self.is_solving_ramo:
                self.curr_namo_object_names = prev_objs_to_move
            curr_state = self.get_state()
            return curr_state, self.infeasible_reward, None

    def check_base_motion_plan_exists(self, goal_robot_xytheta, obj, target_region):
        curr_robot_xytheta = get_body_xytheta(self.robot)
        curr_region = self.get_region_containing(obj)
        try:
            motion_planning_region_name = target_region.name if curr_region.name == target_region.name else 'entire_region'
        except:
            import pdb;pdb.set_trace()

        if self.check_base_pose_feasible(goal_robot_xytheta, obj, target_region):
            motion, status = self.get_base_motion_plan(goal_robot_xytheta, motion_planning_region_name)
            if status == 'NoPath' and self.is_solving_fetching:
                self.disable_objects_in_region(curr_region.name)
                obj.Enable(True)
                motion, status = self.get_base_motion_plan(goal_robot_xytheta, motion_planning_region_name)
                self.enable_objects_in_region(curr_region.name)
        elif self.is_solving_fetching:
            self.disable_objects_in_region(curr_region.name)
            obj.Enable(True)
            if self.check_base_pose_feasible(goal_robot_xytheta, obj, target_region):
                motion, status = self.get_base_motion_plan(goal_robot_xytheta, motion_planning_region_name)
            else:
                motion = None
                status = 'NoPath'
            self.enable_objects_in_region(curr_region.name)
        else:
            motion = None
            status = 'NoPath'

        return motion, status

    def check_base_pose_feasible(self, base_pose, obj, region):
        if region.name == 'bridge_region':
            if not self.is_collision_at_base_pose(base_pose, obj) \
                    and self.is_in_region_at_base_pose(base_pose, obj, robot_region=self.regions['entire_region'],
                                                   obj_region=region):
                return True
        else:
            if not self.is_collision_at_base_pose(base_pose, obj) \
                    and self.is_in_region_at_base_pose(base_pose, obj, robot_region=region,
                                                       obj_region=region):
                return True

        return False

    def check_one_arm_pick_feasible(self, obj, base_pose, g_config, region):
        if self.check_base_pose_feasible(base_pose, obj, region):
            base_motion, base_plan_status = self.get_base_motion_plan(base_pose)
            if base_plan_status == 'HasSolution':
                with self.robot:
                    set_robot_config(base_pose, self.robot)
                    obj.Enable(False)
                    arm_motion, arm_plan_status = self.get_arm_motion_plan(g_config)
                    obj.Enable(True)
                    # todo plan back to the folded_arm location
            if base_plan_status == 'HasSolution' and arm_plan_status == 'HasSolution':
                return {'base_motion': base_motion, 'arm_motion': arm_motion}, 'HasSolution'

        return None, 'NoPath'

    def apply_one_arm_pick_action(self, action, obj, region, check_feasibility, parent_motion):
        pick_base_pose = action['base_pose']
        g_config = action['g_config']

        if g_config is None:
            curr_state = self.get_state()
            return curr_state, self.infeasible_reward, None, None

        obj_to_pick = obj
        if check_feasibility:
            motion, status = self.check_one_arm_pick_feasible(obj_to_pick, pick_base_pose, g_config, region)
        else:
            motion = parent_motion
            status = 'HasSolution'

        if status == 'HasSolution':
            set_robot_config(pick_base_pose, self.robot)
            one_arm_pick_obj(obj_to_pick, self.robot, g_config)
            pick_config = self.robot.GetDOFValues()
            curr_state = self.get_state()
            reward = 0
            return curr_state, reward, pick_config, motion
        else:
            curr_state = self.get_state()
            return curr_state, self.infeasible_reward, None, None

    def apply_one_arm_place_action(self, action, obj, region, check_feasibility, parent_motion):
        base_pose = action['base_pose']
        g_config = action['g_config']
        full_place_config = np.hstack([g_config, base_pose.squeeze()])
        curr_state = self.get_state()

        if check_feasibility:
            motion, status = self.get_arm_base_motion_plan(full_place_config, region.name)
        else:
            motion = parent_motion
            status = 'HasSolution'

        if status == 'HasSolution':
            manip = self.robot.GetManipulator('rightarm_torso')
            self.robot.SetActiveDOFs(manip.GetArmIndices(), DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
            set_active_dof_conf(full_place_config, self.robot)
            one_arm_place_obj(obj, self.robot)
            reward = 1
            return curr_state, reward, motion
        else:
            return curr_state, self.infeasible_reward, None

    def place_object(self, place_base_pose, object=None):
        if object is None:
            obj_to_place = self.robot.GetGrabbed()[0]
        else:
            obj_to_place = object

        robot = self.robot
        leftarm_manip = robot.GetManipulator('leftarm')
        rightarm_manip = robot.GetManipulator('rightarm')

        set_robot_config(place_base_pose, robot)
        place_obj(obj_to_place, robot, leftarm_manip, rightarm_manip)

        if obj_to_place.GetName().find('packing_box') != -1:
            self.update_box_region(obj_to_place)
