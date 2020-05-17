import sys
sys.path.append('../mover_library/')

from mcts_tree_node import StateSaverTreeNode
from manipulation.primitives.savers import DynamicEnvironmentStateSaver

from mover_library.samplers import *
from mover_library.utils import *
from mcts_utils import make_action_hashable, is_action_hashable
from mover_library.utils import draw_robot_at_conf, remove_drawn_configs
import socket

if socket.gethostname() == 'dell-XPS-15-9560':
    from mcts_graphics import write_dot_file

import time
import numpy as np


DEBUG = True


class DynamicEnvironmentStateSaverWithCurrObj(DynamicEnvironmentStateSaver):
    def __init__(self, env, placements, curr_obj, which_operator):
        DynamicEnvironmentStateSaver.__init__(self, env)
        self.curr_obj = curr_obj
        self.which_operator = which_operator
        self.placements = placements


class MCTSTree:
    def __init__(self, root, exploration_parameters):
        self.root = root
        self.nodes = [root]
        self.exploration_parameters = exploration_parameters

    def has_state(self, state):
        return len([n for n in self.nodes if np.all(n.state == state)]) > 0

    def add_node(self, node, action, parent):
        node.parent = parent
        if is_action_hashable(action):
            parent.children[action] = node
        else:
            parent.children[make_action_hashable(action)] = node
        self.nodes.append(node)

    def is_node_just_added(self, node):
        if node == self.root:
            return False

        for action, resulting_child in zip(node.parent.children.keys(), node.parent.children.values()):
            if resulting_child == node:
                return not (action in node.parent.A)  # action that got to the node is not in parent's actions

    def get_leaf_nodes(self):
        return [n for n in self.nodes if len(n.children.keys()) == 0]

    def get_best_trajectory_sum_rewards(self):
        return max([n.sum_ancestor_action_rewards for n in self.get_leaf_nodes()])


class MCTS:
    def __init__(self, widening_parameter, exploration_parameters,
                 sampling_strategy, environment, domain_name, task_plan):

        self.progressive_widening_parameter = widening_parameter
        self.exploration_parameters = exploration_parameters

        self.time_limit = np.inf
        self.discount_rate = 0.9

        self.environment = environment

        self.task_plan = task_plan
        self.n_objs_to_manipulate = np.sum([len(p['objects']) for p in task_plan])
        self.task_plan_idx = 0
        self.obj_plan_idx = 0
        self.s0_node = self.create_node(None, depth=0, reward=0, is_init_node=True)

        self.tree = MCTSTree(self.s0_node, self.exploration_parameters)
        self.found_solution = False
        self.goal_reward = 0

        if domain_name == 'convbelt':
            self.depth_limit = 10
            self.is_satisficing_problem = True
        elif domain_name == 'namo':
            self.depth_limit = np.inf
            self.is_satisficing_problem = False

        self.sampling_strategy = sampling_strategy

    def update_task_plan_indices(self, reward, operator_used):
        if self.environment.is_solving_fetching:
            made_progress = reward > 0 and operator_used.find('place') != -1
        else:
            made_progress = reward > 0
        if made_progress:
            self.obj_plan_idx += 1
            if self.obj_plan_idx == len(self.task_plan[self.task_plan_idx]['objects']):
                self.obj_plan_idx = 0
                self.task_plan_idx += 1

    def reset_task_plan_indices(self):
        self.obj_plan_idx = 0
        self.task_plan_idx = 0

    def create_node(self, parent_action, depth, reward, is_init_node):
        if self.is_goal_reached():
            curr_obj = None
            operator = None
            curr_region = None
        else:
            curr_obj = self.task_plan[self.task_plan_idx]['objects'][self.obj_plan_idx]
            operator = self.environment.which_operator(curr_obj)  # this is after the execution of the operation

            if operator.find('pick') != -1:
                curr_region = self.environment.get_region_containing(curr_obj)
            else:
                curr_region = self.task_plan[self.task_plan_idx]['region']

        state_saver = DynamicEnvironmentStateSaverWithCurrObj(self.environment.env, [], curr_obj,
                                                              operator)
        node = StateSaverTreeNode(curr_obj, curr_region, operator,
                                  self.exploration_parameters, depth, state_saver, is_init_node)
        node.parent_action_reward = reward
        node.parent_action = parent_action
        return node

    @staticmethod
    def get_best_child_node(node):
        if len(node.children) == 0:
            return None
        else:
            # returns the most visited chlid
            # another option is to return the child with best value
            best_child_action_idx = np.argmax(node.N.values())
            best_child_action = node.N.keys()[best_child_action_idx]
            return node.children[best_child_action]

    def retrace_best_plan(self):
        plan = []
        leaves = self.tree.get_leaf_nodes()
        goal_nodes = [leaf for leaf in leaves if leaf.is_goal_node]
        if len(goal_nodes) > 1:
            best_traj_reward = self.tree.get_best_trajectory_sum_rewards()
            curr_node = [leaf for leaf in goal_nodes if leaf.sum_ancestor_action_rewards == best_traj_reward][0]
        else:
            curr_node = goal_nodes[0]

        while curr_node.parent_motion is not None:
            action = curr_node.parent_action
            path = curr_node.parent_motion
            obj = curr_node.parent.obj
            obj_name = obj.GetName()
            operator = curr_node.parent.operator
            plan.insert(0, {'action': action, 'path': path, 'obj_name': obj_name, 'operator': operator})
            curr_node = curr_node.parent
        return plan

    def switch_init_node(self, node):
        self.environment.set_init_state(node.state_saver)
        self.environment.reset_to_init_state()
        self.s0_node.is_init_node = False
        self.s0_node = node
        self.s0_node.is_init_node = True

    def search(self, n_iter=100, n_optimal_iter=0):
        # n_optimal_iter: additional number of iterations you are allowed to run after finding a solution
        depth = 0
        time_to_search = 0
        search_time_to_reward = []
        optimal_iter = 0
        for iter in range(n_iter):
            # todo make the switching work
            """
            if self.s0_node.Nvisited >= 10*(self.s0_node.Nvisited+1):
                best_child = self.get_best_child_node(self.s0_node)
                is_best_child_infeasible = best_child.parent_motion is None
                if is_best_child_infeasible and self.s0_node.parent is not None:
                    print "Switching the initial node failed, back-tracking to parent"
                    self.switch_init_node(self.s0_node.parent)
                    depth -= 1
                else:
                    print "Switching the initial node to the best child"
                    self.switch_init_node(best_child)
                    depth += 1
            """

            print '*****SIMULATION ITERATION %d' % iter
            self.environment.reset_to_init_state(self.s0_node)
            self.reset_task_plan_indices()
            # todo there seem to be something wrong with the Q-values

            stime = time.time()
            self.simulate(self.s0_node, depth)
            time_to_search += time.time() - stime
            if socket.gethostname() == 'dell-XPS-15-9560':
                if self.environment.is_solving_ramo:
                    write_dot_file(self.tree, iter, 'solving_ramo')
                elif self.environment.is_solving_packing:
                    write_dot_file(self.tree, iter, 'solving_packing')
                else:
                    write_dot_file(self.tree, iter, '')

            best_traj_rwd = self.tree.get_best_trajectory_sum_rewards()
            search_time_to_reward.append([time_to_search, best_traj_rwd, self.found_solution])

            if self.found_solution:
                optimal_iter += 1
            if self.found_solution and self.optimal_score_achieved(best_traj_rwd):
                print "Optimal score found"
                plan = self.retrace_best_plan()
                break
            elif self.found_solution and optimal_iter > n_optimal_iter:
                plan = self.retrace_best_plan()
                break
            elif self.found_solution and not self.optimal_score_achieved(best_traj_rwd):
                plan = self.retrace_best_plan()
            elif not self.found_solution:
                plan = None

        self.environment.reset_to_init_state(self.s0_node)
        return search_time_to_reward, plan, self.optimal_score_achieved(best_traj_rwd)

    def optimal_score_achieved(self, best_traj_rwd):
        # return best_traj_rwd == self.environment.optimal_score
        # in the case of namo, this is the length of the object
        # in the case of original problem, this is the number of objects to be packed
        if self.environment.is_solving_fetching:
            return best_traj_rwd == 2
        else:
            return best_traj_rwd == self.n_objs_to_manipulate

    def choose_action(self, curr_node):
        if DEBUG:
            print 'N(A), progressive parameter = %d,%f' % (len(curr_node.A),
                                                       self.progressive_widening_parameter * curr_node.Nvisited)

        n_actions = len(curr_node.A)
        is_time_to_sample = n_actions <= self.progressive_widening_parameter * curr_node.Nvisited
        if len(curr_node.Q.values()) > 0:
            best_Q = np.max(curr_node.Q.values())
            is_time_to_sample = is_time_to_sample or (best_Q == self.environment.infeasible_reward)

        if DEBUG:
            print 'Time to sample new action? ' + str(is_time_to_sample)

        if is_time_to_sample:
            action = self.sample_action(curr_node)
        else:
            action = curr_node.get_best_action()

        return action

    @staticmethod
    def update_node_statistics(curr_node, action, sum_rewards):
        curr_node.Nvisited += 1
        print action
        if is_action_hashable(action):
            hashable_action = action
        else:
            hashable_action = make_action_hashable(action)
        is_action_new = not (hashable_action in curr_node.A)
        if is_action_new:
            curr_node.A.append(hashable_action)
            curr_node.N[hashable_action] = 1
            curr_node.Q[hashable_action] = sum_rewards
            curr_node.sum_rewards_history[hashable_action] = [sum_rewards]
        else:
            curr_node.N[hashable_action] += 1
            curr_node.sum_rewards_history[hashable_action].append(sum_rewards)
            curr_node.Q[hashable_action] += (sum_rewards - curr_node.Q[hashable_action]) / \
                                            float(curr_node.N[hashable_action])

    def is_goal_reached(self):
        return self.task_plan_idx >= len(self.task_plan)

    def simulate(self, curr_node, depth):
        if self.is_goal_reached():
            # arrived at the goal state
            self.found_solution = True
            print "Solution found, returning the goal reward", self.goal_reward
            curr_node.is_goal_node = True
            return self.goal_reward

        if DEBUG:
            print "At depth ", depth
            print "Is it time to pick?", self.environment.is_pick_time()

        action = self.choose_action(curr_node)
        print action, curr_node.operator

        parent_motion = None
        if curr_node.is_action_tried(action):
            if DEBUG:
                print "Executing tree policy, taking action ", action
            next_node = curr_node.get_child_node(action)
            if next_node.parent_motion is None:
                check_feasibility = True # todo put this back
            else:
                parent_motion = next_node.parent_motion
                check_feasibility = False
        else:
            check_feasibility = True  # todo: store path

        if DEBUG:
            print 'Is pick time? ', self.environment.is_pick_time()
            print "Executing action ", action

        next_state, reward, parent_motion = self.apply_action(curr_node, action, check_feasibility, parent_motion)
        print 'Reward ', reward

        self.update_task_plan_indices(reward, action['operator_name']) # create the next node based on the updated task plan progress

        if not curr_node.is_action_tried(action):
            next_node = self.create_node(action, depth+1, reward, is_init_node=False)
            self.tree.add_node(next_node, action, curr_node)
            next_node.sum_ancestor_action_rewards = next_node.parent.sum_ancestor_action_rewards + reward

        if next_node.parent_motion is None and reward != self.environment.infeasible_reward:
            next_node.parent_motion = parent_motion

        is_infeasible_action = reward == self.environment.infeasible_reward
        if is_infeasible_action:
            # this (s,a) is a dead-end
            sum_rewards = reward
        else:
            sum_rewards = reward + self.discount_rate * self.simulate(next_node, depth + 1)

        self.update_node_statistics(curr_node, action, sum_rewards)
        return sum_rewards

    def apply_action(self, node, action, check_feasibility, parent_motion):
        if action is None:
            return None, self.environment.infeasible_reward, None
        which_operator = self.environment.which_operator(node.obj)
        if which_operator == 'two_arm_pick':
            next_state, reward, path = self.environment.apply_two_arm_pick_action(action, node.obj, node.region, check_feasibility, parent_motion)
        elif which_operator == 'two_arm_place':
            next_state, reward, path = self.environment.apply_two_arm_place_action(action, node.obj, node.region, check_feasibility, parent_motion)
        elif which_operator == 'one_arm_pick':
            next_state, reward, path = self.environment.apply_one_arm_pick_action(action, node.obj, node.region, check_feasibility, parent_motion)
        elif which_operator == 'one_arm_place':
            next_state, reward, path = self.environment.apply_one_arm_place_action(action, node.obj, node.region, check_feasibility, parent_motion)
        """
        if self.environment.is_pick_time():
            next_state, reward, pick_conf, path = self.environment.apply_pick_action(action, node.obj, node.region, check_feasibility, parent_motion)
        else:
            next_state, reward, path = self.environment.apply_place_action(action, node.obj, node.region, check_feasibility, parent_motion)
        """
        return next_state, reward, path

    def sample_action(self, node):
        action = self.sampling_strategy.sample_next_point(node)
        return action


