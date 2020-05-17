from planners.forward_search import forward_dfs_search
from NAMO_env import NAMO
from generators.Uniform import UniformPick, UniformPlace

import numpy as np
import sys
import os
import argparse
import socket
import pickle
import time
import openravepy
import manipulation
from mover_library.utils import get_body_xytheta, set_robot_config


def get_pick_base(action, obj):
    grasp_params = action[0, 0:3]
    rel_pick_base_pose = action[0, 3:]
    curr_obj_xy = get_body_xytheta(obj)[0, 0:2]
    abs_pick_base_pose = action[0, 3:]


def play_motion(motion, t_sleep):
    for c in motion:
        set_robot_config(c)
        time.sleep(t_sleep)



def set_color(obj, color):
    env = openravepy.RaveGetEnvironments()[0]
    if type(obj) == str or type(obj) == unicode:
        obj = env.GetKinBody(obj)

    manipulation.bodies.bodies.set_color(obj, color)

def main():
    problem = NAMO()

    states, plan = pickle.load(open('plan.pkl', 'r'))
    obj_idx = 0
    states = states[::-1]
    states = states[::2]
    plan = plan[::-1]
    plan = plan[1:]
    picks = plan[::2]
    places = plan[1::2]
    problem.env.SetViewer('qtcoin')

    pick_paths = []
    place_paths = []
    pick_paths, place_paths, last_path = pickle.load(open('paths.pkl', 'r'))

    for o in problem.objects:
        set_color(o, [0, 0.5, 0])

    set_color(problem.target_obj, [0.8, 0, 0])
    viewer =problem.env.GetViewer()
    cam_transform = np.array([[ 0.99962688, -0.01036221,  0.02527319, -0.18571675],
       [-0.01567289, -0.97537481,  0.21999624, -1.61661613],
       [ 0.02237119, -0.22031026, -0.97517328,  8.26092148],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])

    viewer.SetCamera(cam_transform)
    import pdb;pdb.set_trace()
    time.sleep(3)

    #for pick, place in zip(picks, places):
    for pick, place, pick_path, place_path in zip(picks, places, pick_paths, place_paths):
        problem.curr_obj_name = states[obj_idx]
        abs_pick_base_pose = pick[0, 3:]
        abs_place_base_pose = place
        """
        pick_path,  status = problem.get_motion_plan(abs_pick_base_pose)
        while status!='HasSolution':
            pick_path, status = problem.get_motion_plan(abs_pick_base_pose)
        pick_paths.append(pick_path)
        """

        play_motion(pick_path, 0.05)
        problem.apply_pick_action(pick, True)
        play_motion(place_path, 0.05)

        """
        place_path, status = problem.get_motion_plan(abs_place_base_pose)
        while status!='HasSolution':
            place_path, status = problem.get_motion_plan(abs_place_base_pose)
        place_paths.append(place_path)
        """
        problem.apply_place_action(place)
        obj_idx += 1
    #last_path, status = problem.get_motion_plan(problem.problem['original_path'][-1])
    play_motion(last_path, 0.05)

    # pickle.dump([pick_paths,place_paths,last_path], open('paths.pkl','wb'))
    import pdb;
    pdb.set_trace()


if __name__ == '__main__':
    main()
