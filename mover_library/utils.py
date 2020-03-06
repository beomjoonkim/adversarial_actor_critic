from manipulation.primitives.transforms import *
from manipulation.bodies.bodies import *
from manipulation.constants import PARALLEL_LEFT_ARM, REST_LEFT_ARM, HOLDING_LEFT_ARM, FOLDED_LEFT_ARM, \
    FAR_HOLDING_LEFT_ARM, LOWER_TOP_HOLDING_LEFT_ARM, REGION_Z_OFFSET

from manipulation.primitives.utils import mirror_arm_config
from openravepy import *
import numpy as np
import math
import time

def convert_collision_vec_to_one_hot(c_data):
    n_konf = c_data.shape[1]
    onehot_cdata = []
    for cvec in c_data:
        one_hot_cvec = np.zeros((n_konf, 2))
        for boolean_collision, onehot_collision in zip(cvec, one_hot_cvec):
            onehot_collision[boolean_collision] = 1
        assert (np.all(np.sum(one_hot_cvec, axis=1) == 1))
        onehot_cdata.append(one_hot_cvec)

    onehot_cdata = np.array(onehot_cdata)
    return onehot_cdata


def compute_angle_to_be_set(target_xy, src_xy):
    target_dirn = target_xy - src_xy
    target_dirn = target_dirn / np.linalg.norm(target_dirn)
    if target_dirn[1] < 0:
        # rotation from x-axis, because that is the default rotation
        angle_to_be_set = -math.acos(np.dot(target_dirn, np.array(([1, 0]))))
    else:
        angle_to_be_set = math.acos(np.dot(target_dirn, np.array(([1, 0]))))
    return angle_to_be_set


def convert_rel_to_abs_base_pose(rel_xytheta, src_xy):
    if len(rel_xytheta.shape) == 1: rel_xytheta = rel_xytheta[None, :]
    assert (len(src_xy.shape) == 1)
    ndata = rel_xytheta.shape[0]
    abs_base_pose = np.zeros((ndata, 3))
    abs_base_pose[:, 0:2] = rel_xytheta[:, 0:2] + src_xy[0:2]
    for i in range(ndata):
        th_to_be_set = compute_angle_to_be_set(src_xy[0:2], abs_base_pose[i, 0:2])
        abs_base_pose[i, -1] = th_to_be_set + rel_xytheta[i, -1]
    return abs_base_pose


def set_body_transparency(body, transparency):
    for link in body.GetLinks():
        for geom in link.GetGeometries():
            geom.SetTransparency(transparency)


def set_obj_xytheta(xytheta, obj):
    if isinstance(xytheta, list):
        xytheta = np.array(xytheta)
    xytheta = xytheta.squeeze()
    set_quat(obj, quat_from_angle_vector(xytheta[-1], np.array([0, 0, 1])))
    set_xy(obj, xytheta[0], xytheta[1])


def set_active_dof_conf(conf, robot):
    robot.SetActiveDOFValues(conf.squeeze())


def draw_robot_at_conf(conf, transparency, name, robot, env, color=None):
    held_obj = robot.GetGrabbed()

    newrobot = RaveCreateRobot(env, '')
    newrobot.Clone(robot, 0)
    newrobot.SetName(name)
    env.Add(newrobot, True)
    set_active_dof_conf(conf, newrobot)
    newrobot.Enable(False)
    if color is not None:
        set_color(newrobot, color)

    if len(held_obj) > 0:
        held_obj = robot.GetGrabbed()[0]
        held_obj_trans = held_obj.GetTransform()
        release_obj(newrobot, newrobot.GetGrabbed()[0])
        new_obj = RaveCreateKinBody(env, '')
        new_obj.Clone(held_obj, 0)
        new_obj.SetName(name + '_obj')
        env.Add(new_obj, True)
        new_obj.SetTransform(held_obj_trans)
        for link in new_obj.GetLinks():
            for geom in link.GetGeometries():
                geom.SetTransparency(transparency)

    for link in newrobot.GetLinks():
        for geom in link.GetGeometries():
            geom.SetTransparency(transparency)



def visualize_path(robot, path):
    env = robot.GetEnv()
    if len(path) > 1000:
        path_reduced = path[0:len(path) - 1:int(len(path) * 0.1)]
    else:
        path_reduced = path
    for idx, conf in enumerate(path_reduced):
        is_goal_config = idx == len(path_reduced) - 1
        if is_goal_config:
            draw_robot_at_conf(conf, 0, 'path' + str(idx), robot, env)
        else:
            draw_robot_at_conf(conf, 0.7, 'path' + str(idx), robot, env)
    raw_input("Continue?")
    remove_drawn_configs('path', env)


def get_best_weight_file(train_results_dir):
    try:
        assert (os.path.isfile(train_results_dir + '/best_weight.txt'))
    except:
        print "Run choose_place_weights for " + train_results_dir
        sys.exit(-1)
    with open(train_results_dir + '/best_weight.txt') as fin:
        temp = fin.read().splitlines()
        weight_score = float(temp[0])  # first line is the weight file name
        weight_f_name = temp[1]
    return weight_f_name

def open_gripper(robot):
    robot.SetDOFValues(np.array([0.54800022]), robot.GetActiveManipulator().GetGripperIndices())


def get_ordered_weight_files(train_results_dir):
    try:
        assert (os.path.isfile(train_results_dir + '/wfile_scores.txt'))
    except:
        print "Run choose_place_weights for " + train_results_dir
        sys.exit(-1)

    wfiles = []
    with open(train_results_dir + '/wfile_scores.txt') as fin:
        ordered_weight_file_names = fin.read().splitlines()
        for l in ordered_weight_file_names[::-1]:
            temp = l.split(',')
            wfiles.append(temp[0])

    return wfiles


def determine_best_weight_path_for_given_n_data(parent_dir, n_data):
    place_eval_dir = parent_dir + '/n_data_' + str(n_data) + '/'
    test_mse_list = []
    weight_path_list = []
    for trial_dir in os.listdir(place_eval_dir):
        if trial_dir.find('n_trial') == -1: continue
        trial_train_results_dir = place_eval_dir + trial_dir + '/train_results/'
        try:
            assert (os.path.isfile(trial_train_results_dir + '/best_weight.txt'))
        except:
            continue
            print "Warning: Run train evaluator for" + trial_train_results_dir
            continue
        with open(trial_train_results_dir + '/best_weight.txt') as fin:
            temp = fin.read().splitlines()
            weight_name = temp[0]  # first line is the weight file name
            test_mse = float(temp[1])
        test_mse_list.append(test_mse)
        weight_path_list.append(trial_train_results_dir + weight_name)
    if len(weight_path_list) == 0:
        print "No trained evaluator found"
        sys.exit(-1)
    return weight_path_list[np.argmin(test_mse_list)]


def check_collision_except(exception_body, env):
    assert exception_body != env.GetRobots()[0], 'Collision exception cannot be the robot'

    #exception_body.Enable(False)  # todo what happens to the attached body when I enable and disable the held object?
    #col = env.CheckCollision(env.GetRobots()[0])
    #exception_body.Enable(True)
    # todo optimize this later
    return np.any([env.CheckCollision(env.GetRobots()[0], body) for body in env.GetBodies() if body != exception_body])
    #return col



def set_robot_config(base_pose, robot):
    base_pose = np.array(base_pose)
    base_pose = clean_pose_data(base_pose.astype('float'))

    robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
    base_pose = np.array(base_pose).squeeze()
    """
    while base_pose[-1] < 0:
      try:
        factor = -int(base_pose[-1] /(2*np.pi))
      except:
        import pdb;pdb.set_trace()
      if factor == 0: factor = 1
      base_pose[-1] += factor*2*np.pi
    while base_pose[-1] > 2*np.pi:
      factor = int(base_pose[-1] /(2*np.pi))
      base_pose[-1] -= factor*2*np.pi
    
    if base_pose[-1] <
    if base_pose[-1] > 1.01:
      base_pose[-1] = 1.01
    elif base_pose[-1] < 0.99:
      base_pose[-1] = 0.99
    """
    # print base_pose
    robot.SetActiveDOFValues(base_pose)


def trans_from_xytheta(obj, xytheta):
    rot = rot_from_quat(quat_from_z_rot(xytheta[-1]))
    z = get_point(obj)[-1]
    trans = np.eye(4)
    trans[:3, :3] = rot
    trans[:3, -1] = [xytheta[0], xytheta[1], z]
    return trans


def remove_drawn_configs(name, env):
    for body in env.GetBodies():
        if body.GetName().find(name) != -1:
            env.Remove(body)


def draw_robot_base_configs(configs, robot, env, name='bconf', transparency=0.7):
    for i in range(len(configs)):
        config = configs[i]
        draw_robot_at_conf(config, transparency, name + str(i), robot, env)


def draw_configs(configs, env, name='point', colors=None, transparency=0.1):
    # assert configs[0].shape==(6,), 'Config shape must be (6,)'
    if colors is None:
        for i in range(len(configs)):
            config = configs[i]
            new_body = box_body(env, 0.1, 0.05, 0.05, \
                                name=name + '%d' % i, \
                                color=(1, 0, 0), \
                                transparency=transparency)
            env.Add(new_body);
            set_point(new_body, np.append(config[0:2], 0.075))
            new_body.Enable(False)
            th = config[2]
            set_quat(new_body, quat_from_z_rot(th))
    else:
        for i in range(len(configs)):
            config = configs[i]
            if isinstance(colors, tuple):
                color = colors
            else:
                color = colors[i]
            new_body = box_body(env, 0.1, 0.05, 0.05, \
                                name=name + '%d' % i, \
                                color=color, \
                                transparency=transparency)
            """
            new_body = load_body(env,'mug.xml')
            set_name(new_body, name+'%d'%i)
            set_transparency(new_body, transparency)
            """
            env.Add(new_body);
            set_point(new_body, np.append(config[0:2], 0.075))
            new_body.Enable(False)
            th = config[2]
            set_quat(new_body, quat_from_z_rot(th))


def clean_pose_data(pose_data):
    # fixes angle to be between 0 to 2pi
    if len(pose_data.shape) == 1:
        pose_data = pose_data[None, :]

    data_idx_neg_angles = pose_data[:, -1] < 0
    data_idx_big_angles = pose_data[:, -1] > 2 * np.pi
    pose_data[data_idx_neg_angles, -1] += 2 * np.pi
    pose_data[data_idx_big_angles, -1] -= 2 * np.pi

    # assert( np.all(pose_data[:,-1]>=0) and np.all(pose_data[:,-1] <2*np.pi))
    return pose_data


def compute_occ_vec(key_configs, robot, env):
    occ_vec = []
    with robot:
        for config in key_configs:
            set_robot_config(config, robot)
            collision = env.CheckCollision(robot) * 1
            occ_vec.append(collision)
    return np.array(occ_vec)


def get_robot_xytheta(robot):
    with robot:
        robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
        robot_xytheta = robot.GetActiveDOFValues()
    robot_xytheta = robot_xytheta[None, :]
    clean_pose_data(robot_xytheta)
    return robot_xytheta


def get_body_xytheta(body):
    Tbefore = body.GetTransform()
    body_quat = get_quat(body)
    th1 = np.arccos(body_quat[0]) * 2
    th2 = np.arccos(-body_quat[0]) * 2
    th3 = -np.arccos(body_quat[0]) * 2
    quat_th1 = quat_from_angle_vector(th1, np.array([0, 0, 1]))
    quat_th2 = quat_from_angle_vector(th2, np.array([0, 0, 1]))
    quat_th3 = quat_from_angle_vector(th3, np.array([0, 0, 1]))
    if np.all(np.isclose(body_quat, quat_th1)):
        th = th1
    elif np.all(np.isclose(body_quat, quat_th2)):
        th = th2
    elif np.all(np.isclose(body_quat, quat_th3)):
        th = th3
    else:
        print "This should not happen. Check if object is not standing still"
        import pdb;
        pdb.set_trace()
    if th < 0: th += 2 * np.pi
    assert (th >= 0 and th < 2 * np.pi)

    # set the quaternion using the one found
    set_quat(body, quat_from_angle_vector(th, np.array([0, 0, 1])))
    Tafter = body.GetTransform()
    assert (np.all(np.isclose(Tbefore, Tafter)))
    body_xytheta = np.hstack([get_point(body)[0:2], th])
    body_xytheta = body_xytheta[None, :]
    clean_pose_data(body_xytheta)
    return body_xytheta


GRAB_SLEEP_TIME = 0.05


def grab_obj(robot, obj):
    robot.GetEnv().StopSimulation()
    robot.Grab(obj)
    robot.GetEnv().StartSimulation(0.01)


def release_obj(robot, obj):
    robot.GetEnv().StopSimulation()
    robot.Release(obj)
    robot.GetEnv().StartSimulation(0.01)


def one_arm_pick_obj(obj, robot, g_config):
    open_gripper(robot)
    set_config(robot, g_config, robot.GetManipulator('rightarm_torso').GetArmIndices())
    grab_obj(robot, obj)


def one_arm_place_obj(obj, robot):
    release_obj(robot, obj)
    set_config(robot, mirror_arm_config(FOLDED_LEFT_ARM), robot.GetManipulator('rightarm').GetArmIndices())


def pick_obj(obj, robot, g_configs, left_manip, right_manip):
    set_config(robot, g_configs[0], left_manip.GetArmIndices())
    set_config(robot, g_configs[1], right_manip.GetArmIndices())
    grab_obj(robot, obj)


def place_obj(obj, robot, leftarm_manip, rightarm_manip):
    release_obj(robot, obj)
    set_config(robot, FOLDED_LEFT_ARM, leftarm_manip.GetArmIndices())
    set_config(robot, mirror_arm_config(FOLDED_LEFT_ARM), rightarm_manip.GetArmIndices())


def simulate_path(robot, path, timestep=0.001):
    for p in path:
        set_robot_config(p, robot)
        time.sleep(timestep)
