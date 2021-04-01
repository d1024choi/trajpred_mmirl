import argparse
import os
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from kitti_model import Model
from kitti_utils import DataLoader
from kitti_functions import *
from kitti_functions import _max, _min

def preprocessing(traj, data_scale):

    '''
    :param traj: (seq_length+1, input_dim)
    :return:
    '''

    processed_data = np.copy(traj)
    prev_x = 0
    prev_y = 0

    for t in range(traj.shape[0]):
        processed_data[t, 0] = traj[t, 0] - prev_x
        processed_data[t, 1] = traj[t, 1] - prev_y

        prev_x = traj[t, 0]
        prev_y = traj[t, 1]

    processed_data[:, 0:2] /= data_scale

    return processed_data

def postprocessing(traj, data_scale):

    traj_ext = np.copy(traj)
    traj_ext *= data_scale

    for t in range(1, traj_ext.shape[0]):
        traj_ext[t, 0] += traj_ext[t - 1, 0]
        traj_ext[t, 1] += traj_ext[t - 1, 1]

    return traj_ext

def measure_accuracy_overall(true_traj, est_traj, pred_length):

    seq_length = true_traj.shape[0]
    obs_length = seq_length - pred_length

    x_diff, y_diff = 0, 0
    for i in range(obs_length, seq_length):
        x_diff += abs(true_traj[i, 0] - est_traj[i, 0])
        y_diff += abs(true_traj[i, 1] - est_traj[i, 1])

    return (x_diff/pred_length), (y_diff/pred_length)

def measure_accuracy_endpoint(true_traj, est_traj, pred_length):

    seq_length = true_traj.shape[0]
    obs_length = seq_length - pred_length
    error_traj = true_traj - est_traj

    return error_traj[obs_length:seq_length, 0:2]

def plot_trajectories(x, est_traj_post):

    plt.plot(x[:, 0], x[:, 1], 'bo-', label='true traj.')
    plt.plot(est_traj_post[x.shape[0] - pred_length - 1:, 0], est_traj_post[x.shape[0] - pred_length - 1:, 1], 'y+-.',
             label='est. traj.')
    plt.plot(est_traj_post[0:x.shape[0] - pred_length, 0], est_traj_post[0:x.shape[0] - pred_length, 1], 'y+-',
             label='input for est.')
    plt.legend()
    min_x = np.min(x[:, 0])
    min_y = np.min(x[:, 1])

    max_x = np.max(x[:, 0])
    max_y = np.max(x[:, 1])

    diff_x = abs(np.min(x[:, 0]) - np.max(x[:, 0]))
    diff_y = abs(np.min(x[:, 1]) - np.max(x[:, 1]))
    width = abs(diff_x - diff_y) / 2

    mr = 0.5
    if (diff_x < 2 and diff_y < 2):
        mr = _max(2-diff_x, 2-diff_y) + 0.5

    if (diff_x > diff_y):
        plt.axis([min_x-mr, max_x+mr, min_y-width-mr, max_y+width+mr])
    else:
        plt.axis([min_x-width-mr, max_x+width+mr, min_y-mr, max_y+mr])
    plt.show()

def plot_trajectories_on_map(x_gt, x_est, pred_length, map, x_max, y_max, scale):

    seq_length = x_gt.shape[0]
    obs_length = seq_length - pred_length

    x_gt_max = np.max(x_gt[:, 0])
    x_gt_min = np.min(x_gt[:, 0])

    y_gt_max = np.max(x_gt[:, 1])
    y_gt_min = np.min(x_gt[:, 1])

    x_axis_width = abs(x_gt_max - x_gt_min)
    y_axis_width = abs(y_gt_max - y_gt_min)
    map_size = scale*(3.0*_max(x_axis_width, y_axis_width) + 3.0)
    map_size = _max(_min(map_size, 400), 100)

    x = x_gt[obs_length - 1, 0]
    y = x_gt[obs_length - 1, 1]
    map_roi = np.copy(map_roi_extract(map, x, y, x_max, y_max, scale, int(map_size/2)))
    map_roi_copy = np.copy(map_roi)
    map_row_cnt = map_roi.shape[0] / 2
    map_col_cnt = map_roi.shape[1] / 2

    pose_start_x = x
    pose_start_y = y

    for i in range(seq_length):

        pose_x = int(scale * (x_gt[i, 0] - pose_start_x) + map_row_cnt)
        pose_y = int(scale * (x_gt[i, 1] - pose_start_y) + map_col_cnt)

        pose_x = _min(_max(pose_x, 0), map_roi.shape[0] - 1)
        pose_y = _min(_max(pose_y, 0), map_roi.shape[1] - 1)

        map_roi[pose_x, pose_y, 0] = 0
        map_roi[pose_x, pose_y, 1] = 255
        map_roi[pose_x, pose_y, 2] = 255

        if (i > obs_length-1):
            pose_x = int(scale * (x_est[i, 0] - pose_start_x) + map_row_cnt)
            pose_y = int(scale * (x_est[i, 1] - pose_start_y) + map_col_cnt)

            pose_x = _min(_max(pose_x, 0), map_roi.shape[0] - 1)
            pose_y = _min(_max(pose_y, 0), map_roi.shape[1] - 1)

            map_roi[pose_x, pose_y, 0] = 20
            map_roi[pose_x, pose_y, 1] = 255
            map_roi[pose_x, pose_y, 2] = 57

    #cv2.imshow('test', map_roi)
    #cv2.waitKey(0)

    return map_roi_copy

# ------------------------------------------------------
# load saved network and parameters

path = './save'

# load parameter setting
with open(os.path.join(path, 'config.pkl'), 'rb') as f:
    saved_args = pickle.load(f)

# define model structure
model = Model(saved_args, True)

# load trained weights
ckpt = tf.train.get_checkpoint_state(path)
sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, ckpt.model_checkpoint_path)
print(">> loaded model: ", ckpt.model_checkpoint_path)


# ------------------------------------------------------
# variable definition for validation

# validation setting : observe (seq_length-pred_length-1) and predict (pred_length)
isPlot = True
saved_args.seq_length = 31
pred_length = 20

overall_error_x = 0
overall_error_y = 0
overall_counter = 0
overall_error_vec = np.zeros(shape=(pred_length, 2))

# load validation data
data_loader = DataLoader(saved_args)
data_loader.pointer = 0
data_loader.frame_pointer = 0
x, grid, map, x_max, y_max, scale, dataset_index, NotEndOfData = data_loader.next_sequence_valid()

# for saving validation results
valid_map = []
valid_gt = []
valid_est = []
valid_map_info = []
while (NotEndOfData):

    # subtract previous from current and scaling
    x_prepro = preprocessing(x, saved_args.data_scale)

    # run prediction
    est_traj, _ = model.sample(sess, x_prepro, grid, pred_length, map, x_max, y_max, scale) # modification

    # post processing
    est_traj_post = postprocessing(est_traj, saved_args.data_scale) # modification

    # calculate error
    # abs_err_x, abs_err_y = measure_accuracy_overall(x[:, 0:2], est_traj_post, pred_length)
    err_vector = measure_accuracy_endpoint(x[:, 0:2], est_traj_post, pred_length)

    overall_error_x += np.mean(abs(err_vector[:, 0]))
    overall_error_y += np.mean(abs(err_vector[:, 1]))
    overall_error_vec += np.absolute(err_vector)
    overall_counter += 1

    #if (isPlot):
        #plot_trajectories(x, est_traj_post)
    map_roi = plot_trajectories_on_map(x, est_traj_post, pred_length, map, x_max, y_max, scale)

    valid_map.append(map_roi)
    valid_map_info.append([x_max, y_max, scale, dataset_index])
    valid_gt.append(x)
    valid_est.append(est_traj_post)

    x, grid, map, x_max, y_max, scale, dataset_index, NotEndOfData = data_loader.next_sequence_valid()

print('>> Mean Absolute Error x-axis (meter) : %.2f' % (overall_error_x/overall_counter))
print('>> Mean Absolute Error y-axis (meter) : %.2f' % (overall_error_y/overall_counter))

for i in range(overall_error_vec.shape[0]):
    print('--------------------------------------------------------------------------------')
    print(' > end point MAE (x-axis) at delta %.2f sec : %.2f' % ((float(i+1) * 0.1), (overall_error_vec[i, 0] / overall_counter)))
    print(' > end point MAE (y-axis) at delta %.2f sec : %.2f' % ((float(i+1) * 0.1), (overall_error_vec[i, 1] / overall_counter)))

save_dir = './kitti-select_10_20-seq_len_40-avgfs_3_map_v11.cpkl'
f = open(save_dir, "wb")
pickle.dump((valid_map, valid_map_info, valid_gt, valid_est), f, protocol=2)
f.close()