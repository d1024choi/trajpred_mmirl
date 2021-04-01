import numpy as np
import math
import random
import tensorflow as tf
import cv2
import copy

def _max(a,b):
    if (a>b):
        return a
    else:
        return b

def _min(a,b):
    if (a<b):
        return a
    else:
        return b

def moving_average(traj, fsize=3):

    '''
    traj[i, 0] = dataset id
    traj[i, 1] = object id
    traj[i, 2~3] = target pose
    traj[i, 4~63] = neighbor pose

    '''

    seq_length = traj.shape[0]
    processed_traj = np.copy(traj)

    fsize_h = int(fsize/2)

    for i in range(seq_length):
        if (i > fsize_h-1 and i < seq_length-fsize_h):
            processed_traj[i, 2] = np.mean(traj[i-fsize_h:i+fsize_h+1, 2])
            processed_traj[i, 3] = np.mean(traj[i-fsize_h:i+fsize_h+1, 3])

    return processed_traj

def getSocialMatrix(socialVec, target_pose, neighbor_pose, socialRange, grid_size):

    '''
    :param socialVec: (num_grid, num_grid)
    :param target_pose: (seq_length, 2)
    :param neighbor_pose: (seq_length, 60)
    :param socialRange:
    :param grid_size:
    '''

    num_grid = int(socialRange / grid_size)

    delta_x = neighbor_pose[0, 0] - target_pose[0, 0] + (socialRange/2)
    delta_y = neighbor_pose[0, 1] - target_pose[0, 1] + (socialRange/2)

    grid_idx_x = int(delta_x / grid_size)
    grid_idx_y = (num_grid - 1) - int(delta_y / grid_size)

    # debug
    if (grid_idx_x < 0 or grid_idx_x > (num_grid-1) or grid_idx_y < 0 or grid_idx_y > (num_grid-1)):
        donothing = 0
    else:
        # socialVec[grid_idx_x, grid_idx_y] = 1
        # image-x-axis corresponds to array-column
        socialVec[grid_idx_y, grid_idx_x] = 1

    return socialVec

def random_flip(x, y, map):

    '''
    (confirmed) randomly flip data in the direction of x and y axis
    '''

    if (np.random.rand(1) < 0.5):
        x[:, 0] = -1.0 * x[:, 0]
        y[y[:, :, 0] > -1000, 0] = -1.0 * y[y[:, :, 0] > -1000, 0]
        for i in range(len(map)):
            map[i] = np.flipud(map[i])

    if (np.random.rand(1) < 0.5):
        x[:, 1] = -1.0 * x[:, 1]
        y[y[:, :, 1] > -1000, 1] = -1.0 * y[y[:, :, 1] > -1000, 1]
        for i in range(len(map)):
            map[i] = np.fliplr(map[i])

    return x, y, map

def rotate_around_point(xy, degree, origin=(0, 0)):

    radians = math.radians(degree)
    x, y = xy
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return qx, qy

def random_rotate(tpose, npose):

    '''
    (confirmed) randomly rotate trajectory
     - must be used non-processed trajectory

    tpose = seq_len x 2
    npose = seq_len x 30 x 2
    '''

    tpose_rot = np.copy(tpose)
    npose_rot = np.copy(npose)
    origin = (tpose[0, 0], tpose[0, 1])
    degree = random.randint(1, 359)

    if (np.random.rand(1) < 0.5):
        for i in range(0, tpose.shape[0]):

            # rotate target pose
            rx, ry = rotate_around_point((tpose[i, 0], tpose[i, 1]), degree, origin)
            tpose_rot[i, 0] = rx
            tpose_rot[i, 1] = ry

            # rotate neighbors
            for j in range(30):
                if (npose[i, j, 0] == -1000):
                    continue
                else:
                    rx, ry = rotate_around_point((npose[i, j, 0], npose[i, j, 1]), degree, origin)
                    npose_rot[i, j, 0] = rx
                    npose_rot[i, j, 1] = ry

    return tpose_rot, npose_rot

def map_roi_extract(map, x, y, x_max, y_max, scale, width):


    size_row = map.shape[0]
    size_col = map.shape[1]

    x = scale * (-1.0 * (x - x_max))
    y = scale * ((-1.0 * y) + y_max)

    x_center = int(x.astype('int32'))
    y_center = int(y.astype('int32'))

    # improved 180523
    if (x_center-width < 0 or x_center+width-1 > size_row-1 or y_center-width < 0 or y_center+width-1 > size_col-1):
        part_map = np.zeros(shape=(2*width, 2*width, 3))
    else:
        part_map = np.flipud(np.fliplr(map[x_center - width:x_center + width, y_center - width:y_center + width, :]))

    return part_map

def weight_variable(shape, stddev=0.01, name=None):

    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name=name, initializer=initial)

def bias_variable(shape, init=0.0, name=None):

    initial = tf.constant(init, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name=name, initializer=initial)

def conv_weight_variable(shape, name=None):

    if len(shape) < 4:
        stddev_xavier = math.sqrt(3.0 / (shape[0] + shape[1]))
    else:
        stddev_xavier = math.sqrt(3.0 / ((shape[0]*shape[1]*shape[2]) + (shape[0]*shape[1]*shape[3])))

    initial = tf.truncated_normal(shape, stddev=stddev_xavier)

    return tf.get_variable(initializer=initial, name=name)

def conv_bias_variable(shape, init, name=None):

    initial = tf.constant(init, shape=shape)
    return tf.get_variable(initializer=initial, name=name)

def initialize_conv_filter(shape, name=None):

    W = conv_weight_variable(shape=shape, name=name+'w')
    b = conv_bias_variable(shape=[shape[3]], init=0.0, name=name+'b')

    return W, b

def conv2d_strided_relu(x, W, b, strides, padding):
    conv = tf.nn.conv2d(x, W, strides=strides, padding=padding)

    return tf.nn.relu(tf.nn.bias_add(conv, b))

def max_pool(x, ksize, strides):
    return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding="VALID")


def shallow_convnet(input, w1, b1, w2, b2, w3, b3):

    conv1 = conv2d_strided_relu(input, w1, b1, strides=[1, 1, 1, 1], padding='VALID')
    pool1 = max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

    conv2 = conv2d_strided_relu(pool1, w2, b2, strides=[1, 1, 1, 1], padding='VALID')
    pool2 = max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

    conv3 = conv2d_strided_relu(pool2, w3, b3, strides=[1, 1, 1, 1], padding='VALID')
    pool3 = max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

    #conv3_avg_pool = tf.reduce_mean(conv3, axis=3)
    #conv3_avg_pool_flat = tf.reshape(conv3_avg_pool, [-1, conv3.get_shape().as_list()[1] * conv3.get_shape().as_list()[2]])
    #pool3_flat = tf.reshape(pool3, [-1, pool3.get_shape().as_list()[1] * pool3.get_shape().as_list()[2] * pool3.get_shape().as_list()[3]])
    output = tf.reshape(pool3, [-1, pool3.get_shape().as_list()[1] * pool3.get_shape().as_list()[2] * pool3.get_shape().as_list()[3]])

    #output = tf.nn.relu(tf.nn.xw_plus_b(pool3_flat, fw3, fb3))

    return output

def calculate_reward(fwr, fbr, fc_in, cur_in, next_in):
    '''
    :param conv: (1 x conv_flat_size)
    :param cur_in: (1 x self.input_dim)
    :param next_in: (1 x self.input_dim)
    :return:
    '''

    state_vec = tf.concat([fc_in, cur_in, next_in], axis=1)
    reward = tf.nn.sigmoid(tf.nn.xw_plus_b(state_vec, fwr, fbr))

    return reward

def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
    # eq # 24 and 25 of http://arxiv.org/abs/1308.0850
    epsilon = 1e-20

    norm1 = tf.subtract(x1, mu1)
    norm2 = tf.subtract(x2, mu2)
    # s1s2 = tf.multiply(s1, s2)
    s1s2 = tf.add(tf.multiply(s1, s2), epsilon)

    z = tf.square(tf.div(norm1, s1)) + tf.square(tf.div(norm2, s2)) - \
        2 * tf.div(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2)

    negRho = 1 - tf.square(rho)
    result = tf.exp(tf.div(-z, 2 * negRho))

    denom = 2 * np.pi * tf.multiply(s1s2, tf.sqrt(negRho))

    result = tf.div(result, denom)
    return result

def get_lossfunc(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, x1_data, x2_data):

    result0 = tf_2d_normal(
        x1_data,
        x2_data,
        z_mu1,
        z_mu2,
        z_sigma1,
        z_sigma2,
        z_corr)

    # implementing eq # 26 of http://arxiv.org/abs/1308.0850
    epsilon = 1e-20
    result1 = tf.multiply(result0, z_pi)
    result1 = tf.reduce_sum(result1, 1, keep_dims=True)
    # at the beginning, some errors are exactly zero.
    result = -tf.log(tf.maximum(result1, epsilon))

    return tf.reduce_sum(result)

def get_mixture_coef(output):
    # returns the tf slices containing mdn dist params
    # ie, eq 18 -> 23 of http://arxiv.org/abs/1308.0850
    z = output
    z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = tf.split(
        axis=1, num_or_size_splits=6, value=z[:, 0:])

    # process output z's into MDN paramters

    # softmax all the pi's:
    max_pi = tf.reduce_max(z_pi, 1, keep_dims=True)
    z_pi = tf.subtract(z_pi, max_pi)
    z_pi = tf.exp(z_pi)
    normalize_pi = tf.reciprocal(
        tf.reduce_sum(z_pi, 1, keep_dims=True))
    z_pi = tf.multiply(normalize_pi, z_pi)

    # exponentiate the sigmas and also make corr between -1 and 1.
    z_sigma1 = tf.exp(z_sigma1)
    z_sigma2 = tf.exp(z_sigma2)
    z_corr = tf.tanh(z_corr)

    return [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr]

def sample_gaussian_2d(mu1, mu2, s1, s2, rho):
    mean = [mu1, mu2]
    cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]