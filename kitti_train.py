import argparse
import os
import pickle
import time

from kitti_model import Model
from kitti_utils import DataLoader
from kitti_functions import *


def print_training_info(args, min_avg_loss):

    print('-----------------------------------------------------------')
    print('-------------- IRL LSTM TRAINING INFORMATION --------------')
    print('.network structure: LSTM')
    print('   rnn size (%d)' % (args.rnn_size))
    print('   num layers (%d)' % (args.num_layers))
    print('   num mixture (%d)' % (args.num_mixture))
    print('.network structure: Conv')
    print('   map size : %d x %d' % (args.map_size, args.map_size))
    print('   to be specified in the nearest future')
    print('.dataset')
    print('   ' + args.dataset_path)
    print('.training setting')
    print('   batch size (%d)' % args.batch_size)
    print('   seq_length (%d)' % args.seq_length)
    print('   learning rate (%.5f)' % args.learning_rate)
    print('   reg. lambda (%.5f)' % args.lambda_param)
    print('   grad_clip (%.2f)' % args.grad_clip)
    print('   data scale (%.1f)' % args.data_scale)
    print('   keep prob (%.2f)' % args.keep_prob)
    if (args.data_augmentation == 0):
        print('   data augmentation : none')
    elif (args.data_augmentation == 1):
        print('   data augmentation : random flip x/y')
    elif (args.data_augmentation == 2):
        print('   data augmentation : random rotation (360 degrees)')
    if (min_avg_loss < 0):
        print('.minimum average loss for validation : %.4f' % min_avg_loss)
    print('------------------------------------------------------------')

def main():
    parser = argparse.ArgumentParser()

    # # network structure : LSTM
    parser.add_argument('--rnn_size', type=int, default=256,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    parser.add_argument('--input_dim', type=int, default=2,
                        help='dimension of input vector')
    parser.add_argument('--num_mixture', type=int, default=10,
                        help='number of gaussian mixtures')

    # # training setting
    parser.add_argument('--dataset_path', type=str, default='./kitti-all-typeA-seq_len_40-avgfs_3_map.cpkl',
                        help='dataset path')
    parser.add_argument('--data_augmentation', type=int, default=1,
                        help='0: none, 1: random flip, 2: random rotation')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=30, # ------------------
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=300,
                        help='number of epochs')
    parser.add_argument('--model_dir', type=str, default='save',
                        help='directory to save model to')
    parser.add_argument('--grad_clip', type=float, default=10.0,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--lambda_param', type=float, default=0.0005,
                        help='regularization weight')
    parser.add_argument('--data_scale', type=float, default=10,
                        help='factor to scale raw data down by')
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='dropout keep probability')
    parser.add_argument('--patient_thr', type=float, default=100,
                        help='threshold for early stopping')

    # # social info
    parser.add_argument('--social_range', type=int, default=16,
                        help='maximum distance for considering social neighbor')
    parser.add_argument('--grid_size', type=int, default=4,
                        help='grid size')


    # # map info
    parser.add_argument('--map_size', type=int, default=48,
                        help='width of map image')

    args = parser.parse_args()
    train(args)

def train(args):

    # print training information
    print_training_info(args, 0.0)

    # training data preparation (utils.py)
    data_loader = DataLoader(args)

    if args.model_dir != '' and not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    with open(os.path.join(args.model_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # network model definition (model.py)
    model = Model(args)

    with tf.Session() as sess:

        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())

        best_epoch = 0
        patient = 0
        min_avg_loss = 100000
        for e in range(args.num_epochs):

            # reset batch pointer
            data_loader.reset_batch_pointer()
            # ++++++++++++++++++++++++++++++++++++++++++
            # VERY INPORTANT AT ITERATION ARCHITECTURE
            state = model.state_in.eval()
            # ++++++++++++++++++++++++++++++++++++++++++

            # # Train one epoch -------------------------------------------------------
            start = time.time()
            for b in range(data_loader.num_batches):
                x, y, sg, m, d = data_loader.next_batch()

                feed = {model.input_data: x, model.target_data: y, model.map_data: m, model.state_in: state}

                # train rnn
                train_loss, _ = sess.run([model.cost_pose, model.train_op_pose], feed)

                # train reward
                train_loss, _ = sess.run([model.cost_reward, model.train_op_reward], feed)


            end = time.time()

            # # Validation ------------------------------------------------------------
            valid_loss_list = []
            rwd_gt = []
            rwd_policy = []
            data_loader.reset_batch_pointer()
            # ++++++++++++++++++++++++++++++++++++++++++
            # VERY INPORTANT AT ITERATION ARCHITECTURE
            state = model.state_in.eval()
            # ++++++++++++++++++++++++++++++++++++++++++
            while (True):

                x, y, sg, m, d = data_loader.next_batch_valid()
                if (len(x) == 0):
                    break
                else:
                    feed = {model.input_data: x, model.target_data: y, model.map_data: m, model.state_in: state}
                    rwd_g, rwd_p, valid_loss = sess.run([model.reward_gt_avg, model.reward_est_avg, model.cost_valid], feed)
                    rwd_gt.append(rwd_g)
                    rwd_policy.append(rwd_p)
                    valid_loss_list.append(valid_loss)

            # show current performance
            print('[epoch %03d, time %.2f, p-lvl %02d, %.2f hours left] cost: %.4f'
                  % (e, (end-start), patient, ((end-start)*(args.num_epochs-e-1)/3600.0), np.mean(valid_loss_list)))

            # save every breakthrough
            if (min_avg_loss > np.mean(valid_loss_list)):
                best_epoch = e
                patient = 0
                min_avg_loss = np.mean(valid_loss_list)
                checkpoint_path = os.path.join(args.model_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=e)
                print(">> model saved to {}".format(checkpoint_path))
            else:
                patient += 1

            # early stop
            if (patient > args.patient_thr):
                print_training_info(args, min_avg_loss)
                print('>> early stop triggered ...')
                print('>> Best performance occured at %d epoch, corresponding avg. MAE %.4f' % (best_epoch, min_avg_loss))
                print('>> goodbye ...')
                break

if __name__ == '__main__':
    main()
