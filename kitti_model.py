from kitti_functions import *

class Model():

    def __init__(self, args, infer=False):


        # # ------------------------------------------------------------------
        # # Parameter setting
        self.args = args
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        # in pose & out pose info
        self.input_dim = args.input_dim
        self.num_mixture = args.num_mixture
        NOUT = self.num_mixture * 6  # prob + 2*(mu + sig) + corr

        # social grid info
        self.num_grid = int(args.social_range / args.grid_size)
        self.grid_dim = self.num_grid * self.num_grid

        # semantic map info
        self.map_size = args.map_size

        # convnet info
        self.conv_flat_size = 144
        #self.fc_size_in = self.conv_flat_size + 2*args.rnn_size + 2*self.input_dim
        self.fc_size_in = self.conv_flat_size + 2 * self.input_dim
        self.fc_size_out = 1

        # dim. of input to embedding network
        self.feature_dim = self.input_dim

        # # ------------------------------------------------------------------
        # # Define network structure ------------------------------------
        if args.model == 'rnn':
            cell_fn = tf.contrib.rnn.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = tf.contrib.rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = tf.contrib.rnn.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        def get_cell():
            return cell_fn(args.rnn_size, state_is_tuple=False)

        # TODO : TEST MULTI-LAYERED RNN
        # cell = tf.contrib.rnn.MultiRNNCell([get_cell() for _ in range(args.num_layers)])
        cell = get_cell()
        if (infer == False and args.keep_prob < 1):  # training mode
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=args.keep_prob)
        self.cell = cell

        # Cell states : batch_size x (1, cell.state_size)
        zero_state = tf.split(tf.zeros([args.batch_size, cell.state_size]), axis=0, num_or_size_splits=args.batch_size)
        self.state_in = tf.identity(zero_state, name='state_in')
        self.state_out = tf.split(tf.zeros([args.batch_size, cell.state_size]), axis=0, num_or_size_splits=args.batch_size)

        # Output states : batch_size x (1, args.rnn_size)
        self.output_states = tf.split(tf.zeros([args.batch_size, args.rnn_size]), axis=0, num_or_size_splits=args.batch_size)


        # # -----------------------------------------------------------------------------
        # # Define variables for training

        # multi variate Gaussian
        mvn = tf.contrib.distributions.MultivariateNormalDiag(loc=[0.0, 0.0], scale_diag=[1.0, 1.0])

        # placeholders for input and target data
        self.input_data = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.seq_length, self.input_dim], name='data_in')
        self.target_data = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.seq_length, self.input_dim], name='targets')
        self.map_data = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.seq_length, self.map_size, self.map_size, 3], name='map_data_in')

        # initialize cost
        beta = tf.constant(0.99, name="discount_factor")
        self.cost_pose = tf.constant(0.0, name="cost")
        self.cost_valid = tf.constant(0.0, name="cost_valid")
        self.cost_reward = tf.constant(0.0, name="cost_reward")
        self.reward_gt_avg = tf.constant(0.0, name="reward_average")
        self.reward_est_avg = tf.constant(0.0, name="reward_est_average")

        with tf.variable_scope('convlayer'):

            # conv layer
            cw1, cb1 = initialize_conv_filter(shape=[3, 3, 3, 6], name='conv1')
            cw2, cb2 = initialize_conv_filter(shape=[3, 3, 6, 9], name='conv2')
            cw3, cb3 = initialize_conv_filter(shape=[3, 3, 9, 9], name='conv3')

        with tf.variable_scope('reward'):

            fwr = weight_variable(shape=[self.fc_size_in, self.fc_size_out], stddev=0.01, name='fw3')
            fbr = bias_variable(shape=[self.fc_size_out], init=0.0, name='fb3')

        # fully connected layers for embedding and output
        with tf.variable_scope('rnnlm'):

            # embedding for pose input
            embedding_w = tf.get_variable("embedding_w", initializer=tf.truncated_normal(shape=[self.feature_dim, int(args.rnn_size/2)], stddev=0.01))
            embedding_b = tf.get_variable("embedding_b", initializer=tf.constant(0.0, shape=[int(args.rnn_size/2)]))

            # embedding for conv output
            embedding_cw = tf.get_variable("embedding_cw", initializer=tf.truncated_normal(shape=[self.conv_flat_size, int(args.rnn_size/2)], stddev=0.01))
            embedding_cb = tf.get_variable("embedding_cb", initializer=tf.constant(0.0, shape=[int(args.rnn_size/2)]))

            # output
            output_w = tf.get_variable("output_w", initializer=tf.truncated_normal(shape=[args.rnn_size, NOUT], stddev=0.01))
            output_b = tf.get_variable("output_b", initializer=tf.constant(0.0, shape=[NOUT]))



        # # ----------------------------------------------------------------------
        # Processing map info
        conv_out = tf.unstack(tf.zeros(shape=[args.batch_size, args.seq_length, self.conv_flat_size]), axis=1)  # <args.batch_size x self.fc_size_out>, ...
        map_batches = tf.unstack(self.map_data, axis=1) # <args.batch_size x self.map_size x self.map_size x 3>, ...
        for sidx in range(args.seq_length):
            map_batch = map_batches[sidx] # <args.batch_size x self.map_size x self.map_size x 3>
            conv_out[sidx] = shallow_convnet(map_batch, cw1, cb1, cw2, cb2, cw3, cb3)  # <args.batch_size x self.fc_size_out>

        conv_out_reform = tf.stack(conv_out, axis=1) # <args.batch_size x args.seq_length x self.fc_size_out>
        conv_seqs = tf.unstack(conv_out_reform, axis=0) # <args.seq_length x self.fc_size_out> ...


        # # ----------------------------------------------------------------------
        # Processing pose and social info

        # batch_size x (seq_length x input_dim)
        input_seqs = tf.unstack(self.input_data, axis=0)
        target_seqs = tf.unstack(self.target_data, axis=0)

        # batch_size x (seq_length x embedding_size)
        embedding_seqs = tf.unstack(tf.zeros(shape=[args.batch_size, args.seq_length, args.rnn_size]), axis=0)

        # embedding operation
        for i in range(args.batch_size):
            embedding_pose = tf.nn.relu(tf.nn.xw_plus_b(input_seqs[i], embedding_w, embedding_b))
            embedding_conv = tf.nn.relu(tf.nn.xw_plus_b(conv_seqs[i], embedding_cw, embedding_cb))
            embedding_seqs[i] = tf.concat([embedding_pose, embedding_conv], axis=1)

        # # --------------------------------------------------------------------
        # # For each sequence in the input batch
        for b in range(args.batch_size):

            # current embedding sequence : (seq_length x rnn_size)
            current_emd_seq = embedding_seqs[b]

            # current target sequence : (seq_length x input_dim)
            current_tgt_seq = target_seqs[b]

            # current input sequence : (seq_length x input_dim)
            current_input_seq = input_seqs[b]

            # current map sequence
            current_map_seq = conv_seqs[b]

            # # For each frame in a sequence
            for f in range(args.seq_length):

                # current embedding frame : (1, args.rnn_size)
                current_emd_frame = tf.reshape(current_emd_seq[f], shape=(1, args.rnn_size))

                # current target frame : (1, args.input_dim)
                current_tgt_frame = tf.reshape(current_tgt_seq[f], shape=(1, args.input_dim))

                # current input frame : (1, args.input_dim)
                current_input_frame = tf.reshape(current_input_seq[f], shape=(1, args.input_dim))

                # current map frame : (1, self.fc_size_in)
                current_map_frame = tf.reshape(current_map_seq[f], shape=(1, self.conv_flat_size))

                with tf.variable_scope("rnnlm") as scope:
                    if (b > 0 or f > 0):
                        scope.reuse_variables()

                    # go through LSTM cell
                    self.output_states[b], zero_state[b] = cell(current_emd_frame, zero_state[b])

                    # store current state
                    self.state_out[b] = zero_state[b]

                    # fully connected layer for output
                    output = tf.nn.xw_plus_b(self.output_states[b], output_w, output_b)

                    # target data
                    [x1_data, x2_data] = tf.split(axis=1, num_or_size_splits=2, value=current_tgt_frame)

                    # gaussian mixture coefficients
                    [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr] = get_mixture_coef(output)

                    self.pi = o_pi
                    self.mu1 = o_mu1
                    self.mu2 = o_mu2
                    self.sigma1 = o_sigma1
                    self.sigma2 = o_sigma2
                    self.corr = o_corr

                    # # calc reward for ground-truth trajectory ----------------
                    reward_gt = calculate_reward(fwr, fbr, current_map_frame, current_input_frame, current_tgt_frame)


                    # # calc reward for estimates ------------------------------
                    # reward_group = tf.unstack(tf.zeros(shape=(1, 2)), axis=1)
                    idx_max_pi = tf.cast(tf.argmax(o_pi, axis=1), dtype=tf.int32)[0]
                    idx_min_pi = tf.cast(tf.argmin(o_pi, axis=1), dtype=tf.int32)[0]

                    # # reward from max pi distribution
                    # random pose sampling ~N(0, I)
                    rand_pose = tf.unstack(mvn.sample([1]), axis=1)
                    rand_x = rand_pose[0]
                    rand_y = rand_pose[1]

                    # shift and scaling
                    next_x = o_mu1[0][idx_max_pi] + tf.multiply(rand_x, o_sigma1[0][idx_max_pi])
                    next_y = o_mu2[0][idx_max_pi] + tf.multiply(rand_y, o_sigma2[0][idx_max_pi])
                    next_pose = tf.reshape(tf.stack([next_x, next_y]), shape=(1, self.input_dim))

                    # gather rewards
                    reward_max_pi = calculate_reward(fwr, fbr, current_map_frame, current_input_frame, next_pose)

                    # # reward from min pi distribution
                    # random pose sampling ~N(0, I)
                    rand_pose = tf.unstack(mvn.sample([1]), axis=1)
                    rand_x = rand_pose[0]
                    rand_y = rand_pose[1]

                    # shift and scaling
                    next_x = o_mu1[0][idx_min_pi] + tf.multiply(rand_x, o_sigma1[0][idx_min_pi])
                    next_y = o_mu2[0][idx_min_pi] + tf.multiply(rand_y, o_sigma2[0][idx_min_pi])
                    next_pose = tf.reshape(tf.stack([next_x, next_y]), shape=(1, self.input_dim))

                    # gather rewards
                    reward_min_pi = calculate_reward(fwr, fbr, current_map_frame, current_input_frame, next_pose)


                    # find pose that maximizes reward
                    reward_est_max = tf.maximum(reward_max_pi, reward_min_pi)
                    reward_est_min = tf.minimum(reward_max_pi, reward_min_pi)

                    # reward loss : we want to maximize reward_gt - reward_current_policy
                    lossfunc_reward = -tf.log(reward_gt - reward_est_max + 1.0 + 1e-20)  # by criterion 2
                    self.cost_reward += lossfunc_reward

                    # likelihood loss
                    beta_pow = tf.pow(beta, float(args.seq_length - f))
                    lossfunc = get_lossfunc(o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, x1_data, x2_data)
                    self.cost_pose += (lossfunc + 0.0116*tf.multiply(beta_pow, -1.0*tf.log(reward_est_min + 1e-20)))

                    # # Test code -------------------------
                    self.cost_valid += lossfunc
                    self.reward_gt_avg += reward_gt
                    self.reward_est_avg += reward_est_max


        # # --------------------------------------------------------------------
        print('>> network configuration is done ...')

        # normalize cost
        self.cost_pose /= (args.batch_size * args.seq_length)
        self.cost_valid /= (args.batch_size * args.seq_length)
        self.cost_reward /= (args.batch_size * args.seq_length)

        # Test code -----------------------------
        self.reward_gt_avg /= (args.batch_size * args.seq_length)
        self.reward_est_avg /= (args.batch_size * args.seq_length)

        # weight regularization by l2-norm
        tvars = tf.trainable_variables()

        # trainable variables in conv layer
        tvars_conv = [var for var in tvars if 'convlayer' in var.name]
        l2_conv = args.lambda_param * sum(tf.nn.l2_loss(tvar) for tvar in tvars_conv)

        # trainable variables in reward layer
        tvars_reward = [var for var in tvars if 'reward' in var.name]
        l2_reward = args.lambda_param * sum(tf.nn.l2_loss(tvar) for tvar in tvars_reward)

        # trainable variables in rnn layer
        tvars_pose = [var for var in tvars if 'rnnlm' in var.name]
        l2_pose = args.lambda_param * sum(tf.nn.l2_loss(tvar) for tvar in tvars_pose)

        # conv layer need to be trained while training reward layer and rnn layer
        tvars_conv_reward = copy.copy(tvars_reward)
        tvars_conv_pose = copy.copy(tvars_pose)
        for var in tvars_conv:
            tvars_conv_reward.append(var)
            tvars_conv_pose.append(var)


        print('############ Trainable variables : reward ############')
        for var in tvars_conv_reward:
            print(var)

        print('############ Trainable variables : pose ############')
        for var in tvars_conv_pose:
            print(var)


        # add to overall cost
        self.cost_pose = self.cost_pose + l2_pose + l2_conv
        self.cost_reward = self.cost_reward + l2_reward + l2_conv

        # gradient clipping
        grads_pose, _ = tf.clip_by_global_norm(tf.gradients(self.cost_pose, tvars_conv_pose), args.grad_clip)
        grads_reward = tf.gradients(self.cost_reward, tvars_conv_reward)

        # define optimizer, chaged 180523
        optimizer = tf.train.AdamOptimizer(args.learning_rate)

        # define train operation
        self.train_op_pose = optimizer.apply_gradients(zip(grads_pose, tvars_conv_pose))
        self.train_op_reward = optimizer.apply_gradients(zip(grads_reward, tvars_conv_reward))


    def pose_sampling(self, o_pi, o_mu1, o_mu2):

        cur_y, cur_x = 0, 0
        for j in range(self.num_mixture):
            cur_x += o_pi[0][j] * o_mu1[0][j]
            cur_y += o_pi[0][j] * o_mu2[0][j]

        return cur_x, cur_y

    def recall_map(self, map, traj, frame_idx, x_max, y_max, scale):
        x = np.sum(traj[:frame_idx + 1, 0]) * self.args.data_scale
        y = np.sum(traj[:frame_idx + 1, 1]) * self.args.data_scale
        map_roi = map_roi_extract(map, x, y, x_max, y_max, scale, int(self.map_size / 2)).reshape(1, self.map_size, self.map_size, 3)

        return map_roi

    def sample(self, sess, traj, grid, pred_length, map, x_max, y_max, scale):

        # parameters and variables initialization
        seq_length = traj.shape[0]
        obs_length = seq_length - pred_length
        est_traj = np.copy(traj)

        # ----------------------------------------------------------------
        # processing observed trajectories
        prev_state = sess.run(self.cell.zero_state(1, tf.float32))
        prev_state = prev_state.reshape(1, 1, 512)
        for i in range(1, obs_length-1):

            cur_pos = traj[i, :].reshape(1, 1, self.input_dim)
            cur_map = self.recall_map(map, traj, i, x_max, y_max, scale).reshape(1, 1, self.map_size, self.map_size, 3)
            feed = {self.input_data: cur_pos, self.map_data: cur_map, self.state_in: np.array(prev_state)}
            prev_state = sess.run(self.state_out, feed)


        # ----------------------------------------------------------------
        # predict future trajectories
        cur_pos = traj[obs_length-1, :].reshape(1, 1, self.input_dim)
        cur_map = self.recall_map(map, traj, obs_length-1, x_max, y_max, scale).reshape(1, 1, self.map_size, self.map_size, 3)

        #rewards = []
        for i in range(obs_length, seq_length):

            # copy of previous state
            # prev_state_copy = np.copy(prev_state)

            # predict next position
            feed = {self.input_data: cur_pos, self.map_data: cur_map, self.state_in: np.array(prev_state)}
            [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, prev_state] = \
                sess.run([self.pi, self.mu1, self.mu2, self.sigma1, self.sigma2, self.corr, self.state_out], feed)

            # sampling from gaussian mixture model
            cur_x, cur_y = self.pose_sampling(o_pi, o_mu1, o_mu2)

            # calculate reward of current prediction
            #next_pos = np.array([cur_x, cur_y]).reshape(1, 1, self.input_dim)
            #feed = {self.rwd_in_cur_pose: cur_pos[0], self.rwd_in_next_pose: next_pos[0], self.rwd_in_cur_state: np.array(prev_state_copy[0]), self.rwd_in_map_data: cur_map}
            #reward = sess.run(self.reward_call, feed)
            #rewards.append(reward)

            # save prediction and delay
            est_traj[i, :] = [np.mean(cur_x), np.mean(cur_y)]
            cur_pos = np.array([cur_x, cur_y]).reshape(1, 1, self.input_dim)
            cur_map = self.recall_map(map, est_traj, i, x_max, y_max, scale).reshape(1, 1, self.map_size, self.map_size, 3)

        return est_traj, 0

    def run_prediction(self, sess, input, state):

        feed = {self.input_data: input.reshape(1, 1, self.input_dim),
                    self.state_in: state.reshape(1, 1, self.args.rnn_size*2)}

        [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, prev_state] = \
                sess.run([self.pi, self.mu1, self.mu2, self.sigma1, self.sigma2, self.corr, self.state_out], feed)

        return o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, prev_state

    def sample_beamsearch(self, sess, traj, grid, pred_length, map, x_max, y_max, scale, beam_width, num_candidates):

        def get_pi_idx(x, pdf):
            N = pdf.size
            accumulate = 0
            for i in range(0, N):
                accumulate += pdf[i]
                if (accumulate >= x):
                    return i
            print('error with sampling ensemble')
            return -1

        def sample_gaussian_2d(mu1, mu2, s1, s2, rho):
            mean = [mu1, mu2]
            cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
            x = np.random.multivariate_normal(mean, cov, 1)
            return x[0][0], x[0][1]

        # # -------------------------------------------------------------
        # parameter setting
        seq_length = traj.shape[0]
        obs_length = seq_length - pred_length
        cell_size = 2 * self.args.rnn_size
        num_candidates = self.args.num_mixture


        # # --------------------------------------------------------------
        # # process along the observation sequence
        prev_state = sess.run(self.cell.zero_state(1, tf.float32))
        prev_state = prev_state.reshape(1, 1, cell_size)

        ''' the first frame of traj is not used !! '''
        ''' from 1 to obs_length-2 are processed !! '''
        for i in range(1, obs_length-1):

            ''' 
            # current position and state vector
            cur_pos = traj[i, :].reshape(1, 1, self.input_dim)
            feed = {self.input_data: cur_pos,
                    self.state_in: np.array(prev_state)}
            prev_state = sess.run(self.state_out, feed)
            '''


            # TODO : test here if the trained reward function is reliable !!
            # # TEST CODE ---------------------------------------------------
            # copy current state
            prev_state_copy = np.copy(prev_state)

            # current position and state vector
            cur_pos = traj[i, :].reshape(1, 1, self.input_dim)
            feed = {self.input_data: cur_pos,
                    self.state_in: np.array(prev_state)}            
            
            # run prediction
            [o_pi, o_mu1, o_mu2, prev_state] = \
                sess.run([self.pi, self.mu1, self.mu2, self.state_out], feed)

            # sampling path
            next_x, next_y = self.pose_sampling(o_pi, o_mu1, o_mu2)
            next_pos_est = np.array([next_x, next_y]).reshape(1, 1, self.input_dim)
            #next_pos_est = np.array([1.0, 1.0]).reshape(1, 1, self.input_dim)

            # load map roi
            cur_map = self.recall_map(map, traj, i, x_max, y_max, scale)

            # calculate reward for ground-truth --------------
            next_pos_gt = traj[i+1, :].reshape(1, 1, self.input_dim)
            feed = {self.rwd_in_cur_pose: cur_pos.reshape(1, 2),
                    self.rwd_in_next_pose: next_pos_gt.reshape(1, 2),
                    self.rwd_in_cur_state: prev_state_copy.reshape(1, 512),
                    self.rwd_in_map_data: cur_map}
            reward_gt = sess.run(self.reward_call, feed)

            # calculate reward for estimate --------------
            feed = {self.rwd_in_cur_pose: cur_pos.reshape(1, 2),
                    self.rwd_in_next_pose: next_pos_est.reshape(1, 2),
                    self.rwd_in_cur_state: prev_state_copy.reshape(1, 512),
                    self.rwd_in_map_data: cur_map}
            reward_est = sess.run(self.reward_call, feed)
            #plt.imshow(np.squeeze(cur_map))
            #plt.show()
            a = 0
            # # ------------------------------------------------------------



        # # -------------------------------------------------------------
        # # prediction based on the observation

        # accumulate the scores of the best 'beam_width' candidates
        acc_score = np.ones(shape=(beam_width, 1))

        # the features of the best 'beam_width' candidates
        cur_pos_beam = np.zeros(shape=(beam_width, self.input_dim))

        # previous state vectors
        prev_state_beam = np.zeros(shape=(beam_width, cell_size))

        # current state vectors
        curr_state_beam = np.zeros(shape=(beam_width, cell_size))

        # copy previous state vectors for reward calculation
        prev_state_beam_copy = np.zeros(shape=(beam_width, cell_size))

        ''' current input is now (obs_length-1)-th frame !!! '''
        cur_map_beam = []
        for i in range(beam_width):
            cur_pos_beam[i, :] = traj[obs_length-1, :]
            prev_state_beam[i, :] = np.array(prev_state)[0]
            curr_state_beam[i, :] = np.array(prev_state)[0]
            cur_map_beam.append(self.recall_map(map, traj, obs_length - 1, x_max, y_max, scale))


        # # ---------------------------------------------------------------
        # # for all time indices for prediction
        cur_seq_len = 1
        est_traj_acc = np.zeros(shape=(beam_width * num_candidates, cur_seq_len, self.input_dim))
        for i in range(obs_length, seq_length):

            # the number of predicted frames until now
            cur_seq_len = i - obs_length + 1

            # for all the best 'beam_width' candidates
            score_mem = np.zeros(shape=(beam_width*num_candidates, 1))
            pose_mem = np.zeros(shape=(beam_width*num_candidates, 2))

            ''' for the first iteration, the best beam-width poses are all traj[obs_length-1, :]'''
            beam_width_iter = beam_width
            if (i == obs_length):
                beam_width_iter = 1

            # for each beam-line
            for j in range(beam_width_iter):

                # copy previous state
                prev_state_beam_copy[j, :] = prev_state_beam[j, :]

                # run prediction
                [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, prev_state] \
                    = self.run_prediction(sess, cur_pos_beam[j, :], prev_state_beam[j, :])

                # save current state
                curr_state_beam[j, :] = np.array(prev_state)[0]

                # for all possible candidate poses
                for k in range(num_candidates):

                    #idx = get_pi_idx(random.random(), o_pi[0])
                    #dx, dy = sample_gaussian_2d(
                    #    o_mu1[0][idx], o_mu2[0][idx], o_sigma1[0][idx], o_sigma2[0][idx], o_corr[0][idx])

                    #est_traj_acc[j * num_candidates + k, cur_seq_len-1, 0] = dx
                    #est_traj_acc[j * num_candidates + k, cur_seq_len-1, 1] = dy

                    #pose_mem[j * num_candidates + k, 0] = dx
                    #pose_mem[j * num_candidates + k, 1] = dy

                    #next_pos = np.array([dx, dy]).reshape(1, self.input_dim)

                    est_traj_acc[j * num_candidates + k, cur_seq_len-1, 0] = o_mu1[0, k]
                    est_traj_acc[j * num_candidates + k, cur_seq_len-1, 1] = o_mu2[0, k]

                    pose_mem[j * num_candidates + k, 0] = o_mu1[0, k]
                    pose_mem[j * num_candidates + k, 1] = o_mu2[0, k]

                    next_pos = np.array([o_mu1[0, k], o_mu2[0, k]]).reshape(1, self.input_dim)

                    feed = {self.rwd_in_cur_pose: cur_pos_beam[j, :].reshape(1, 2),
                            self.rwd_in_next_pose: next_pos,
                            self.rwd_in_cur_state: prev_state_beam_copy[j, :].reshape(1, 512),
                            self.rwd_in_map_data: cur_map_beam[j]}
                    reward = sess.run(self.reward_call, feed)

                    score_mem[j * num_candidates + k, 0] += acc_score[j, 0] * o_pi[0, k]
                    # score_mem[j * num_candidates + k, 0] += acc_score[j, 0] * reward[0]

            # sort score in descending order
            indices = np.argsort(-1.0*score_mem, axis=0)

            cur_map_beam = []
            best_trajs = np.zeros(shape=(beam_width, cur_seq_len, self.input_dim))
            for b in range(beam_width):

                # target candidate index
                target_idx = int(indices[b])

                # corresponding beam index
                beam_idx = int(target_idx / num_candidates)  # beam index

                # store current state
                prev_state_beam[b, :] = curr_state_beam[beam_idx, :]

                # store current pose
                cur_pos_beam[b, 0] = pose_mem[target_idx, 0]
                cur_pos_beam[b, 1] = pose_mem[target_idx, 1]
                cur_pos_beam[b, 2:] = traj[i, 2:]

                # store score
                acc_score[b, 0] = score_mem[target_idx]

                # store traj
                best_trajs[b, :, 0] = est_traj_acc[target_idx, :, 0]
                best_trajs[b, :, 1] = est_traj_acc[target_idx, :, 1]

                # extract roi map corresponding to the best trajectories
                traj_cur = np.zeros(shape=(i+1, 2))
                traj_cur[:obs_length, :] = np.copy(traj[:obs_length, :])
                traj_cur[obs_length:i+1, :] = np.copy(best_trajs[b, :, :])
                cur_map = self.recall_map(map, traj_cur, i, x_max, y_max, scale)
                cur_map_beam.append(cur_map)


            # for the next time step prediction
            est_traj_acc = np.zeros(shape=(beam_width * num_candidates, cur_seq_len+1, self.input_dim))
            for b in range(beam_width):
                for j in range(num_candidates):
                    est_traj_acc[b * num_candidates + j, :cur_seq_len, 0] = best_trajs[b, :, 0]
                    est_traj_acc[b * num_candidates + j, :cur_seq_len, 1] = best_trajs[b, :, 1]

        return best_trajs, acc_score