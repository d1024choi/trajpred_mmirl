import pickle
from kitti_functions import *

class DataLoader:

    def __init__(self, args):

        self.dataset_path = args.dataset_path
        self.batch_size = args.batch_size
        self.batch_size_valid = 1
        self.seq_length = args.seq_length
        self.scale_factor = args.data_scale
        self.social_range = args.social_range
        self.social_grid_size = args.grid_size
        self.map_size = args.map_size
        self.is_apply_social = 0
        self.data_augmentation = args.data_augmentation


        self.load_preprocessed_data()
        self.reset_batch_pointer()
        print('>> Dataset loading and analysis process are done...')

    def reset_batch_pointer(self, ):
        self.pointer = 0
        self.frame_pointer = 0

    def tick_batch_pointer(self):
        self.pointer += 1
        if (self.pointer >= len(self.train_data)):
            self.pointer = 0

    def load_preprocessed_data(self):

        '''
        raw_data is a list that has three components
        component1) trajectory data for training
        component2) trajectory data for validation and visualization
        '''

        f = open(self.dataset_path, 'rb')
        raw_data = pickle.load(f)
        f.close()

        # for training data --------------------
        counter = 0
        self.train_data = []
        for data in raw_data[0]:
            scaled_data = np.copy(data)
            self.train_data.append(scaled_data)
            counter += int(len(scaled_data) - self.seq_length)

        # assume we visit every frame as a start point of a short trajectory
        # in one training epoch
        self.num_batches = int(counter / self.batch_size)


        # for validation data --------------------
        self.valid_data = []
        for data in raw_data[1]:
            scaled_data = np.copy(data)
            self.valid_data.append(scaled_data)

        # for map data ----------------------------
        self.map = []
        for data in raw_data[2]:
            self.map.append(data)

        # for map info ----------------------------
        self.map_info = []
        for data in raw_data[3]:
            self.map_info.append(data)


    def preprocess_sequence(self, seq, isValid, isDiff):
        '''
        dataset id (0)
        object id (1)
        target pose (2~3)
        neighbor pose (4~63)
        '''

        seq_len = seq.shape[0]
        seq_tpose = np.copy(seq[:, 2:4])
        seq_npose = np.copy(seq[:, 4:64]).reshape(seq_len, 30, 2)


        # # load map
        dataset_index = int(seq[0, 0])
        map = self.map[dataset_index]
        x_max, y_max, scale = self.map_info[dataset_index]

        # # map roi extraction ------------------------------------------
        seq_map = []
        for i in range(seq_tpose.shape[0]):
            x = seq_tpose[i, 0]
            y = seq_tpose[i, 1]

            corr_map = map_roi_extract(map, x, y, x_max, y_max, scale, int(self.map_size/2))
            seq_map.append(corr_map)

            # # TEST code ------------------------------------
            ''' 
            map_ = np.copy(np.copy(map_roi[i]))
            map_row_cnt = map_.shape[0] / 2
            map_col_cnt = map_.shape[1] / 2

            pose_start_x = seq_tpose[i, 0]
            pose_start_y = seq_tpose[i, 1]

            for kappa in range(0, seq_tpose.shape[0]-i):

                pose_x = int(3 * (seq_tpose[i+kappa, 0] - pose_start_x) + map_row_cnt)
                pose_y = int(3 * (seq_tpose[i+kappa, 1] - pose_start_y) + map_col_cnt)

                pose_x = _min(_max(pose_x, 0), map_.shape[0] - 1)
                pose_y = _min(_max(pose_y, 0), map_.shape[1] - 1)

                map_[pose_x, pose_y, 0] = 0
                map_[pose_x, pose_y, 1] = int(255.0 * float(i+kappa) / float(seq_tpose.shape[0]-1))
                map_[pose_x, pose_y, 2] = 255

            cv2.imshow('test', map_)
            cv2.waitKey(0)
            '''


        # # apply augmentation -------------------------------------------
        # 0: none, 1: random flip, 2: random rotation, 3: random flip+scaling, 4: random rotation+scaling
        if (isValid):
            donothing = 0
        else:
            if (self.data_augmentation == 1):
                seq_tpose, seq_npose, seq_map = random_flip(seq_tpose, seq_npose, seq_map)
            elif (self.data_augmentation == 2):
                # TODO : random rotation of map needs to be implemented
                seq_tpose, seq_npose = random_rotate(seq_tpose, seq_npose)

        # # TEST code ------------------------------------
        ''' 
        for i in range(seq_tpose.shape[0]):
            map_ = np.copy(np.copy(map_roi[i]))
            map_row_cnt = map_.shape[0] / 2
            map_col_cnt = map_.shape[1] / 2

            pose_start_x = seq_tpose[i, 0]
            pose_start_y = seq_tpose[i, 1]

            for kappa in range(0, seq_tpose.shape[0]-i):

                pose_x = int(3 * (seq_tpose[i+kappa, 0] - pose_start_x) + map_row_cnt)
                pose_y = int(3 * (seq_tpose[i+kappa, 1] - pose_start_y) + map_col_cnt)

                pose_x = _min(_max(pose_x, 0), map_.shape[0] - 1)
                pose_y = _min(_max(pose_y, 0), map_.shape[1] - 1)

                map_[pose_x, pose_y, 0] = 0
                map_[pose_x, pose_y, 1] = int(255.0 * float(i+kappa) / float(seq_tpose.shape[0]-1))
                map_[pose_x, pose_y, 2] = 255

            cv2.imshow('test', map_)
            cv2.waitKey(0)
        '''

        # # TEST CODE-------------------------------------
        '''' 
        ego = np.copy(seq_tpose)
        plt.plot(ego[:, 0], ego[:, 1], 'o')
        for i in range(30):
            ngh = np.squeeze(seq_npose[:, i, :]) # seq_len x 2
            ngh_ = ngh[ngh[:, 0]>-1000, :]
            if (len(ngh_) > 1):
                plt.plot(ngh_[:, 0], ngh[:, 1], '+')
        plt.show()
        '''

        # # create social vectors (ok) -----------------------------------------
        num_grid = int(self.social_range / self.social_grid_size)
        # seq_sgrid = np.zeros(shape=(seq_len, num_grid, num_grid))
        seq_sgrid = np.zeros(shape=(seq_len, num_grid*num_grid))
        for i in range(seq_len):
            social_grid = np.zeros(shape=(num_grid, num_grid))
            target_pose = seq_tpose[i, :].reshape(1, 2)
            neighbors_pose = seq_npose[i, :]
            for j in range(30):
                if (neighbors_pose[j, 0] == -1000 or neighbors_pose[j, 0] == 1000):
                    continue
                else:
                    neighbor_pose = neighbors_pose[j, :].reshape(1, 2)
                    social_grid = getSocialMatrix(social_grid, target_pose, neighbor_pose, self.social_range, self.social_grid_size)

            seq_sgrid[i, :] = social_grid.reshape(1, num_grid*num_grid)


        # # pose difference -----------------------------------------------
        seq_tpose_cur = np.copy(seq_tpose[1:, :])      # set_tpose[1:seq_len-1]
        seq_tpose_pre = np.copy(seq_tpose[:-1, :])     # set_tpose[0:seq_len-2]
        seq_tpose_diff = seq_tpose_cur - seq_tpose_pre

        if (isDiff):
            return (seq_tpose_diff/self.scale_factor), np.copy(seq_sgrid[1:, :]), np.array(seq_map[1:])
        else:
            return seq_tpose_cur, np.copy(seq_sgrid[1:, :]), np.array(seq_map[1:])

    def next_batch(self):

        '''
        Read a batch randomly
        :x_batch: <batch size x seq_length x input_dim>
        :y_batch: <batch size x seq_length x input_dim>
        :d_batch: <batch size x seq_length>
        '''

        x_batch = []
        y_batch = []
        sg_batch = []
        map_batch = []
        d_batch = []
        for i in range(self.batch_size):

            data = self.train_data[self.pointer]

            idx = random.randint(0, len(data) - self.seq_length - 2)
            seq_all = np.copy(data[idx:idx + self.seq_length + 2])

            # TODO : non-preprocessed data needs to be augmented and processed HERE
            seq_all_proc, seq_sgrid, seq_map = self.preprocess_sequence(seq_all, isValid=False, isDiff=True)

            seq_x = np.copy(seq_all_proc[0:self.seq_length])
            seq_y = np.copy(seq_all_proc[1:self.seq_length + 1])
            seq_sgrid_x = np.copy(seq_sgrid[0:self.seq_length, :])

            y_batch.append(seq_y)
            x_batch.append(seq_x)
            sg_batch.append(seq_sgrid_x)
            map_batch.append(seq_map[0:self.seq_length])
            d_batch.append([self.pointer, idx])


            ''' 
            if len(data) is smaller than 50, self.seq_length is 24
            n_batch is 1, therefore, (1.0 / n_batch) is 1
            then the following is the same as
            if random.random() < 1, then go next with prob. 1

            if len(data) is greater than 50, self.seq_length is 24
            n_batch is 2, therefore, (1.0 / n_batch) is 0.5
            then the following is the same as
            if random.random() < 0.5, then go next with prob. 0.5
            '''
            n_batch = int(len(data) / (self.seq_length + 2))
            if random.random() < (1.0 / float(n_batch)):
                self.tick_batch_pointer()

        return x_batch, y_batch, sg_batch, map_batch, d_batch

    def next_batch_valid(self):

        '''
        Read a batch randomly for validation during training
        :x_batch: <batch size x seq_length x input_dim>
        :y_batch: <batch size x seq_length x input_dim>
        :d_batch: <batch size x seq_length>
        '''

        x_batch = []
        y_batch = []
        sg_batch = []
        map_batch = []
        d_batch = []

        counter = 0
        while (len(x_batch) < self.batch_size):

            data = self.valid_data[self.pointer]

            if (self.frame_pointer < len(data) - self.seq_length - 1):

                idx = self.frame_pointer
                seq_all = np.copy(data[idx:idx + self.seq_length + 2])

                # TODO : non-preprocessed data needs to be augmented and processed HERE
                seq_all_proc, seq_sgrid, seq_map = self.preprocess_sequence(seq_all, isValid=True, isDiff=True)

                seq_x = np.copy(seq_all_proc[0:self.seq_length])
                seq_y = np.copy(seq_all_proc[1:self.seq_length + 1])
                seq_sgrid_x = np.copy(seq_sgrid[0:self.seq_length, :])

                y_batch.append(seq_y)
                x_batch.append(seq_x)
                sg_batch.append(seq_sgrid_x)
                map_batch.append(seq_map[0:self.seq_length])
                d_batch.append([self.pointer, idx])

                # move a quarter of seq. length steps
                self.frame_pointer += int(self.seq_length/4)

            else:
                if (self.pointer >= len(self.valid_data)-1):
                    x_batch = []
                    y_batch = []
                    sg_batch = []
                    d_batch = []
                    return x_batch, y_batch, sg_batch, map_batch, d_batch
                else:
                    self.pointer += 1
                    self.frame_pointer = 0

            counter += 1

        return x_batch, y_batch, sg_batch, map_batch, d_batch

    def next_sequence_valid(self):

        '''

        dataset id (0)
        object id (1)
        target pose (2~3)
        neighbor pose (4~63)

        Read a batch randomly for validation and visualization
        :x_batch: <batch size x seq_length x input_dim>
        :y_batch: <batch size x seq_length x input_dim>
        :d_batch: <batch size x seq_length>
        '''

        NotEndOfData = True
        while(NotEndOfData):
            if (self.pointer >= len(self.valid_data)):
                x = []
                grid = []
                map = []
                x_max = []
                y_max = []
                scale = []
                dataset_index = []
                NotEndOfData = False
                break
            else:
                if (self.frame_pointer >= len(self.valid_data[self.pointer]) - self.seq_length - 2):
                    self.frame_pointer = 0
                    self.pointer += 1
                else:
                    data = self.valid_data[self.pointer]
                    idx = self.frame_pointer

                    seq_all = np.copy(data[idx:idx + self.seq_length + 1])

                    # # load map
                    dataset_index = int(seq_all[0, 0])
                    map = self.map[dataset_index]
                    x_max, y_max, scale = self.map_info[dataset_index]

                    # TODO : non-preprocessed data needs to be augmented and processed HERE
                    seq_all_proc, seq_sgrid, seq_map = self.preprocess_sequence(seq_all, isValid=True, isDiff=False)

                    x = np.copy(seq_all_proc[0:self.seq_length + 1])
                    grid = np.copy(seq_sgrid[0:self.seq_length + 1])

                    print('seq_pointer %d, frame_pointer %d' % (self.pointer, self.frame_pointer))
                    self.frame_pointer += int(self.seq_length + 1)
                    break


        return x, grid, map, x_max, y_max, scale, dataset_index, NotEndOfData