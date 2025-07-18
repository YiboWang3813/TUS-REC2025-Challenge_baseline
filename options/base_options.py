# Hyperparameters for dataroot and GPU

import argparse
import os

self_data_dir = '/raid/liujie/code_recon/data/ultrasound/Freehand_US_data_train_2025'

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # self.parser.add_argument('--DATA_PATH', type=str, default='data/frames_transfs', help='foldername of dataset path')
        # self.parser.add_argument('--FILENAME_CALIB', type=str, default='data/calib_matrix.csv',help='dataroot of calibration matrix')
        # self.parser.add_argument('--LANDMARK_PATH', type=str, default='data/landmarks', help='foldername of path for landmark')
        self.parser.add_argument('--DATA_PATH', type=str, default='{}/frames_transfs'.format(self_data_dir), help='foldername of dataset path')
        self.parser.add_argument('--FILENAME_CALIB', type=str, default='{}/calib_matrix.csv'.format(self_data_dir), help='dataroot of calibration matrix')
        self.parser.add_argument('--LANDMARK_PATH', type=str, default='{}/landmarks'.format(self_data_dir), help='foldername of path for landmark')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu id: e.g., 0,1,2...')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain

        args = vars(self.opt)

        print('----------Option----------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('----------Option----------')

        # create saved result path
        saved_results = 'seq_len' + str(self.opt.NUM_SAMPLES) + '__' + self.opt.model_name + '__' + 'lr' + str(self.opt.LEARNING_RATE)\
        + '__pred_type_'+str(self.opt.PRED_TYPE) + '__label_type_'+str(self.opt.LABEL_TYPE) 
        self.opt.SAVE_PATH = os.path.join('results', saved_results)
        
        if not os.path.exists(os.path.join(os.getcwd(),self.opt.SAVE_PATH)):
            os.makedirs(os.path.join(os.getcwd(),self.opt.SAVE_PATH))
        if not os.path.exists(os.path.join(os.getcwd(),self.opt.SAVE_PATH, 'saved_model')):
            os.makedirs(os.path.join(os.getcwd(),self.opt.SAVE_PATH, 'saved_model'))
        if not os.path.exists(os.path.join(os.getcwd(),self.opt.SAVE_PATH, 'train_results')):
            os.makedirs(os.path.join(os.getcwd(),self.opt.SAVE_PATH, 'train_results'))
        if not os.path.exists(os.path.join(os.getcwd(),self.opt.SAVE_PATH, 'val_results')):
            os.makedirs(os.path.join(os.getcwd(),self.opt.SAVE_PATH, 'val_results'))

        # save configurations
        file_name = os.path.join(self.opt.SAVE_PATH, 'config.txt')
        with open(file_name, 'a') as opt_file:
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s' % (str(k), str(v)))
                opt_file.write('\n')
        
        return self.opt
