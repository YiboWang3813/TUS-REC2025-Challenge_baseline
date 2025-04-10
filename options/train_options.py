# Hyperparameters for training and generating DDFs

from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--PRED_TYPE', type=str,default='parameter',help='network output type: {"transform", "parameter", "point", "quaternion"}')
        self.parser.add_argument('--LABEL_TYPE', type=str,default='point',help='label type: {"point", "parameter", "transform"}')
        self.parser.add_argument('--NUM_SAMPLES', type=int,default=2,help='number of input frames/imgs')
        self.parser.add_argument('--SAMPLE_RANGE', type=int,default=2,help='from which the input frames/imgs are selected from')
        self.parser.add_argument('--NUM_PRED', type=int,default=1,help='to those frames/imgs, transformation matrix are predicted ')
        self.parser.add_argument('--model_name', type=str,default='efficientnet_b1',help='network name')

        self.parser.add_argument('--retrain',default=False,action='store_true',help='whether retrain the model from a specific epoch')
        self.parser.add_argument('--retrain_epoch', type=str,default='00000000',help='retrain from which epoch')
        self.parser.add_argument('--MINIBATCH_SIZE', type=int,default=16,help='input batch size')
        self.parser.add_argument('--LEARNING_RATE',type=float,default=1e-4,help='learing rate')
        self.parser.add_argument('--NUM_EPOCHS',type =int,default=int(1e8),help='# of epochs to train')
        self.parser.add_argument('--FREQ_INFO', type=int, default=10,help='frequency of printing info')
        self.parser.add_argument('--FREQ_SAVE', type=int, default=100,help='frequency of saving model')
        self.parser.add_argument('--val_fre', type=int, default=1,help='frequency of validation')

        self.parser.add_argument('--FILENAME_VAL', type=str, default="fold_03", help='validation json file')
        self.parser.add_argument('--FILENAME_TEST', type=str, default="fold_04", help='test json file')
        self.parser.add_argument('--FILENAME_TRAIN', type=list,default=["fold_00", "fold_01", "fold_02"],help='training json file')

        self.isTrain= True
