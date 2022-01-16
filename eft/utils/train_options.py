import os
import json
import argparse
import numpy as np
from collections import namedtuple
import datetime

class TrainOptions():

    def __init__(self):
        self.parser = argparse.ArgumentParser()

        req = self.parser.add_argument_group('Required')
        req.add_argument('--name', required=True, help='Name of the experiment')

        gen = self.parser.add_argument_group('General')
        gen.add_argument('--time_to_run', type=int, default=np.inf, help='Total time to run in seconds. Used for training in environments with timing constraints')
        gen.add_argument('--resume', dest='resume', default=False, action='store_true', help='Resume from checkpoint (Use latest checkpoint by default')
        gen.add_argument('--num_workers', type=int, default=12, help='Number of processes used for data loading')
        pin = gen.add_mutually_exclusive_group()
        pin.add_argument('--pin_memory', dest='pin_memory', action='store_true')
        pin.add_argument('--no_pin_memory', dest='pin_memory', action='store_false')
        gen.set_defaults(pin_memory=True)

        io = self.parser.add_argument_group('io')
        io.add_argument('--log_dir', default='logs', help='Directory to store logs')
        io.add_argument('--checkpoint', default=None, help='Path to checkpoint')
        io.add_argument('--from_json', default=None, help='Load options from json file instead of the command line')
        io.add_argument('--pretrained_checkpoint', default=None, help='Load a pretrained checkpoint at the beginning training') 

        train = self.parser.add_argument_group('Training Options')
        train.add_argument('--num_epochs', type=int, default=500, help='Total number of training epochs')
        train.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
        train.add_argument('--batch_size', type=int, default=256, help='Batch size')
        train.add_argument('--summary_steps', type=int, default=100, help='Summary saving frequency')
        # train.add_argument('--test_steps', type=int, default=1000, help='Testing frequency during training')
        train.add_argument('--test_epoch_inter', type=int, default=1, help='Testing frequency during training')
        train.add_argument('--save_epoch_inter', type=int, default=100, help='Save frequency during training')
        train.add_argument('--checkpoint_steps', type=int, default=10000, help='Checkpoint saving frequency')
        train.add_argument('--img_res', type=int, default=224, help='Rescale bounding boxes to size [img_res, img_res] before feeding them in the network') 
        train.add_argument('--rot_factor', type=float, default=30, help='Random rotation in the range [-rot_factor, rot_factor]') 
        train.add_argument('--noise_factor', type=float, default=0.4, help='Random rotation in the range [-rot_factor, rot_factor]') 
        train.add_argument('--scale_factor', type=float, default=0.25, help='rescale bounding boxes by a factor of [1-options.scale_factor,1+options.scale_factor]') 
        train.add_argument('--ignore_3d', default=False, action='store_true', help='Ignore GT 3D data (for unpaired experiments') 
        train.add_argument('--shape_loss_weight', default=0, type=float, help='Weight of per-vertex loss') 
        train.add_argument('--keypoint_loss_weight', default=5., type=float, help='Weight of 2D and 3D keypoint loss') 
        train.add_argument('--pose_loss_weight', default=1., type=float, help='Weight of SMPL pose loss') 
        train.add_argument('--beta_loss_weight', default=0.001, type=float, help='Weight of SMPL betas loss') 
        train.add_argument('--openpose_train_weight', default=0.,  type=float, help='Weight for OpenPose keypoints during training') 
        train.add_argument('--gt_train_weight', default=1., help='Weight for GT keypoints during training') 
        train.add_argument('--run_smplify', default=False, action='store_true', help='Run SMPLify during training') 
        train.add_argument('--smplify_threshold', type=float, default=100., help='Threshold for ignoring SMPLify fits during training') 
        train.add_argument('--num_smplify_iters', default=100, type=int, help='Number of SMPLify iterations') 

        #My DB
        train.add_argument('--db_set', default='coco', type=str, help='used DB for training') 


        #Set target db npz file. If None, save here the default file name from config.DATASET_FILES
        train.add_argument('--db_cocoall3d_name', default=None, type=str, help='target npz file name') 
        train.add_argument('--db_coco3d_name', default=None, type=str, help='target npz file name') 
        train.add_argument('--db_mpii3d_name', default=None, type=str, help='target npz file name')        #Should be set correctly here
        train.add_argument('--db_lspet3d_name', default=None, type=str, help='target npz file name')        #Should be set correctly here

        train.add_argument('--db_posetrack3d_name', default=None, type=str, help='target npz file name')        #Should be set correctly here
        train.add_argument('--db_cocoplus3d_name', default=None, type=str, help='target npz file name')        #Should be set correctly here

        # train.add_argument('--db_p_name', default='lspet_11-01-46573_lspet.npz', type=str, help='target npz file name')        #Should be set correctly here

        #Other training option
        train.add_argument('--noEval', dest='noEval', action='store_true', help='do eval')
        train.add_argument('--bUseWeakProj', default=False, dest='bUseWeakProj', action='store_true', help='if Yes, use weak projection')
        train.add_argument('--bUseKneePrior', default=False, dest='bUseKneePrior', action='store_true', help='if Yes, use knee prior for exemplar')
        train.add_argument('--bFacePartTest', default=False, dest='bFacePartTest', action='store_true', help='if Yes, consider more extremely thing')


        # Extreme Crop Test
        train.add_argument('--bUpperBodyTest', default=False, dest='bUpperBodyTest', action='store_true', help='if Yes, randomly choose upperbody cropping')
        train.add_argument('--upperBodyTest_prob', type=float, default=0.3, help='30 percent chance of upper body only test') 
        train.add_argument('--extCrop_bEnforceUpperBody', default=False,  action='store_true', help='Always upper body croppping') 

        train.add_argument('--multilevel_crop', default=-1, type=int, dest='multilevel_crop', help='default -1 (no crop). 0 (head only) - 8 (fullbody)')
        train.add_argument('--multilevel_crop_rand', action='store_true', help='If true, do random crop. Higher priority than multilevel_crop')
        train.add_argument('--multilevel_crop_rand_prob', default=0.3, type=int, dest='multilevel_crop_rand_prob', help='probability to apply random body cropping. 0.3 means 30 percent chance')
        

        #Exemplar Mode
        train.add_argument('--exemplar_targetIdx', default=-1, type=str, help='Target Idx currently processing exemplar tuning')         
        train.add_argument('--bExemplarMode', dest='bExemplarMode', default=False,  action='store_true', help='Exemplar Setting (e.g., no aug)')
        train.add_argument('--bExemplarWith3DSkel', dest='bExemplarWith3DSkel', default=False,  action='store_true', help='If true, use 3D skeleton for Exemplar SMPL fitting')
        train.add_argument('--bNotSkipExemplar', dest='bNotSkipExemplar', default=False,  action='store_true', help='If true, always rerun')
        
        
        train.add_argument('--bExemplar_dataLoaderStart', type=int, default=-1, help='Start from this ')
        train.add_argument('--bExemplar_dataLoaderEnd', type=int, default=-1, help='Run until this')
        train.add_argument('--maxExemplarIter', default=20, type=int, help='Max Iteration for exemplar tuning') 


        train.add_argument('--bExemplar_analysis_testloss', dest='bExemplar_analysis_testloss', default=False,  action='store_true', help='If True, run test after each exemplar tuning')
        train.add_argument('--bExemplar_badsample_finder', dest='bExemplar_badsample_finder', default=False,  action='store_true', help='If True, run test after each exemplar tuning')


        #SMPLX model is used 
        train.add_argument('--bUseSMPLX', default=False, dest='bUseSMPLX', action='store_true', help='if Yes, use SMPL-X instead of smpl')
        train.add_argument('--bUseHand3D', default=False, dest='bUseHand3D', action='store_true', help='if Yes, use Hand 3d joint')
        train.add_argument('--bUseHand2D', default=False, dest='bUseHand2D', action='store_true', help='if Yes, use Hand 2d keypoint')

        #Ablation Study Option
        train.add_argument('--ablation_loss_2dkeyonly',default=False, action='store_true', help="ablation study: using 2d keypoint loss only") 
        train.add_argument('--ablation_loss_noSMPLloss',default=False, action='store_true', help="ablation study: no smpl parameter") 
        train.add_argument('--ablation_no_pseudoGT',default=False, action='store_true', help="ablation study: not suing pseudoGT and using 2D keypoint loss only for 2D datasets.") 


        #Ablation Study Option
        # train.add_argument('--ablation_freeze_afterResnet',default=False, action='store_true', help="ablation study: using 2d keypoint loss only") 
        train.add_argument('--ablation_layerteset_onlyLayer4',default=False, action='store_true', help="ablation study") 
        train.add_argument('--ablation_layerteset_onlyAfterRes',default=False, action='store_true', help="ablation study") 
        train.add_argument('--ablation_layerteset_Layer4Later',default=False, action='store_true', help="ablation study") 
        train.add_argument('--ablation_layerteset_onlyRes',default=False, action='store_true', help="ablation study") 


        train.add_argument('--ablation_layerteset_Layer3Later',default=False, action='store_true', help="ablation study") 
        train.add_argument('--ablation_layerteset_Layer2Later',default=False, action='store_true', help="ablation study") 
        train.add_argument('--ablation_layerteset_Layer1Later',default=False, action='store_true', help="ablation study") 
        train.add_argument('--ablation_layerteset_all',default=False, action='store_true', help="ablation study") 
        train.add_argument('--ablation_layerteset_onlyRes_withconv1',default=False, action='store_true', help="ablation study") 

        train.add_argument('--ablation_layerteset_decOnly',default=False, action='store_true', help="ablation study") 
        train.add_argument('--ablation_layerteset_fc2Later',default=False, action='store_true', help="ablation study")    

        train.add_argument('--ablation_layerteset_onlyRes50LastConv',default=False, action='store_true', help="ablation study")    


        #Ablation Study. SMPLify different weight
        train.add_argument('--ablation_smplify_noCamOptFirst',default=False, action='store_true', help="ablation study")    
        train.add_argument('--ablation_smplify_noPrior',default=False, action='store_true', help="ablation study")    
        
        #EFT Option
        train.add_argument('--lr_eft', type=float, default=5e-6, help='Learning rate for EFT') 
        train.add_argument('--eft_thresh_keyptErr_2d', type=float, default=1e-4, help='2D keypoint error threshold to stop EFT in DB geneneration') 
        train.add_argument('--eft_thresh_keyptErr_2d_testtime', type=float, default=2e-4, help='2D keypoint error threshold to stop EFT in testing time') 
        train.add_argument('--eft_withHip2D', default=False, action="store_true", help='If set, use hip 2d keypoint for EFT. Default False') 

        #EFT Debug options
        train.add_argument('--bDebug_visEFT', default=False, action='store_true', help='If true, show EFT process visualization') 

        shuffle_train = train.add_mutually_exclusive_group()
        shuffle_train.add_argument('--shuffle_train', dest='shuffle_train', action='store_true', help='Shuffle training data')
        shuffle_train.add_argument('--no_shuffle_train', dest='shuffle_train', action='store_false', help='Don\'t shuffle training data')
        shuffle_train.set_defaults(shuffle_train=True)
        return 

    def parse_args(self, params=None) :
        """Parse input arguments."""
        if params is not None:
            self.args = self.parser.parse_args(params)
        else:
            self.args = self.parser.parse_args()
        # If config file is passed, override all arguments with the values from the config file
        if self.args.from_json is not None:
            path_to_json = os.path.abspath(self.args.from_json)
            with open(path_to_json, "r") as f:
                json_args = json.load(f)

                #The following should now overwrite
                if self.args.resume:
                    json_args['resume'] = self.args.resume
                    json_args['num_epochs'] = self.args.num_epochs

                    for k in self.args.__dict__:
                        if k not in json_args.keys():
                            json_args[k] = self.args.__dict__[k]

                #Make it as namedtuple
                json_args = namedtuple("json_args", json_args.keys())(**json_args)
                return json_args
        else:
            now = datetime.datetime.now()
            newName = '{:02d}-{:02d}-{}-{}'.format(now.month, now.day, now.hour*3600 + now.minute*60 + now.second,self.args.name)
            # newName = now.strftime("%m-%d-%H:%M") + '-'+ self.args.name
            print(">>> Set logDir: {}".format(newName))
            # self.args.log_dir = os.path.join(os.path.abspath(self.args.log_dir), self.args.name)
            self.args.log_dir = os.path.join(os.path.abspath(self.args.log_dir), newName)
            self.args.log_dir = self.args.log_dir + '-' + str(int(np.random.rand()*10000))
            if os.path.exists(self.args.log_dir):
                self.args.log_dir = self.args.log_dir + '-' + str(int(np.random.rand()*10))

            self.args.summary_dir = os.path.join(self.args.log_dir, 'tensorboard')
            if not os.path.exists(self.args.log_dir):
                os.makedirs(self.args.log_dir)
            self.args.checkpoint_dir = os.path.join(self.args.log_dir, 'checkpoints')
            if not os.path.exists(self.args.checkpoint_dir):
                os.makedirs(self.args.checkpoint_dir)
            self.save_dump()
            return self.args

    def save_dump(self):
        """Store all argument values to a json file.
        The default location is logs/expname/config.json.
        """
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        with open(os.path.join(self.args.log_dir, "config.json"), "w") as f:
            json.dump(vars(self.args), f, indent=4)
        
        # with open(os.path.join(self.args.log_dir, "dbinfo.json"), "w") as f:
        #     json.dump(vars(config.DATASET_FILES), f, indent=4)
        
        return
