# Original code from SPIN: https://github.com/nkolot/SPIN

import torch
import torch.nn as nn
import numpy as np
from torchgeometry import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
import cv2

from bodymocap.datasets import MixedDataset, BaseDataset
from bodymocap.models import hmr, SMPL, SMPLX


from bodymocap.smplify import SMPLify
from bodymocap.utils.geometry import batch_rodrigues, perspective_projection, estimate_translation
from bodymocap.core import BaseTrainer

from bodymocap.core import config
from bodymocap.core import constants
from .fits_dict import FitsDict

from renderer import viewer2D
from renderer import glViewer


from bodymocap.apps.eval import run_evaluation #For test
from bodymocap.utils.timer import Timer
g_timer = Timer()

from bodymocap.utils import CheckpointDataLoader, CheckpointSaver
from bodymocap.utils.imutils import convert_smpl_to_bbox, convert_bbox_to_oriIm, conv_bboxinfo_center2topleft, deNormalizeBatchImg
from bodymocap.utils.geometry import weakProjection_gpu

from bodymocap.train import Trainer, normalize_2dvector


import bodymocap.utils.smpl_utils as smpl_utils#import visSMPLoutput, getSMPLoutput, getSMPLoutput_imgspace


from tqdm import tqdm

import os
import datetime
import pickle

from bodymocap.utils.pose_utils import reconstruction_error, reconstruction_error_fromMesh

class EFTFitter(Trainer):
    def init_fn(self):
        self.train_ds = MixedDataset(self.options, ignore_3d=self.options.ignore_3d, is_train=True)

        self.model = hmr(config.SMPL_MEAN_PARAMS, pretrained=True).to(self.device)

        if self.options.bExemplarMode:
            # lr = 1e-5   #5e-5 * 0.2       #original
            lr = self.options.lr_eft# 5e-6       #New EFT
        else:
            lr = self.options.lr
            
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                        #   lr=self.options.lr,
                                            lr =lr,
                                          weight_decay=0)

        if self.options.bUseSMPLX:      #SMPL-X model           #No change is required for HMR training. SMPL-X ignores hand and other parts.
                                                                #SMPL uses 23 joints, while SMPL-X uses 21 joints, automatically ignoring the last two joints of SMPL 
            self.smpl = SMPLX(config.SMPL_MODEL_DIR,        
                            batch_size=self.options.batch_size,
                            create_transl=False).to(self.device)
        else:       #Original SMPL
            self.smpl = SMPL(config.SMPL_MODEL_DIR,
                            batch_size=self.options.batch_size,
                            create_transl=False).to(self.device)

        if True:
            self.smpl_male = SMPL(config.SMPL_MODEL_DIR,
                        gender='male',
                        create_transl=False).to(self.device)
            self.smpl_female = SMPL(config.SMPL_MODEL_DIR,
                            gender='female',
                            create_transl=False).to(self.device)
            

        # Per-vertex loss on the shape
        self.criterion_shape = nn.L1Loss().to(self.device)
        # Keypoint (2D and 3D) loss
        # No reduction because confidence weighting needs to be applied
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        # Loss for SMPL parameter regression
        self.criterion_regr = nn.MSELoss().to(self.device)
        self.models_dict = {'model': self.model}
        self.optimizers_dict = {'optimizer': self.optimizer}
        self.focal_length = constants.FOCAL_LENGTH

        # Initialize SMPLify fitting module
        self.smplify = SMPLify(step_size=1e-2, batch_size=self.options.batch_size, num_iters=self.options.num_smplify_iters, focal_length=self.focal_length)
        
        if self.options.pretrained_checkpoint is not None:
            print(">>> Load Pretrained mode: {}".format(self.options.pretrained_checkpoint))
            self.load_pretrained(checkpoint_file=self.options.pretrained_checkpoint)
            self.backupModel()

        #This should be called here after loading model
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = torch.nn.DataParallel(self.model)      #Failed... 


        # Load dictionary of fits
        self.fits_dict = FitsDict(self.options, self.train_ds)

        # Create renderer
        self.renderer = None# Renderer(focal_length=self.focal_length, img_res=self.options.img_res, faces=self.smpl.faces)

        #debug
        from torchvision.transforms import Normalize
        self.de_normalize_img = Normalize(mean=[ -constants.IMG_NORM_MEAN[0]/constants.IMG_NORM_STD[0]    , -constants.IMG_NORM_MEAN[1]/constants.IMG_NORM_STD[1], -constants.IMG_NORM_MEAN[2]/constants.IMG_NORM_STD[2]], std=[1/constants.IMG_NORM_STD[0], 1/constants.IMG_NORM_STD[1], 1/constants.IMG_NORM_STD[2]])


    #Use openpose hip definition
    def keypoint_3d_loss_panopticDB(self, pred_keypoints_3d_49, gt_keypoints_3d, has_pose_3d):
        """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence.
        """
        pred_keypoints_3d = pred_keypoints_3d_49[:, 25:, :]
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
        conf = conf[has_pose_3d == 1]           #N, 24, 1

        #disable hips
        conf[:,2,:] = 0
        conf[:,3,:] = 0
        # conf[:,:6,:] = 0
        # conf[:,:6,:] = 0

        
        pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]         #N,24, 1
        if len(gt_keypoints_3d) > 0:
            # gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
            gt_nose = gt_keypoints_3d[:, 19,:]
            gt_keypoints_3d = gt_keypoints_3d - gt_nose[:, None, :]
            # pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
            pred_nose = pred_keypoints_3d[:, 19,:]# + pred_keypoints_3d[:, 3,:]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_nose[:, None, :]
            return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)
    


    #Use openpose hip definition
    def keypoint_3d_hand_loss_panopticDB(self, pred_right_hand_joints_3d, pred_left_hand_joints_3d, gt_lhand_3d, gt_rhand_3d):
        """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence.
        """

        loss = torch.tensor(0).to(self.device)
        for lr_pred, lr_gt in [ [pred_right_hand_joints_3d, gt_rhand_3d] , [pred_left_hand_joints_3d, gt_lhand_3d]  ]:
            conf = lr_gt[:, :, -1].unsqueeze(-1).clone()

            #use only nuckles
            conf[:, [0,2,3,4, 6,7,8, 10,11,12, 14, 15,16, 18, 19,20] ,:] = 0

            lr_gt = lr_gt[:, :, :-1].clone()

        
            # gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
            gt_origin = lr_gt[:, [0],:]
            lr_gt = lr_gt - gt_origin
            # pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
            pred_origin = lr_pred[:, [0],:]# + pred_keypoints_3d[:, 3,:]) / 2
            lr_pred = lr_pred - pred_origin
            loss  = loss + (conf * self.criterion_keypoints(lr_pred, lr_gt)).mean()
    
        return loss * 5.0     



    def exemplerTrainingMode(self):

        for module in self.model.modules():
            if type(module)==False:
                continue

            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                # print(module)
                module.eval()
                for m in module.parameters():
                    m.requires_grad =False
            if isinstance(module, nn.Dropout):
                # print(module)
                module.eval()
                for m in module.parameters():
                    m.requires_grad =False



    #Given a batch, run HMR training
    #Assumes a single sample in the batch, requiring batch norm disabled
    #Loss is a bit different from original HMR training
    def run_eft_step(self, input_batch, iterIdx=0):

        self.model.train()

        if self.options.bExemplarMode:
            self.exemplerTrainingMode()

        # Get data from the batch
        images = input_batch['img'] # input image
        gt_keypoints_2d = input_batch['keypoints'].clone()# 2D keypoints           #[N,49,3]
        gt_pose = input_batch['pose'] # SMPL pose parameters                #[N,72]
        gt_betas = input_batch['betas'] # SMPL beta parameters              #[N,10]

        
        JOINT_SCALING_3D = 1.0 #3D joint scaling 
        gt_joints = input_batch['pose_3d']*JOINT_SCALING_3D # 3D pose                        #[N,24,4]
        has_pose_3d = input_batch['has_pose_3d'].byte()==1 # flag that indicates whether 3D pose is valid
        indices = input_batch['sample_index'] # index of example inside its dataset
        batch_size = images.shape[0]


        is_flipped = input_batch['is_flipped'] # flag that indicates whether image was flipped during data augmentation
        rot_angle = input_batch['rot_angle'] # rotation angle used for data augmentation
        dataset_name = input_batch['dataset_name'] # name of the dataset the image comes from

        # if True:        #Hands
        #     gt_lhand_3d = input_batch['lhand_3d']
        #     gt_rhand_3d = input_batch['rhand_3d']

        # Get GT vertices and model joints
        # Note that gt_model_joints is different from gt_joints as it comes from SMPL
        # gt_out = self.smpl(betas=gt_betas, body_pose=gt_pose[:,3:], global_orient=gt_pose[:,:3])
        # gt_model_joints = gt_out.joints             #[N, 49, 3]     #Note this is different from gt_joints!
        # gt_vertices = gt_out.vertices

        # Get current best fits from the dictionary
        
        index_cpu = indices.cpu()
        if self.options.bExemplar_dataLoaderStart>=0:
            index_cpu +=self.options.bExemplar_dataLoaderStart      #Bug fixed.

        #Check existing SPIN fits
        opt_pose, opt_betas, opt_validity = self.fits_dict[(dataset_name, index_cpu, rot_angle.cpu(), is_flipped.cpu())]
        opt_pose = opt_pose.to(self.device)
        opt_betas = opt_betas.to(self.device)
        

        # if self.options.bUseSMPLX: #TODO: why we need this?
        #     opt_joints = opt_output.joints.detach()

        # if self.options.bExemplarWith3DSkel:
        #     gt_joints[:,2,3]=0  #Ignore Hips
        #     gt_joints[:,3,3]=0  #Ignore Hips
            
        #assuer that non valid opt has GT values
        # if len(has_smpl[opt_validity==0])>0:
        #     assert min(has_smpl[opt_validity==0])  #All should be True

        # else:       #Assue 3D DB!
        #     opt_pose = gt_pose
        #     opt_betas = gt_betas
        #     opt_vertices = gt_vertices
        #     opt_joints = gt_model_joints

        # De-normalize 2D keypoints from [-1,1] to pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()                     
        gt_keypoints_2d_orig[:, :, :-1] = 0.5 * self.options.img_res * (gt_keypoints_2d_orig[:, :, :-1] + 1)        #49: (25+24) x 3 

        # if gt_keypoints_2d[0,20,-1]>0:        #Checking foot keypoint
        #     print(gt_keypoints_2d_orig[0,19:25])

        # Feed images in the network to predict camera and SMPL parameters
        pred_rotmat, pred_betas, pred_camera = self.model(images)

        pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
        pred_joints_3d = pred_output.joints

        # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
        # This camera translation can be used in a full perspective projection
        pred_cam_t = torch.stack([pred_camera[:,1],
                                  pred_camera[:,2],
                                  2*self.focal_length/(self.options.img_res * pred_camera[:,0] +1e-9)],dim=-1)

        camera_center = torch.zeros(batch_size, 2, device=self.device)
      
        # weakProjection_gpu################
        pred_keypoints_2d = weakProjection_gpu(pred_joints_3d, pred_camera[:,0], pred_camera[:,1:] )           #N, 49, 2

        if True:    #Ignore hips and hip centers, foot
            LENGTH_THRESHOLD = 0.0089 #1/112.0     #at least it should be 5 pixel

            #Disable Hips by default
            if self.options.eft_withHip2D==False:     
                gt_keypoints_2d[:,2+25,2]=0
                gt_keypoints_2d[:,3+25,2]=0
                gt_keypoints_2d[:,14+25,2]=0

            # #Compute angle knee to ankle orientation
            gt_boneOri_leftLeg = gt_keypoints_2d[:,5+25,:2]  -  gt_keypoints_2d[:,4+25,:2]             #Left lower leg orientation     #(N,2)
            gt_boneOri_leftLeg, leftLegLeng = normalize_2dvector(gt_boneOri_leftLeg)

            if leftLegLeng>LENGTH_THRESHOLD:
                leftLegValidity = gt_keypoints_2d[:,5+25, 2]  * gt_keypoints_2d[:,4+25, 2]
                pred_boneOri_leftLeg = pred_keypoints_2d[:,5+25,:2]  -  pred_keypoints_2d[:,4+25,:2]
                pred_boneOri_leftLeg, _ = normalize_2dvector(pred_boneOri_leftLeg)
                loss_legOri_left  = torch.ones(1).to(self.device) - torch.dot(gt_boneOri_leftLeg.view(-1),pred_boneOri_leftLeg.view(-1))
            else:
                loss_legOri_left = torch.zeros(1).to(self.device)
                leftLegValidity  = torch.zeros(1).to(self.device)

            gt_boneOri_rightLeg = gt_keypoints_2d[:,0+25,:2] -  gt_keypoints_2d[:,1+25,:2]            #Right lower leg orientation
            gt_boneOri_rightLeg, rightLegLeng = normalize_2dvector(gt_boneOri_rightLeg)
            if rightLegLeng>LENGTH_THRESHOLD:

                rightLegValidity = gt_keypoints_2d[:,0+25, 2]  * gt_keypoints_2d[:,1+25, 2]
                pred_boneOri_rightLeg = pred_keypoints_2d[:,0+25,:2]  -  pred_keypoints_2d[:,1+25,:2]
                pred_boneOri_rightLeg, _ = normalize_2dvector(pred_boneOri_rightLeg)
                loss_legOri_right  = torch.ones(1).to(self.device) - torch.dot(gt_boneOri_rightLeg.view(-1),pred_boneOri_rightLeg.view(-1))
            else:
                loss_legOri_right = torch.zeros(1).to(self.device)
                rightLegValidity = torch.zeros(1).to(self.device)
            # print("leftLegLeng: {}, rightLegLeng{}".format(leftLegLeng,rightLegLeng ))
            loss_legOri = leftLegValidity* loss_legOri_left + rightLegValidity* loss_legOri_right

            # loss_legOri = torch.zeros(1).to(self.device)
            # if leftLegValidity.item():
            #     loss_legOri = loss_legOri + (pred_boneOri_leftLeg).mean()
            # if rightLegValidity.item():
            #     loss_legOri = loss_legOri + self.criterion_regr(gt_boneOri_rightLeg, pred_boneOri_rightLeg)
            # print(loss_legOri)

            #Disable Foots
            gt_keypoints_2d[:,5+25,2]=0     #Left foot
            gt_keypoints_2d[:,0+25,2]=0     #Right foot

        # Compute 2D reprojection loss for the keypoints
        loss_keypoints_2d = self.keypoint_loss(pred_keypoints_2d, gt_keypoints_2d,
                                            self.options.openpose_train_weight,
                                            self.options.gt_train_weight)

        # # Compute 3D keypoint loss
        if self.options.bExemplarWith3DSkel:
            # loss_keypoints_3d = self.keypoint_3d_loss(pred_joints_3d, gt_joints, has_pose_3d)
            loss_keypoints_3d = self.keypoint_3d_loss_panopticDB(pred_joints_3d, gt_joints, has_pose_3d)
        else:
            loss_keypoints_3d = torch.tensor(0)
       
        # loss_keypoints_3d = self.keypoint_3d_loss_modelSkel(pred_joints_3d, gt_model_joints[:,25:,:], has_pose_3d)

        loss_regr_betas_noReject = torch.mean(pred_betas**2)

        #Prevent bending knee?
        # red_rotmat[0,6,:,:] - 

        loss = self.options.keypoint_loss_weight * loss_keypoints_2d  + \
                self.options.beta_loss_weight * loss_regr_betas_noReject  + \
                ((torch.exp(-pred_camera[:,0]*10)) ** 2 ).mean() 
        
        if self.options.bExemplarWith3DSkel:
            loss = loss + self.options.keypoint_loss_weight * loss_keypoints_3d
            # loss = loss_keypoints_3d        #TODO: DEBUGGIN

        if True:        #Leg orientation loss
            loss = loss + 0.005*loss_legOri
        #Put zeor preference on knees
        #Disabled. Not working
        # if self.options.bUseKneePrior:
        #     eyemat = torch.eye(3,3).repeat(pred_rotmat.shape[0],1).view((-1,3,3)).to(self.device)
        #     kneePrior = ((pred_rotmat[:,5,:,:]  - eyemat)**2).mean() + ((pred_rotmat[:,4,:,:]  - eyemat)**2).mean()
        #     loss = loss + kneePrior*0.001

        # print(loss_regr_betas)
        loss *= 60
        # print("loss2D: {}, loss3D: {}".format( self.options.keypoint_loss_weight * loss_keypoints_2d,self.options.keypoint_loss_weight * loss_keypoints_3d  )  )

        # Do backprop
        self.optimizer.zero_grad()
        loss.backward()

        # g_timer.tic()
        self.optimizer.step()
        # g_timer.toc(bPrint =True)

        # Pack output arguments for tensorboard logging
        output = {'pred_vertices': 0, #pred_vertices.detach(),
                  'opt_vertices': 0,
                  'pred_cam_t': 0,#pred_cam_t.detach(),
                  'opt_cam_t': 0}

        #Save result
        output={}
        output['pred_pose_rotmat'] = pred_rotmat.detach().cpu().numpy()
        output['pred_shape'] = pred_betas.detach().cpu().numpy()
        output['pred_camera'] = pred_camera.detach().cpu().numpy()
        
        #If there exists SPIN fits, save that for comparison later
        output['opt_pose'] = opt_pose.detach().cpu().numpy()
        output['opt_beta'] = opt_betas.detach().cpu().numpy()
        
        output['sampleIdx'] = input_batch['sample_index'].detach().cpu().numpy()     #To use loader directly
        output['imageName'] = input_batch['imgname']
        output['scale'] = input_batch['scale'] .detach().cpu().numpy()
        output['center'] = input_batch['center'].detach().cpu().numpy()

        if 'annotId' in input_batch.keys():
            output['annotId'] = input_batch['annotId'].detach().cpu().numpy()

        if 'subjectId' in input_batch.keys():
            if input_batch['subjectId'][0]!="":
                output['subjectId'] = input_batch['subjectId'][0].item()
            

        #To save new db file
        output['keypoint2d'] = input_batch['keypoints_original'].detach().cpu().numpy()
        output['keypoint2d_cropped'] = input_batch['keypoints'].detach().cpu().numpy()


        losses = {'loss': loss.detach().item(),
                  'loss_keypoints': loss_keypoints_2d.detach().item(),
                  'loss_keypoints_3d': loss_keypoints_3d.detach().item(),
                #   'loss_regr_pose': loss_regr_pose.detach().item(),
                  'loss_regr_betas': loss_regr_betas_noReject.detach().item()}
                #   'loss_shape': loss_shape.detach().item()}


        if self.options.bDebug_visEFT:#g_debugVisualize:    #Debug Visualize input
            
            # print("Image Name: {}".format(output['imageName']))
            for b in range(batch_size):

                # DenormalizeImg
                curImgVis = deNormalizeBatchImg(images[b].cpu())
                viewer2D.ImShow(curImgVis, name='rawIm', scale=4.0)

                # Visualize GT 2D keypoints
                if True:
                    gt_keypoints_2d_orig_vis = gt_keypoints_2d_orig.detach().cpu().numpy()
                    gt_keypoints_2d_orig_vis[b,:25,2] = 0       #Don't show openpose
                    curImgVis = viewer2D.Vis_Skeleton_2D_SPIN49(gt_keypoints_2d_orig_vis[b,:,:2], gt_keypoints_2d_orig_vis[b,:,2], bVis= False, image=curImgVis)
                # curImgVis = viewer2D.Vis_Skeleton_2D_Openpose18(gt_keypoints_2d_orig[b,:,:2].cpu().numpy(), gt_keypoints_2d_orig[b,:,2], bVis= False, image=curImgVis)

                ############### Visualize Mesh #################
                #Visualize SMPL in image space
                pred_smpl_output, pred_smpl_output_bbox  = smpl_utils.visSMPLoutput_bboxSpace(self.smpl, {"pred_rotmat":pred_rotmat, "pred_shape":pred_betas, "pred_camera":pred_camera}
                                                    , image = curImgVis, waittime=-1)

            
                #Visualize GT Mesh
                if False:
                    gtOut = {"pred_pose":gt_pose, "pred_shape":gt_betas, "pred_camera":pred_camera}
                    # _, gt_smpl_output_bbox = smpl_utils.getSMPLoutput_bboxSpace(self.smpl, gtOut)
                    _, gt_smpl_output_bbox = smpl_utils.getSMPLoutput_bboxSpace(self.smpl_male, gtOut)          #Assuming Male model
                    gt_smpl_output_bbox['body_mesh']['color']  = glViewer.g_colorSet['hand']
                    glViewer.addMeshData( [gt_smpl_output_bbox['body_mesh']], bComputeNormal=True)

                ############### Visualize Skeletons ############### 
                glViewer.setSkeleton( [pred_smpl_output_bbox['body_joints_vis'] ])

                if False:
                    glViewer.addSkeleton( [gt_smpl_output_bbox['body_joints_vis'] ], colorRGB= glViewer.g_colorSet['hand'] )

                if True:
                    glViewer.show(1)
                elif False:   #Render to Files in original image space
            
                    #Get Skeletons
                    img_original = cv2.imread(input_batch['imgname'][0])
                    # viewer2D.ImShow(img_original, waitTime=0)
                    bboxCenter =  input_batch['center'].detach().cpu()[0]
                    bboxScale = input_batch['scale'].detach().cpu()[0]
                    imgShape = img_original.shape[:2]
                    smpl_output, smpl_output_bbox, smpl_output_imgspace  = smpl_utils.getSMPLoutput_imgSpace(self.smpl, {"pred_rotmat":pred_rotmat, "pred_shape":pred_betas, "pred_camera":pred_camera},
                                                                    bboxCenter, bboxScale, imgShape)

                    glViewer.setBackgroundTexture(img_original)       #Vis raw video as background
                    glViewer.setWindowSize(img_original.shape[1]*2, img_original.shape[0]*2)       #Vis raw video as background
                    glViewer.setMeshData([smpl_output_imgspace['body_mesh']], bComputeNormal = True )       #Vis raw video as background
                    glViewer.setSkeleton([])
                
                    imgname = os.path.basename(input_batch['imgname'][0])[:-4]
                    fileName = "{0}_{1}_{2:04d}".format(dataset_name[0],  imgname, iterIdx)

                    # rawImg = cv2.putText(rawImg,data['subjectId'],(100,100), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0),2)
                    glViewer.render_on_image('/home/hjoo/temp/render_eft', fileName, img_original, scaleFactor=2)

                else:
                    #Render
                    if True:
                        imgname = output['imageName'][b]
                        root_imgname = os.path.basename(imgname)[:-4]
                        renderRoot=f'/home/hjoo/temp/render_eft/eft_{root_imgname}'
                        imgname='{:04d}'.format(iterIdx)

                        # smpl_utils.renderSMPLoutput(renderRoot,'overlaid','raw',imgname=imgname)
                        smpl_utils.renderSMPLoutput(renderRoot,'overlaid','mesh',imgname=imgname)
                        smpl_utils.renderSMPLoutput(renderRoot,'overlaid','skeleton',imgname=imgname)
                        smpl_utils.renderSMPLoutput(renderRoot,'side','mesh',imgname=imgname)




                # # Show projection of SMPL sksleton
                # if False:
                #     pred_keypoints_2d_vis = pred_keypoints_2d[b,:,:2].detach().cpu().numpy()
                #     pred_keypoints_2d_vis = 0.5 * self.options.img_res * (pred_keypoints_2d_vis + 1)        #49: (25+24) x 3 
                #     if glViewer.g_bShowSkeleton:
                #         curImgVis = viewer2D.Vis_Skeleton_2D_general(pred_keypoints_2d_vis, bVis= False, image=curImgVis)
                # viewer2D.ImShow(curImgVis, scale=2.0, waitTime=1)


        bCompute3DError = True
        if bCompute3DError:
            pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices

            gt_output = self.smpl_male(betas=gt_betas, body_pose=gt_pose[:,3:], global_orient=gt_pose[:,:3])
            gt_vertices = gt_output.vertices
            # Reconstuction_error
            J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()        #17,6890
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).cuda()
            joint_mapper_h36m = constants.H36M_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.H36M_TO_J14

            r_error = reconstruction_error_fromMesh(J_regressor_batch, joint_mapper_h36m, pred_vertices, gt_vertices)

            # print("r_error:{}".format(r_error[0]*1000) )

            losses['r_error'] = r_error[0]*1000
        else:
            losses['r_error'] = 0

        return output, losses

       # #For all sample in the current trainingDB
  
    #Given a batch, run HMR training
    #Assumes a single sample in the batch, requiring batch norm disabled
    #Loss is a bit different from original HMR training
    def run_eft_step_wHand(self, input_batch):

        self.model.train()

        if self.options.bExemplarMode:
            self.exemplerTrainingMode()

        # Get data from the batch
        images = input_batch['img'] # input image
        gt_keypoints_2d = input_batch['keypoints'].clone()# 2D keypoints           #[N,49,3]
        gt_pose = input_batch['pose'] # SMPL pose parameters                #[N,72]
        gt_betas = input_batch['betas'] # SMPL beta parameters              #[N,10]


        #3D joint scaling 
        JOINT_SCALING_3D = 1.0
        gt_joints = input_batch['pose_3d']*JOINT_SCALING_3D # 3D pose                        #[N,24,4]
        # has_smpl = input_batch['has_smpl'].byte() ==1 # flag that indicates whether SMPL parameters are valid
        has_pose_3d = input_batch['has_pose_3d'].byte()==1 # flag that indicates whether 3D pose is valid
        indices = input_batch['sample_index'] # index of example inside its dataset
        batch_size = images.shape[0]


        is_flipped = input_batch['is_flipped'] # flag that indicates whether image was flipped during data augmentation
        rot_angle = input_batch['rot_angle'] # rotation angle used for data augmentation
        dataset_name = input_batch['dataset_name'] # name of the dataset the image comes from


        if self.options.bUseHand3D:        #Hands
            gt_lhand_3d = input_batch['lhand_3d']
            gt_rhand_3d = input_batch['rhand_3d']

        # Get GT vertices and model joints
        # Note that gt_model_joints is different from gt_joints as it comes from SMPL
        gt_out = self.smpl(betas=gt_betas, body_pose=gt_pose[:,3:], global_orient=gt_pose[:,:3])
        gt_model_joints = gt_out.joints             #[N, 49, 3]     #Note this is different from gt_joints!
        gt_vertices = gt_out.vertices

        # Get current best fits from the dictionary
        
        index_cpu = indices.cpu()
        if self.options.bExemplar_dataLoaderStart>=0:
            index_cpu +=self.options.bExemplar_dataLoaderStart      #Bug fixed.

        #Check existing SPIN fits
        opt_pose, opt_betas, opt_validity = self.fits_dict[(dataset_name, index_cpu, rot_angle.cpu(), is_flipped.cpu())]
        opt_pose = opt_pose.to(self.device)
        opt_betas = opt_betas.to(self.device)
      

        # De-normalize 2D keypoints from [-1,1] to pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()                     
        gt_keypoints_2d_orig[:, :, :-1] = 0.5 * self.options.img_res * (gt_keypoints_2d_orig[:, :, :-1] + 1)        #49: (25+24) x 3 

        # if gt_keypoints_2d[0,20,-1]>0:        #Checking foot keypoint
        #     print(gt_keypoints_2d_orig[0,19:25])

        # Feed images in the network to predict camera and SMPL parameters
        pred_rotmat, pred_betas, pred_camera = self.model(images)

        pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
        pred_joints_3d = pred_output.joints

        if self.options.bUseSMPLX:
            pred_right_hand_joints_3d = pred_output.right_hand_joints
            pred_left_hand_joints_3d = pred_output.left_hand_joints

        # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
        # This camera translation can be used in a full perspective projection
        pred_cam_t = torch.stack([pred_camera[:,1],
                                  pred_camera[:,2],
                                  2*self.focal_length/(self.options.img_res * pred_camera[:,0] +1e-9)],dim=-1)

        camera_center = torch.zeros(batch_size, 2, device=self.device)
      
        # weakProjection_gpu################
        pred_keypoints_2d = weakProjection_gpu(pred_joints_3d, pred_camera[:,0], pred_camera[:,1:] )           #N, 49, 2

        if True:    #Ignore hips and hip centers, foot
            LENGTH_THRESHOLD = 0.0089 #1/112.0     #at least it should be 5 pixel

            #Disable parts
            if True:
                gt_keypoints_2d[:,2+25,2]=0
                gt_keypoints_2d[:,3+25,2]=0
                gt_keypoints_2d[:,14+25,2]=0

            # #Compute angle knee to ankle orientation
            gt_boneOri_leftLeg = gt_keypoints_2d[:,5+25,:2]  -  gt_keypoints_2d[:,4+25,:2]             #Left lower leg orientation     #(N,2)
            gt_boneOri_leftLeg, leftLegLeng = normalize_2dvector(gt_boneOri_leftLeg)

            if leftLegLeng>LENGTH_THRESHOLD:
                leftLegValidity = gt_keypoints_2d[:,5+25, 2]  * gt_keypoints_2d[:,4+25, 2]
                pred_boneOri_leftLeg = pred_keypoints_2d[:,5+25,:2]  -  pred_keypoints_2d[:,4+25,:2]
                pred_boneOri_leftLeg, _ = normalize_2dvector(pred_boneOri_leftLeg)
                loss_legOri_left  = torch.ones(1).to(self.device) - torch.dot(gt_boneOri_leftLeg.view(-1),pred_boneOri_leftLeg.view(-1))
            else:
                loss_legOri_left = torch.zeros(1).to(self.device)
                leftLegValidity  = torch.zeros(1).to(self.device)

            gt_boneOri_rightLeg = gt_keypoints_2d[:,0+25,:2] -  gt_keypoints_2d[:,1+25,:2]            #Right lower leg orientation
            gt_boneOri_rightLeg, rightLegLeng = normalize_2dvector(gt_boneOri_rightLeg)
            if rightLegLeng>LENGTH_THRESHOLD:

                rightLegValidity = gt_keypoints_2d[:,0+25, 2]  * gt_keypoints_2d[:,1+25, 2]
                pred_boneOri_rightLeg = pred_keypoints_2d[:,0+25,:2]  -  pred_keypoints_2d[:,1+25,:2]
                pred_boneOri_rightLeg, _ = normalize_2dvector(pred_boneOri_rightLeg)
                loss_legOri_right  = torch.ones(1).to(self.device) - torch.dot(gt_boneOri_rightLeg.view(-1),pred_boneOri_rightLeg.view(-1))
            else:
                loss_legOri_right = torch.zeros(1).to(self.device)
                rightLegValidity = torch.zeros(1).to(self.device)
            # print("leftLegLeng: {}, rightLegLeng{}".format(leftLegLeng,rightLegLeng ))
            loss_legOri = leftLegValidity* loss_legOri_left + rightLegValidity* loss_legOri_right

            # loss_legOri = torch.zeros(1).to(self.device)
            # if leftLegValidity.item():
            #     loss_legOri = loss_legOri + (pred_boneOri_leftLeg).mean()
            # if rightLegValidity.item():
            #     loss_legOri = loss_legOri + self.criterion_regr(gt_boneOri_rightLeg, pred_boneOri_rightLeg)
            # print(loss_legOri)

            #Disable Foots
            # gt_keypoints_2d[:,5+25,2]=0     #Left foot
            # gt_keypoints_2d[:,0+25,2]=0     #Right foot

       
        # Compute 2D reprojection loss for the keypoints
        loss_keypoints_2d = self.keypoint_loss(pred_keypoints_2d, gt_keypoints_2d,
                                            self.options.openpose_train_weight,
                                            self.options.gt_train_weight)

        # # Compute 3D keypoint loss
        # loss_keypoints_3d = self.keypoint_3d_loss(pred_joints_3d, gt_joints, has_pose_3d)
        loss_keypoints_3d = self.keypoint_3d_loss_panopticDB(pred_joints_3d, gt_joints, has_pose_3d)
        
        # # loss_keypoints_3d = self.keypoint_3d_loss_modelSkel(pred_joints_3d, gt_model_joints[:,25:,:], has_pose_3d)

        loss_regr_betas_noReject = torch.mean(pred_betas**2)

        #Prevent bending knee?
        # red_rotmat[0,6,:,:] - 

        loss = self.options.keypoint_loss_weight * loss_keypoints_2d  + \
                self.options.beta_loss_weight * loss_regr_betas_noReject  + \
                ((torch.exp(-pred_camera[:,0]*10)) ** 2 ).mean() 
        
        if self.options.bExemplarWith3DSkel:
            loss = loss + self.options.keypoint_loss_weight * loss_keypoints_3d
            # loss = loss_keypoints_3d        #TODO: DEBUGGIN


        ##### Compute 3D hand joint loss especially for panoptic stuido
        if self.options.bUseHand3D:
            
            loss_keypoints_3d_hand = self.keypoint_3d_hand_loss_panopticDB(pred_right_hand_joints_3d, pred_left_hand_joints_3d, gt_lhand_3d, gt_rhand_3d)
            loss = loss + self.options.keypoint_loss_weight * loss_keypoints_3d_hand

        if True:        #Leg orientation loss
            loss = loss + 0.005*loss_legOri
        #Put zeor preference on knees
        #Disabled. Not working
        # if self.options.bUseKneePrior:
        #     eyemat = torch.eye(3,3).repeat(pred_rotmat.shape[0],1).view((-1,3,3)).to(self.device)
        #     kneePrior = ((pred_rotmat[:,5,:,:]  - eyemat)**2).mean() + ((pred_rotmat[:,4,:,:]  - eyemat)**2).mean()
        #     loss = loss + kneePrior*0.001

        # print(loss_regr_betas)
        loss *= 60
        # print("loss2D: {}, loss3D: {}".format( self.options.keypoint_loss_weight * loss_keypoints_2d,self.options.keypoint_loss_weight * loss_keypoints_3d  )  )

        # Do backprop
        self.optimizer.zero_grad()
        loss.backward()

        # g_timer.tic()
        self.optimizer.step()
        # g_timer.toc(bPrint =True)

        # Pack output arguments for tensorboard logging
        output = {'pred_vertices': 0, #pred_vertices.detach(),
                  'opt_vertices': 0,
                  'pred_cam_t': 0,#pred_cam_t.detach(),
                  'opt_cam_t': 0}

        #Save result
        output={}
        output['pred_pose_rotmat'] = pred_rotmat.detach().cpu().numpy()
        output['pred_shape'] = pred_betas.detach().cpu().numpy()
        output['pred_camera'] = pred_camera.detach().cpu().numpy()
        
        #If there exists SPIN fits, save that for comparison later
        output['opt_pose'] = opt_pose.detach().cpu().numpy()
        output['opt_beta'] = opt_betas.detach().cpu().numpy()
        
        output['sampleIdx'] = input_batch['sample_index'].detach().cpu().numpy()     #To use loader directly
        output['imageName'] = input_batch['imgname']
        output['scale'] = input_batch['scale'] .detach().cpu().numpy()
        output['center'] = input_batch['center'].detach().cpu().numpy()

        if 'annotId' in input_batch.keys():
            output['annotId'] = input_batch['annotId'].detach().cpu().numpy()

        if 'subjectId' in input_batch.keys():
            output['subjectId'] = input_batch['subjectId'][0].item()
            # print(output['subjectId'])

        #To save new db file
        output['keypoint2d'] = input_batch['keypoints_original'].detach().cpu().numpy()
        output['keypoint2d_cropped'] = input_batch['keypoints'].detach().cpu().numpy()


        #Save GT gt_lhand_3d, gt_rhand_3d
        if self.options.bUseHand3D:
            output['gt_lhand_3d'] = gt_lhand_3d.detach().cpu().numpy()
            output['gt_rhand_3d'] = gt_rhand_3d.detach().cpu().numpy()

        losses = {'loss': loss.detach().item(),
                  'loss_keypoints': loss_keypoints_2d.detach().item(),
                  'loss_keypoints_3d': loss_keypoints_3d.detach().item(),
                #   'loss_regr_pose': loss_regr_pose.detach().item(),
                  'loss_regr_betas': loss_regr_betas_noReject.detach().item()}
                #   'loss_shape': loss_shape.detach().item()}


        if self.options.bDebug_visEFT:#g_debugVisualize:    #Debug Visualize input
            
            # print("Image Name: {}".format(output['imageName']))
            for b in range(batch_size):
                #denormalizeImg
                curImgVis = images[b]     #3,224,224
                curImgVis = self.de_normalize_img(curImgVis).cpu().numpy()
                curImgVis = np.transpose( curImgVis , (1,2,0) )*255.0
                curImgVis =curImgVis[:,:,[2,1,0]] 


                #Denormalize image
                curImgVis = np.ascontiguousarray(curImgVis, dtype=np.uint8)
                originalImgVis = curImgVis.copy()
                viewer2D.ImShow(curImgVis, name='rawIm')

                gt_keypoints_2d_orig_vis = gt_keypoints_2d_orig.detach().cpu().numpy()
                curImgVis = viewer2D.Vis_Skeleton_2D_SPIN49(gt_keypoints_2d_orig_vis[b,:,:2], gt_keypoints_2d_orig_vis[b,:,2], bVis= False, image=curImgVis)
                # curImgVis = viewer2D.Vis_Skeleton_2D_Openpose18(gt_keypoints_2d_orig[b,:,:2].cpu().numpy(), gt_keypoints_2d_orig[b,:,2], bVis= False, image=curImgVis)

                
                # Show projection of SMPL sksleton
                if False:
                    pred_keypoints_2d_vis = pred_keypoints_2d[b,:,:2].detach().cpu().numpy()
                    pred_keypoints_2d_vis = 0.5 * self.options.img_res * (pred_keypoints_2d_vis + 1)        #49: (25+24) x 3 
                    if glViewer.g_bShowSkeleton:
                        curImgVis = viewer2D.Vis_Skeleton_2D_general(pred_keypoints_2d_vis, bVis= False, image=curImgVis)
                viewer2D.ImShow(curImgVis, scale=2.0, waitTime=1)


                #Get camera pred_params
                pred_camera_vis = pred_camera.detach().cpu().numpy()
               
                ############### Visualize Mesh ############### 
                pred_vert_vis = pred_vertices[b].detach().cpu().numpy() 
                camParam_scale = pred_camera_vis[b,0]
                camParam_trans = pred_camera_vis[b,1:]
                pred_vert_vis = convert_smpl_to_bbox(pred_vert_vis, camParam_scale, camParam_trans)
                pred_meshes = {'ver': pred_vert_vis, 'f': self.smpl.faces}


                # glViewer.setMeshData([pred_meshes, opt_meshes], bComputeNormal= True)
                glViewer.setMeshData([pred_meshes], bComputeNormal= True)
                # glViewer.setMeshData([pred_meshes, opt_meshes], bComputeNormal= True)
                # glViewer.setMeshData([opt_meshes], bComputeNormal= True)
                # glViewer.SetMeshColor('red')

                ############### Visualize Skeletons ############### 
                #Vis pred-SMPL joint
                pred_joints_vis = pred_joints_3d[b,:,:3].detach().cpu().numpy()  #[N,49,3]
                # pred_joints_pelvis = (pred_joints_vis[25+2,:] + pred_joints_vis[25+3,:]) / 2
                pred_joints_nose = pred_joints_vis[25+19,:]# + pred_joints_vis[25+3,:]) / 2

                pred_joints_vis = convert_smpl_to_bbox(pred_joints_vis, camParam_scale, camParam_trans)
                pred_joints_vis = pred_joints_vis.ravel()[:,np.newaxis]
                glViewer.setSkeleton( [pred_joints_vis], colorRGB=(0,0, 255), jointType='spin')


                if self.options.bUseSMPLX:
                    for pred_rl in [pred_right_hand_joints_3d, pred_left_hand_joints_3d]:
                        pred_joints_vis = pred_rl[b,:,:3].detach().cpu().numpy()  #[N,49,3]
                        pred_joints_vis = convert_smpl_to_bbox(pred_joints_vis, camParam_scale, camParam_trans)
                        pred_joints_vis = pred_joints_vis.ravel()[:,np.newaxis]
                        glViewer.addSkeleton( [pred_joints_vis], colorRGB=(0,0, 255), jointType='hand_panopticdb')  #GT: blue

                # #Vis GT  joint  (not model (SMPL) joint!!)
                if has_pose_3d[b] and self.options.bExemplarWith3DSkel:
                    # gt_jointsVis = gt_model_joints[b,:,:3].cpu().numpy()        #[N,49,3]
                    gt_jointsVis = gt_joints[b,:,:3].cpu().numpy()        #[N,49,3]
                    # gt_pelvis = (gt_jointsVis[ 2,:] + gt_jointsVis[ 3,:]) / 2

                    gt_jointsVis = convert_smpl_to_bbox(gt_jointsVis, camParam_scale, camParam_trans)
                    gt_nose = gt_jointsVis[ 19,:]
                    gt_jointsVis = gt_jointsVis- gt_nose + pred_joints_nose
                    
                    gt_jointsVis = gt_jointsVis.ravel()[:,np.newaxis]
                    # gt_jointsVis*=pred_camera_vis[b,0]
                    # gt_jointsVis[::3] += pred_camera_vis[b,1]
                    # gt_jointsVis[1::3] += pred_camera_vis[b,2]
                    # gt_jointsVis*=112

                    glViewer.addSkeleton( [gt_jointsVis], colorRGB=(255, 0, 0), jointType='spin')

                    #Vis hand
                    if self.options.bUseHand3D:
                        for gt_rl in [gt_lhand_3d, gt_rhand_3d]:
                            gt_3d = gt_rl[b,:,:3].cpu().numpy()        #[N,49,3]
                            gt_3d = convert_smpl_to_bbox(gt_3d, camParam_scale, camParam_trans)
                            gt_3d = gt_3d - gt_nose + pred_joints_nose
                            gt_3d = gt_3d.ravel()[:,np.newaxis]
                            glViewer.addSkeleton( [gt_3d], colorRGB=(255,0,0), jointType='hand_panopticdb')     #GT: red


                # # glViewer.show()

                glViewer.setBackgroundTexture(curImgVis)       #Vis raw video as background
                # glViewer.setBackgroundTexture(originalImgVis)       #Vis raw video as background
                glViewer.setWindowSize(curImgVis.shape[1]*3, curImgVis.shape[0]*3)
                glViewer.SetOrthoCamera(True)
                glViewer.show(1)
                # glViewer.show_SMPL(bSaveToFile = True, bResetSaveImgCnt = False, mode = 'camera')
                # glViewer.show_SMPL(bSaveToFile = True, bResetSaveImgCnt = False, mode = 'free')

                # continue

        return output, losses

       # #For all sample in the current trainingDB
  



    #Given a batch, run HMR training
    #Assumes a single sample in the batch, requiring batch norm disabled
    #Loss is a bit different from original HMR training
    def run_smplify(self, input_batch):

        self.model.eval()

        # Get data from the batch
        images = input_batch['img'] # input image
        gt_keypoints_2d = input_batch['keypoints'].clone()# 2D keypoints           #[N,49,3]
        # gt_pose = input_batch['pose'] # SMPL pose parameters                #[N,72]
        # gt_betas = input_batch['betas'] # SMPL beta parameters              #[N,10]

        #3D joint scaling 
        JOINT_SCALING_3D = 1.0
        # gt_joints = input_batch['pose_3d']*JOINT_SCALING_3D # 3D pose                        #[N,24,4]
        # has_smpl = input_batch['has_smpl'].byte() ==1 # flag that indicates whether SMPL parameters are valid
        # has_pose_3d = input_batch['has_pose_3d'].byte()==1 # flag that indicates whether 3D pose is valid
        indices = input_batch['sample_index'] # index of example inside its dataset
        batch_size = images.shape[0]

        is_flipped = input_batch['is_flipped'] # flag that indicates whether image was flipped during data augmentation
        rot_angle = input_batch['rot_angle'] # rotation angle used for data augmentation
        dataset_name = input_batch['dataset_name'] # name of the dataset the image comes from

        bboxCenter =  input_batch['center'].detach().cpu()[0]
        bboxScale = input_batch['scale'].detach().cpu()[0]
        imgname = input_batch['imgname'][0]

        # Get GT vertices and model joints
        # Note that gt_model_joints is different from gt_joints as it comes from SMPL
        # gt_out = self.smpl(betas=gt_betas, body_pose=gt_pose[:,3:], global_orient=gt_pose[:,:3])
        # gt_model_joints = gt_out.joints             #[N, 49, 3]     #Note this is different from gt_joints!
        # gt_vertices = gt_out.vertices

        # Get current best fits from the dictionary
        index_cpu = indices.cpu()
        if self.options.bExemplar_dataLoaderStart>=0:
            index_cpu +=self.options.bExemplar_dataLoaderStart      #Bug fixed.

        #Check existing SPIN fits
        opt_pose, opt_betas, opt_validity = self.fits_dict[(dataset_name, index_cpu, rot_angle.cpu(), is_flipped.cpu())]
        opt_pose = opt_pose.to(self.device)
        opt_betas = opt_betas.to(self.device)

        # De-normalize 2D keypoints from [-1,1] to pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()          #Weak perspective projection version           
        # gt_keypoints_2d_orig[:, :, :-1] = 0.5 * self.options.img_res * (gt_keypoints_2d_orig[:, :, :-1] + 1)        #49: (25+24) x 3 

        # Predict Initial Estimation via original model
        init_pred_rotmat, init_pred_betas, init_pred_camera = self.model(images)
        if self.options.bDebug_visEFT:
            smpl_utils.visSMPLoutput_bboxSpace(self.smpl, {"pred_rotmat":init_pred_rotmat, "pred_shape":init_pred_betas, "pred_camera":init_pred_camera }, image = images[0])

            # DenormalizeImg
            curImgVis = deNormalizeBatchImg(images[0].cpu())
            viewer2D.ImShow(curImgVis, name='rawIm',waitTime=1)

            # Visualize GT 2D keypoints
            gt_keypoints_2d_orig_vis = gt_keypoints_2d_orig.detach().cpu().numpy() 
            gt_keypoints_2d_orig_vis[:,:,:2] = (1.0+gt_keypoints_2d_orig_vis[:,:,:2])*112
            gt_keypoints_2d_orig_vis[:,:25,2] *=0
            curImgVis = viewer2D.Vis_Skeleton_2D_SPIN49(gt_keypoints_2d_orig_vis[0,:,:2], gt_keypoints_2d_orig_vis[0,:,2]>0.1, bVis= False, image=curImgVis)
            viewer2D.ImShow(curImgVis, name='skel', waitTime=1, scale=4)
        else: 
            curImgVis = None

        ####################################
        ### Run SMPLify
        # Convert predicted rotation matrices to axis-angle
        pred_rotmat_hom = torch.cat([init_pred_rotmat.detach().view(-1, 3, 3).detach(), torch.tensor([0,0,1], dtype=torch.float32,
            device=self.device).view(1, 3, 1).expand(batch_size * 24, -1, -1)], dim=-1)
        init_pred_pose = rotation_matrix_to_angle_axis(pred_rotmat_hom).contiguous().view(batch_size, -1)
        # tgm.rotation_matrix_to_angle_axis returns NaN for 0 rotation, so manually hack it
        init_pred_pose[torch.isnan(init_pred_pose)] = 0.0

        g_timer.tic()
        new_opt_vertices, new_opt_joints,\
        new_opt_pose, new_opt_betas,\
        new_opt_cam_t, new_reprjec_loss = self.smplify.run_withWeakProj(
                                    init_pred_pose.detach(), init_pred_betas.detach(),
                                    init_pred_camera.detach(),
                                    0.5 * self.options.img_res * torch.ones(batch_size, 2, device=self.device),
                                    gt_keypoints_2d_orig, 
                                    bDebugVis = self.options.bDebug_visEFT,
                                    bboxInfo={'bboxCenter':bboxCenter, 'bboxScale':bboxScale, 'imgname':imgname},
                                    imagevis = curImgVis,        #Image for visualization
                                    ablation_smplify_noCamOptFirst= self.options.ablation_smplify_noCamOptFirst,
                                    ablation_smplify_noPrior = self.options.ablation_smplify_noPrior
                                    )
        g_timer.toc(average =True, bPrint=True,title="SMPLify whole process")
        new_reprjec_loss = new_reprjec_loss.mean(dim=-1)


        #Save result
        output={}
        output['pred_pose_rotmat_init'] = init_pred_rotmat.detach().cpu().numpy()
        output['pred_shape_init'] = init_pred_betas.detach().cpu().numpy()
        output['pred_camera_init'] = init_pred_camera.detach().cpu().numpy()

        #Convert pose to rotmat
        new_opt_rot_mats = batch_rodrigues(new_opt_pose.view(-1, 3)).view([batch_size, -1, 3, 3])
        if self.options.bDebug_visEFT:
            print("Final")
            smpl_utils.visSMPLoutput_bboxSpace(self.smpl, {"pred_rotmat":new_opt_rot_mats, "pred_shape":new_opt_betas, "pred_camera":new_opt_cam_t }, image = images[0])
        
        output['pred_pose_rotmat'] = new_opt_rot_mats.cpu().numpy()
        output['pred_shape'] = new_opt_betas.detach().cpu().numpy()
        output['pred_camera'] = new_opt_cam_t.detach().cpu().numpy()

        
        
        #If there exists SPIN fits, save that for comparison later
        output['spin_pose'] = opt_pose.detach().cpu().numpy()
        output['spin_beta'] = opt_betas.detach().cpu().numpy()
        
        output['sampleIdx'] = input_batch['sample_index'].detach().cpu().numpy()     #To use loader directly
        output['imageName'] = input_batch['imgname']
        output['scale'] = input_batch['scale'] .detach().cpu().numpy()
        output['center'] = input_batch['center'].detach().cpu().numpy()

        if 'annotId' in input_batch.keys():
            output['annotId'] = input_batch['annotId'].detach().cpu().numpy()

        if 'subjectId' in input_batch.keys():
            if input_batch['subjectId'][0]!="":
                output['subjectId'] = input_batch['subjectId'][0].item()

        #To save new db file
        output['keypoint2d'] = input_batch['keypoints_original'].detach().cpu().numpy()
        output['keypoint2d_cropped'] = input_batch['keypoints'].detach().cpu().numpy()
        output['loss_keypoints_2d'] = new_reprjec_loss
        output['numOfIteration'] = self.smplify.num_iters
        
        return output

       # #For all sample in the current trainingDB
  

    #Run EFT
    #Save output as seperate pkl files
    def eftAllInDB(self, test_dataset_3dpw = None, test_dataset_h36m= None, bExportPKL = True):

        if config.bIsDevfair:
            now = datetime.datetime.now()
            # newName = '{:02d}-{:02d}-{}'.format(now.month, now.day, now.hour*3600 + now.minute*60 + now.second)
            newName = '{:02d}-{:02d}'.format(now.month, now.day)
            outputDir = newName + '_' + self.options.db_set + '_' + self.options.name
        else:
            now = datetime.datetime.now()
            # outputDir = self.options.db_set
            newName = 'test_{:02d}-{:02d}-{}'.format(now.month, now.day, now.hour*3600 + now.minute*60 + now.second)
            outputDir = newName + '_' + self.options.db_set + '_' + self.options.name

        exemplarOutputPath = os.path.join(config.EXEMPLAR_OUTPUT_ROOT , outputDir)
        if not os.path.exists(exemplarOutputPath):
            os.mkdir(exemplarOutputPath)

        """Training process."""
        # Run training for num_epochs epochs
        # Create new DataLoader every epoch and (possibly) resume from an arbitrary step inside an epoch
        train_data_loader = CheckpointDataLoader(self.train_ds,checkpoint=self.checkpoint,
                                                    batch_size=1,       #Always o1
                                                    num_workers=self.options.num_workers,
                                                    pin_memory=self.options.pin_memory,
                                                    shuffle=False)      #No Shuffle      
        
        maxExemplarIter = self.options.maxExemplarIter
       
        # Iterate over all batches in an epoch
        outputList ={}
        for step, batch in enumerate(tqdm(train_data_loader)):#, desc='Epoch '+str(epoch),
                                        #     total=len(self.train_ds) // self.options.batch_size,
                                        #     initial=train_data_loader.checkpoint_batch_idx),
                                        # train_data_loader.checkpoint_batch_idx):
            
            #3DPW test
            # if 'downtown_bus_00' not in batch['imgname']:
            #     continue

            #Only performed for 1/100 data (roughly hundred level)
            if self.options.bExemplar_analysis_testloss:
                sampleIdx = batch['sample_index'][0].item()
                if sampleIdx%100 !=0:
                    continue
            
            if self.options.bExemplar_badsample_finder:
                sampleIdx = batch['sample_index'][0].item()
                # if sampleIdx%100 !=0:
                #     continue

            bSkipExisting  =  self.options.bNotSkipExemplar==False     #bNotSkipExemplar ===True --> bSkipExisting==False
            if bSkipExisting:
                if self.options.db_set =='panoptic':
                    # fileNameOnly = os.path.basename(output['imageName'][0])[:-4]
                    # fileNameOnly = (batch['pkl_save_name'][0])[:-4].replace("/","-")

                    sampleIdx = batch['sample_index'][0].item()
                    if self.options.bExemplar_dataLoaderStart>=0:
                        sampleIdx +=self.options.bExemplar_dataLoaderStart

                    sampleIdxSaveFrame = 100* (int(sampleIdx/100.0) + 1)
                    
                    fileName = '{:08d}.pkl'.format(sampleIdxSaveFrame)

                    outputPath = os.path.join(exemplarOutputPath,fileName)
                    # print(">> checking: {}".format(outputPath))
                    if os.path.exists(outputPath):
                        print("Skipped: {}".format(outputPath))
                        continue
                elif '3dpw' in self.options.db_set:

                    fileNameOnly = os.path.basename(batch['imgname'][0])[:-4]
                    seqName = os.path.basename(os.path.dirname(batch['imgname'][0]))
                    fileNameOnly = f"{seqName}_{fileNameOnly}"

                    sampleIdx = batch['sample_index'][0].item()
                    if self.options.bExemplar_dataLoaderStart>=0:
                        sampleIdx +=self.options.bExemplar_dataLoaderStart
                    fileName = '{}_{}.pkl'.format(fileNameOnly,sampleIdx)
                    outputPath = os.path.join(exemplarOutputPath,fileName)
                    if os.path.exists(outputPath):
                        print("Skipped: {}".format(outputPath))
                        continue
                else:
                    fileNameOnly = os.path.basename(batch['imgname'][0])[:-4]
                    sampleIdx = batch['sample_index'][0].item()
                    if self.options.bExemplar_dataLoaderStart>=0:
                        sampleIdx +=self.options.bExemplar_dataLoaderStart
                    fileName = '{}_{}.pkl'.format(fileNameOnly,sampleIdx)
                    outputPath = os.path.join(exemplarOutputPath,fileName)
                    if os.path.exists(outputPath):
                        print("Skipped: {}".format(outputPath))
                        continue
                    
            g_timer.tic()
            self.reloadModel()  #For each sample


            #Freeze non resnet part model
            if self.options.ablation_layerteset_onlyLayer4:
                # self.model.conv1.requires_grad = False
                # self.model.bn1.requires_grad = False
                # self.model.relu.requires_grad = False
                # self.model.maxpool.requires_grad = False
                # self.model.layer1.requires_grad = False
                # self.model.layer2.requires_grad = False
                # self.model.layer3.requires_grad = False
                # self.model.layer4.requires_grad = False
                # self.model.fc1.requires_grad = False
                # self.model.drop1.requires_grad = False
                # self.model.fc2.requires_grad = False
                # self.model.drop2.requires_grad = False
                # self.model.decpose.requires_grad = False
                # self.model.decshape.requires_grad = False
                # self.model.deccam.requires_grad = False
                for par in self.model.parameters():
                    par.requires_grad = False

                for name, par in self.model.named_parameters():      #Optimize Layer 4 of resnet
                    # print(name)
                    # if 'fc' in name or 'decpose' in name or 'decshape' in name or 'deccam' in name:
                    #     print(f"activate {name}")
                    #     par.requires_grad = True
                    if 'layer4' in name:
                        # print(f">>  Activate {name}")
                        par.requires_grad = True

            if self.options.ablation_layerteset_onlyAfterRes:   #Optimize  HMR FC part
                for par in self.model.parameters():
                    par.requires_grad = False

                for name, par in self.model.named_parameters():
                    if 'fc' in name or 'decpose' in name or 'decshape' in name or 'deccam' in name:
                        # print(f"activate {name}")
                        par.requires_grad = True
            
            if self.options.ablation_layerteset_Layer4Later:        #Optimize Layer 4 of resent +  HMR FC part
                for par in self.model.parameters():
                    par.requires_grad = False

                for name, par in self.model.named_parameters():
                    if 'layer4' in name or 'fc' in name or 'decpose' in name or 'decshape' in name or 'deccam' in name:
                        # print(f"activate {name}")
                        par.requires_grad = True
            
            if self.options.ablation_layerteset_onlyRes:
                for par in self.model.parameters():
                    par.requires_grad = False

                for name, par in self.model.named_parameters():
                    if 'layer' in name:
                        # print(f"activate {name}")
                        par.requires_grad = True

            
            if self.options.ablation_layerteset_Layer3Later:
                for par in self.model.parameters():
                    par.requires_grad = False

                for name, par in self.model.named_parameters():
                    if 'layer3' in name or 'layer4' in name or 'fc' in name or 'decpose' in name or 'decshape' in name or 'deccam' in name:
                        # print(f"activate {name}")
                        par.requires_grad = True

            if self.options.ablation_layerteset_Layer2Later:
                for par in self.model.parameters():
                    par.requires_grad = False

                for name, par in self.model.named_parameters():
                    if 'layer2' in name or 'layer3' in name or 'layer4' in name or 'fc' in name or 'decpose' in name or 'decshape' in name or 'deccam' in name:
                        # print(f"activate {name}")
                        par.requires_grad = True

            if self.options.ablation_layerteset_Layer1Later:
                for par in self.model.parameters():
                    par.requires_grad = False

                for name, par in self.model.named_parameters():
                    if 'layer1' in name or 'layer2' in name or 'layer3' in name or 'layer4' in name or 'fc' in name or 'decpose' in name or 'decshape' in name or 'deccam' in name:
                        # print(f"activate {name}")
                        par.requires_grad = True

            if self.options.ablation_layerteset_all:    #No Freeze. debugging purpose
                for par in self.model.parameters():
                    par.requires_grad = False

                for name, par in self.model.named_parameters():
                    if 'conv1' in name or 'layer' in name or 'fc' in name or 'decpose' in name or 'decshape' in name or 'deccam' in name:
                        # print(f"activate {name}")
                        par.requires_grad = True

            if self.options.ablation_layerteset_onlyRes_withconv1:      #Only use ResNet. Freeze HMR part all
                for par in self.model.parameters():
                    par.requires_grad = False

                for name, par in self.model.named_parameters():
                    if 'conv1' in name or 'layer' in name:
                        # print(f"activate {name}")
                        par.requires_grad = True


            if self.options.ablation_layerteset_decOnly:
                for par in self.model.parameters():
                    par.requires_grad = False

                for name, par in self.model.named_parameters():
                    if 'decpose' in name or 'decshape' in name or 'deccam' in name:
                        # print(f"activate {name}")
                        par.requires_grad = True


            if self.options.ablation_layerteset_fc2Later:
                for par in self.model.parameters():
                    par.requires_grad = False

                for name, par in self.model.named_parameters():
                    if 'fc2' in name or 'decpose' in name or 'decshape' in name or 'deccam' in name:
                        # print(f"activate {name}")
                        par.requires_grad = True


            #Freeze all except the last layer of Resnet
            if self.options.ablation_layerteset_onlyRes50LastConv:
                for par in self.model.parameters():
                    par.requires_grad = False

                for name, par in self.model.named_parameters():
                    if 'layer4.2.conv3' in name:
                        # print(f"activate {name}")
                        par.requires_grad = True
            

            # g_timer.toc(average =False, bPrint=True,title="reload")
            # self.exemplerTrainingMode()

            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}

            output_backup={}
            for it in range(maxExemplarIter):
                

                ##########################################################################################
                ##### RUN EFT
                ##########################################################################################
                # g_timer.tic()
                if self.options.bUseHand3D:
                    output, losses = self.run_eft_step_wHand(batch)
                else:
                    output, losses = self.run_eft_step(batch, iterIdx=it)
                

                #Check frozeon layers
                # if self.options.abl:
                #     sumVal =0
                #     for par in  self.model.layer4.parameters():
                #         sumVal +=par.mean()
                #     print("fc1 {}, self.model.layer4 {}".format(self.model.fc1.weight.mean(),  sumVal))

                # g_timer.toc(average =True, bPrint=True,title="eachStep")

                output['loss_keypoints_2d'] = losses['loss_keypoints']
                output['loss'] = losses['loss']
                
                if it==0:
                    output_backup['pred_shape'] = output['pred_shape'].copy()
                    output_backup['pred_pose_rotmat'] = output['pred_pose_rotmat'].copy()
                    output_backup['pred_camera'] = output['pred_camera'].copy()

                    output_backup['loss_keypoints_2d'] = output['loss_keypoints_2d']
                    output_backup['loss'] = output['loss']

                    # #Save the first output here for coparison (why??)
                    # batch['pose'] =  torch.tensor(output['pred_pose_rotmat'].copy())# SMPL pose parameters                #[N,72]
                    # batch['betas'] =  torch.tensor(output['pred_shape'].copy()) # SMPL beta parameters              #[N,10]
                    # pred_rotmat_hom = torch.cat([batch['pose'].view(-1, 3, 3), torch.tensor([0,0,1], dtype=torch.float32,).view(1, 3, 1).expand(batch['pose'].shape[0] * 24, -1, -1)], dim=-1)
                    # batch['pose'] = rotation_matrix_to_angle_axis(pred_rotmat_hom).contiguous().view(batch['pose'].shape[0], -1)
                
                # print("keypoint loss: {}".format(output['loss_keypoints_2d']))
                
                # Thresholding by 2D keypoint error
                if True:
                    if output['loss_keypoints_2d']< self.options.eft_thresh_keyptErr_2d: # 1e-4:
                        # glViewer.show(0)
                        break

            g_timer.toc(average =True, bPrint=True,title="wholeEFT")
            
            if self.options.bDebug_visEFT:
                # glViewer.show(0)

                if False:   #Render to File
                    imgname = output['imageName'][0]
                    root_imgname = os.path.basename(imgname)[:-4]
                    renderRoot=f'/home/hjoo/temp/render_eft/eft_{root_imgname}'
                    smpl_utils.renderSMPLoutput_merge(renderRoot)

                glViewer.show(0)

            output['pred_shape_init'] = output_backup['pred_shape'] 
            output['pred_pose_rotmat_init']  = output_backup['pred_pose_rotmat']
            output['pred_camera_init'] = output_backup['pred_camera']

            output['loss_init'] = output_backup['loss'] 
            output['loss_keypoints_2d_init']  = output_backup['loss_keypoints_2d']
            output['numOfIteration'] = it

            if self.options.bUseSMPLX:
                output['smpltype'] = 'smplx'
            else:
                output['smpltype'] = 'smpl'

            #Exemplar Tuning Analysis
            if self.options.bExemplar_analysis_testloss and test_dataset_3dpw is not None:
                print(">> Testing : test set size:{}".format(len(test_dataset_3dpw)))
                error_3dpw = self.test(test_dataset_3dpw, '3dpw')
                output['test_error_3dpw'] = error_3dpw
                error_h36m = self.test(test_dataset_h36m, 'h36m-p1')
                output['test_error_h36m'] = error_h36m
            
            if self.options.bExemplar_badsample_finder and test_dataset_3dpw is not None:
                print(">> Testing : test set size:{}".format(len(test_dataset_3dpw)))
                error_3dpw = self.test(test_dataset_3dpw, '3dpw')
                output['test_error_3dpw'] = error_3dpw

            if bExportPKL:    #Export Output to PKL files
                if self.options.db_set =='panoptic' or "haggling" in self.options.db_set:
                    # fileNameOnly = os.path.basename(output['imageName'][0])[:-4]
                    fileNameOnly = (output['imageName'][0])[:-4].replace("/","-")

                    sampleIdx = output['sampleIdx'][0]
                    if self.options.bExemplar_dataLoaderStart>=0:
                        sampleIdx +=self.options.bExemplar_dataLoaderStart

                    if sampleIdx%100==0:
                        outputList[sampleIdx] = output

                        # fileName = '{:80d}.pkl'.format(fileNameOnly,sampleIdx)
                        fileName = '{:08d}.pkl'.format(sampleIdx)
                        outputPath = os.path.join(exemplarOutputPath,fileName)
                        print("Saved:{}".format(outputPath))
                        with open(outputPath,'wb') as f:
                            pickle.dump(outputList,f)       #Bug fixed
                            f.close()
                        
                        outputList ={}      #reset
                    else:
                        outputList[sampleIdx] = output

                elif "3dpw" in self.options.db_set:
                    fileNameOnly = os.path.basename(output['imageName'][0])[:-4]
                    seqName = os.path.basename(os.path.dirname(output['imageName'][0]))
                    fileNameOnly = f"{seqName}_{fileNameOnly}"
                    # fileNameOnly = (output['imageName'][0])[:-4].replace("/","-")

                    sampleIdx = output['sampleIdx'][0].item()
                    if self.options.bExemplar_dataLoaderStart>=0:
                        sampleIdx +=self.options.bExemplar_dataLoaderStart

                    fileName = '{}_{}.pkl'.format(fileNameOnly,sampleIdx)
                    outputPath = os.path.join(exemplarOutputPath,fileName)

                    print("Saved:{}".format(outputPath))
                    with open(outputPath,'wb') as f:
                        pickle.dump(output,f)       
                        f.close()

                else:
                    fileNameOnly = os.path.basename(output['imageName'][0])[:-4]
                    # fileNameOnly = (output['imageName'][0])[:-4].replace("/","-")

                    sampleIdx = output['sampleIdx'][0].item()
                    if self.options.bExemplar_dataLoaderStart>=0:
                        sampleIdx +=self.options.bExemplar_dataLoaderStart

                    fileName = '{}_{}.pkl'.format(fileNameOnly,sampleIdx)
                    outputPath = os.path.join(exemplarOutputPath,fileName)

                    print("Saved:{}".format(outputPath))
                    with open(outputPath,'wb') as f:
                        pickle.dump(output,f)       
                        f.close()
                        
            # # # Tensorboard logging every summary_steps steps
            # if self.step_count % self.options.summary_steps == 0:
            #     self.train_summaries(batch, *out)

    

    #Run EFT
    #Save output as seperate pkl files
    def eftAllInDB_3dpwtest(self, test_dataset_3dpw = None, test_dataset_h36m= None, bExportPKL = True):

        if config.bIsDevfair:
            now = datetime.datetime.now()
            # newName = '{:02d}-{:02d}-{}'.format(now.month, now.day, now.hour*3600 + now.minute*60 + now.second)
            newName = '{:02d}-{:02d}'.format(now.month, now.day)
            outputDir = newName + '_' + self.options.db_set + '_' + self.options.name
        else:
            now = datetime.datetime.now()
            # outputDir = self.options.db_set
            newName = 'test_{:02d}-{:02d}-{}'.format(now.month, now.day, now.hour*3600 + now.minute*60 + now.second)
            outputDir = newName + '_' + self.options.db_set + '_' + self.options.name

        exemplarOutputPath = os.path.join(config.EXEMPLAR_OUTPUT_ROOT , outputDir)
        if not os.path.exists(exemplarOutputPath):
            os.mkdir(exemplarOutputPath)

        """Training process."""
        # Run training for num_epochs epochs
        # Create new DataLoader every epoch and (possibly) resume from an arbitrary step inside an epoch
        train_data_loader = CheckpointDataLoader(self.train_ds,checkpoint=self.checkpoint,
                                                    batch_size=1,       #Always o1
                                                    num_workers=self.options.num_workers,
                                                    pin_memory=self.options.pin_memory,
                                                    shuffle=False)      #No Shuffle      
        
        maxExemplarIter = self.options.maxExemplarIter
       
        # Iterate over all batches in an epoch
        outputList ={}
        reconError =[]
        for step, batch in enumerate(tqdm(train_data_loader)):#, desc='Epoch '+str(epoch),
                                        #     total=len(self.train_ds) // self.options.batch_size,
                                        #     initial=train_data_loader.checkpoint_batch_idx),
                                        # train_data_loader.checkpoint_batch_idx):
            # if step==100:
            #     break
            #3DPW test
            # if 'downtown_bus_00' not in batch['imgname']:
            #     continue

            #Only performed for 1/100 data (roughly hundred level)
            if self.options.bExemplar_analysis_testloss:
                sampleIdx = batch['sample_index'][0].item()
                if sampleIdx%100 !=0:
                    continue
            
            if self.options.bExemplar_badsample_finder:
                sampleIdx = batch['sample_index'][0].item()
                # if sampleIdx%100 !=0:
                #     continue

            bSkipExisting  =  self.options.bNotSkipExemplar==False     #bNotSkipExemplar ===True --> bSkipExisting==False
            if bSkipExisting:
                if self.options.db_set =='panoptic':
                    # fileNameOnly = os.path.basename(output['imageName'][0])[:-4]
                    # fileNameOnly = (batch['pkl_save_name'][0])[:-4].replace("/","-")

                    sampleIdx = batch['sample_index'][0].item()
                    if self.options.bExemplar_dataLoaderStart>=0:
                        sampleIdx +=self.options.bExemplar_dataLoaderStart

                    sampleIdxSaveFrame = 100* (int(sampleIdx/100.0) + 1)
                    
                    fileName = '{:08d}.pkl'.format(sampleIdxSaveFrame)

                    outputPath = os.path.join(exemplarOutputPath,fileName)
                    # print(">> checking: {}".format(outputPath))
                    if os.path.exists(outputPath):
                        print("Skipped: {}".format(outputPath))
                        continue
                elif '3dpw' in self.options.db_set:
                    fileNameOnly = os.path.basename(batch['imgname'][0])[:-4]
                    seqName = os.path.basename(os.path.dirname(batch['imgname'][0]))
                    fileNameOnly = f"{seqName}_{fileNameOnly}"

                    sampleIdx = batch['sample_index'][0].item()
                    if self.options.bExemplar_dataLoaderStart>=0:
                        sampleIdx +=self.options.bExemplar_dataLoaderStart
                    fileName = '{}_{}.pkl'.format(fileNameOnly,sampleIdx)
                    outputPath = os.path.join(exemplarOutputPath,fileName)
                    if os.path.exists(outputPath):
                        print("Skipped: {}".format(outputPath))
                        continue
                else:
                    fileNameOnly = os.path.basename(batch['imgname'][0])[:-4]
                    sampleIdx = batch['sample_index'][0].item()
                    if self.options.bExemplar_dataLoaderStart>=0:
                        sampleIdx +=self.options.bExemplar_dataLoaderStart
                    fileName = '{}_{}.pkl'.format(fileNameOnly,sampleIdx)
                    outputPath = os.path.join(exemplarOutputPath,fileName)
                    if os.path.exists(outputPath):
                        print("Skipped: {}".format(outputPath))
                        continue
                    
            # g_timer.tic()
            self.reloadModel()  #For each sample
            # g_timer.toc(average =False, bPrint=True,title="reload")
            # self.exemplerTrainingMode()

            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}

            output_backup={}
            reconErrorInfo ={}
            for it in range(maxExemplarIter):
                
                g_timer.tic()
                if self.options.bUseHand3D:
                    output, losses = self.run_eft_step_wHand(batch)
                else:
                    output, losses = self.run_eft_step(batch)
                
                #Check r_error

                
                reconErrorInfo[it] = (losses['r_error'], losses['loss_keypoints'])


                # g_timer.toc(average =False, bPrint=True,title="eachStep"
                output['loss_keypoints_2d'] = losses['loss_keypoints']
                output['loss'] = losses['loss']

                if it==0:
                    output_backup['pred_shape'] = output['pred_shape'].copy()
                    output_backup['pred_pose_rotmat'] = output['pred_pose_rotmat'].copy()
                    output_backup['pred_camera'] = output['pred_camera'].copy()

                    output_backup['loss_keypoints_2d'] = output['loss_keypoints_2d']
                    output_backup['loss'] = output['loss']

                    # #Save the first output here for coparison (why??)
                    # batch['pose'] =  torch.tensor(output['pred_pose_rotmat'].copy())# SMPL pose parameters                #[N,72]
                    # batch['betas'] =  torch.tensor(output['pred_shape'].copy()) # SMPL beta parameters              #[N,10]
                    # pred_rotmat_hom = torch.cat([batch['pose'].view(-1, 3, 3), torch.tensor([0,0,1], dtype=torch.float32,).view(1, 3, 1).expand(batch['pose'].shape[0] * 24, -1, -1)], dim=-1)
                    # batch['pose'] = rotation_matrix_to_angle_axis(pred_rotmat_hom).contiguous().view(batch['pose'].shape[0], -1)

            reconError.append(reconErrorInfo)
            output['reconErrorInfo'] = reconErrorInfo

            output['pred_shape_init'] = output_backup['pred_shape'] 
            output['pred_pose_rotmat_init']  = output_backup['pred_pose_rotmat']
            output['pred_camera_init'] = output_backup['pred_camera']

            output['loss_init'] = output_backup['loss'] 
            output['loss_keypoints_2d_init']  = output_backup['loss_keypoints_2d']
            output['numOfIteration'] = it

            if self.options.bUseSMPLX:
                output['smpltype'] = 'smplx'
            else:
                output['smpltype'] = 'smpl'

            #Exemplar Tuning Analysis
            if self.options.bExemplar_analysis_testloss and test_dataset_3dpw is not None:
                print(">> Testing : test set size:{}".format(len(test_dataset_3dpw)))
                error_3dpw = self.test(test_dataset_3dpw, '3dpw')
                output['test_error_3dpw'] = error_3dpw
                error_h36m = self.test(test_dataset_h36m, 'h36m-p1')
                output['test_error_h36m'] = error_h36m
            
            if self.options.bExemplar_badsample_finder and test_dataset_3dpw is not None:
                print(">> Testing : test set size:{}".format(len(test_dataset_3dpw)))
                error_3dpw = self.test(test_dataset_3dpw, '3dpw')
                output['test_error_3dpw'] = error_3dpw

            if bExportPKL:    #Export Output to PKL files
                if self.options.db_set =='panoptic' or "haggling" in self.options.db_set:
                    # fileNameOnly = os.path.basename(output['imageName'][0])[:-4]
                    fileNameOnly = (output['imageName'][0])[:-4].replace("/","-")

                    sampleIdx = output['sampleIdx'][0]
                    if self.options.bExemplar_dataLoaderStart>=0:
                        sampleIdx +=self.options.bExemplar_dataLoaderStart

                    if sampleIdx%100==0:
                        outputList[sampleIdx] = output

                        # fileName = '{:80d}.pkl'.format(fileNameOnly,sampleIdx)
                        fileName = '{:08d}.pkl'.format(sampleIdx)
                        outputPath = os.path.join(exemplarOutputPath,fileName)
                        print("Saved:{}".format(outputPath))
                        with open(outputPath,'wb') as f:
                            pickle.dump(outputList,f)       #Bug fixed
                            f.close()
                        
                        outputList ={}      #reset
                    else:
                        outputList[sampleIdx] = output

                elif "3dpw" in self.options.db_set:
                    fileNameOnly = os.path.basename(output['imageName'][0])[:-4]
                    seqName = os.path.basename(os.path.dirname(output['imageName'][0]))
                    fileNameOnly = f"{seqName}_{fileNameOnly}"
                    # fileNameOnly = (output['imageName'][0])[:-4].replace("/","-")

                    sampleIdx = output['sampleIdx'][0].item()
                    if self.options.bExemplar_dataLoaderStart>=0:
                        sampleIdx +=self.options.bExemplar_dataLoaderStart

                    fileName = '{}_{}.pkl'.format(fileNameOnly,sampleIdx)
                    outputPath = os.path.join(exemplarOutputPath,fileName)

                    print("Saved:{}".format(outputPath))
                    with open(outputPath,'wb') as f:
                        pickle.dump(output,f)       
                        f.close()

                else:
                    fileNameOnly = os.path.basename(output['imageName'][0])[:-4]
                    # fileNameOnly = (output['imageName'][0])[:-4].replace("/","-")

                    sampleIdx = output['sampleIdx'][0].item()
                    if self.options.bExemplar_dataLoaderStart>=0:
                        sampleIdx +=self.options.bExemplar_dataLoaderStart

                    fileName = '{}_{}.pkl'.format(fileNameOnly,sampleIdx)
                    outputPath = os.path.join(exemplarOutputPath,fileName)

                    print("Saved:{}".format(outputPath))
                    with open(outputPath,'wb') as f:
                        pickle.dump(output,f)       
                        f.close()
                        
            # # # Tensorboard logging every summary_steps steps
            # if self.step_count % self.options.summary_steps == 0:
            #     self.train_summaries(batch, *out)

        if False:       #Display the best iteration
            reconErrorPerIter=[]
            for it in range(maxExemplarIter):
                print(it)
                reconErrorPerIter.append([d[it][0] for d in reconError])
            # viewer2D.Plot(reconError)
            reconErrorPerIter = np.array(reconErrorPerIter)
            for it in range(maxExemplarIter):
                print("{}: reconError:{}".format(it, np.mean(reconErrorPerIter[it,:])))
            #Find the best
            print("Best: reconError:{}".format(np.mean(np.min(reconErrorPerIter,axis=0))))



    #Run SMPLify
    #Save output as seperate pkl files
    def smplifyAllInDB(self, test_dataset_3dpw = None, test_dataset_h36m= None, bExportPKL = True):

        if config.bIsDevfair:
            now = datetime.datetime.now()
            # newName = '{:02d}-{:02d}-{}'.format(now.month, now.day, now.hour*3600 + now.minute*60 + now.second)
            newName = '{:02d}-{:02d}'.format(now.month, now.day)
            outputDir = newName + '_' + self.options.db_set + '_' + self.options.name
        else:
            now = datetime.datetime.now()
            # outputDir = self.options.db_set
            newName = 'test_{:02d}-{:02d}-{}'.format(now.month, now.day, now.hour*3600 + now.minute*60 + now.second)
            outputDir = newName + '_' + self.options.db_set + '_' + self.options.name

        exemplarOutputPath = os.path.join(config.EXEMPLAR_OUTPUT_ROOT , outputDir)
        if not os.path.exists(exemplarOutputPath):
            os.mkdir(exemplarOutputPath)

        """Training process."""
        # Run training for num_epochs epochs
        # Create new DataLoader every epoch and (possibly) resume from an arbitrary step inside an epoch
        train_data_loader = CheckpointDataLoader(self.train_ds,checkpoint=self.checkpoint,
                                                    batch_size=1,       #Always o1
                                                    num_workers=self.options.num_workers,
                                                    pin_memory=self.options.pin_memory,
                                                    shuffle=False)      #No Shuffle      
        
        # maxExemplarIter = self.options.maxExemplarIter
       
        # Iterate over all batches in an epoch
        outputList ={}
        for step, batch in enumerate(tqdm(train_data_loader)):#, desc='Epoch '+str(epoch),
                                        #     total=len(self.train_ds) // self.options.batch_size,
                                        #     initial=train_data_loader.checkpoint_batch_idx),
                                        # train_data_loader.checkpoint_batch_idx):
            

            #Only performed for 1/100 data (roughly hundred level)
            if self.options.bExemplar_analysis_testloss:
                sampleIdx = batch['sample_index'][0].item()
                if sampleIdx%100 !=0:
                    continue
            
            if self.options.bExemplar_badsample_finder:
                sampleIdx = batch['sample_index'][0].item()
                # if sampleIdx%100 !=0:
                #     continue

            bSkipExisting  =  self.options.bNotSkipExemplar==False     #bNotSkipExemplar ===True --> bSkipExisting==False
            if bSkipExisting:
                if self.options.db_set =='panoptic':
                    # fileNameOnly = os.path.basename(output['imageName'][0])[:-4]
                    # fileNameOnly = (batch['pkl_save_name'][0])[:-4].replace("/","-")

                    sampleIdx = batch['sample_index'][0].item()
                    if self.options.bExemplar_dataLoaderStart>=0:
                        sampleIdx +=self.options.bExemplar_dataLoaderStart

                    sampleIdxSaveFrame = 100* (int(sampleIdx/100.0) + 1)
                    
                    fileName = '{:08d}.pkl'.format(sampleIdxSaveFrame)

                    outputPath = os.path.join(exemplarOutputPath,fileName)
                    # print(">> checking: {}".format(outputPath))
                    if os.path.exists(outputPath):
                        print("Skipped: {}".format(outputPath))
                        continue
                elif '3dpw' in self.options.db_set:
                    fileNameOnly = os.path.basename(batch['imgname'][0])[:-4]
                    seqName = os.path.basename(os.path.dirname(batch['imgname'][0]))
                    fileNameOnly = f"{seqName}_{fileNameOnly}"

                    sampleIdx = batch['sample_index'][0].item()
                    if self.options.bExemplar_dataLoaderStart>=0:
                        sampleIdx +=self.options.bExemplar_dataLoaderStart
                    fileName = '{}_{}.pkl'.format(fileNameOnly,sampleIdx)
                    outputPath = os.path.join(exemplarOutputPath,fileName)
                    if os.path.exists(outputPath):
                        print("Skipped: {}".format(outputPath))
                        continue
                else:
                    fileNameOnly = os.path.basename(batch['imgname'][0])[:-4]
                    sampleIdx = batch['sample_index'][0].item()
                    if self.options.bExemplar_dataLoaderStart>=0:
                        sampleIdx +=self.options.bExemplar_dataLoaderStart
                    fileName = '{}_{}.pkl'.format(fileNameOnly,sampleIdx)
                    outputPath = os.path.join(exemplarOutputPath,fileName)
                    if os.path.exists(outputPath):
                        print("Skipped: {}".format(outputPath))
                        continue
                    
            # g_timer.tic()
            self.reloadModel()  #For each sample
            # g_timer.toc(average =False, bPrint=True,title="reload")
            # self.exemplerTrainingMode()

            output_backup={}
            
            ##############################
            ## Run SMPLify Iteration
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
            g_timer.tic()
            output = self.run_smplify(batch)
            
            #additional outputs
            output['pretrained_checkpoint']= self.options.pretrained_checkpoint     #For initial model
            output['method'] ='smplify'
            if self.options.bUseSMPLX:
                output['smpltype'] = 'smplx'
            else:
                output['smpltype'] = 'smpl'

            """"
            Ouput should hvae the following

            output={}
            output['pred_pose_rotmat']
            output['pred_shape'] 
            output['pred_camera']

            output['pred_pose_rotmat_init'] = init_pred_rotmat.detach().cpu().numpy()
            output['pred_shape_init'] = init_pred_betas.detach().cpu().numpy()
            output['pred_camera_init'] = init_pred_camera.detach().cpu().numpy()
        
            #If there exists SPIN fits, save that for comparison later
            output['spin_pose'] = opt_pose.detach().cpu().numpy()       #was called opt_pose, opt_beta
            output['spin_beta'] = opt_betas.detach().cpu().numpy()

            output['sampleIdx'] = input_batch['sample_index'].detach().cpu().numpy()     #To use loader directly
            output['imageName'] = input_batch['imgname']

            #Bbox info
            output['scale'] = input_batch['scale'] .detach().cpu().numpy()
            output['center'] = input_batch['center'].detach().cpu().numpy()

            output['annotId'] = input_batch['annotId'].detach().cpu().numpy()
            output['subjectId'] = input_batch['subjectId'][0].item()

            #To save new db file
            output['keypoint2d'] = input_batch['keypoints_original'].detach().cpu().numpy()
            output['keypoint2d_cropped'] = input_batch['keypoints'].detach().cpu().numpy()

            output['loss_keypoints_2d'] = losses['loss_keypoints']
            output['loss_keypoints_2d_init']  = output_backup['loss_keypoints_2d']
            output['loss'] = losses['loss_keypoints']
            output['loss_init'] = output_backup['loss'] 
            output['numOfIteration'] = it
            output['smpltype'] = 'smplx'  or 'smp'
            """


            if bExportPKL:    #Export Output to PKL files
                if self.options.db_set =='panoptic':
                    # fileNameOnly = os.path.basename(output['imageName'][0])[:-4]
                    fileNameOnly = (output['imageName'][0])[:-4].replace("/","-")

                    sampleIdx = output['sampleIdx'][0]
                    if self.options.bExemplar_dataLoaderStart>=0:
                        sampleIdx +=self.options.bExemplar_dataLoaderStart

                    if sampleIdx%100==0:
                        outputList[sampleIdx] = output

                        # fileName = '{:80d}.pkl'.format(fileNameOnly,sampleIdx)
                        fileName = '{:08d}.pkl'.format(sampleIdx)
                        outputPath = os.path.join(exemplarOutputPath,fileName)
                        print("Saved:{}".format(outputPath))
                        with open(outputPath,'wb') as f:
                            pickle.dump(outputList,f)       #Bug fixed
                            f.close()
                        
                        outputList ={}      #reset
                    else:
                        outputList[sampleIdx] = output

                elif "3dpw" in self.options.db_set:
                    fileNameOnly = os.path.basename(output['imageName'][0])[:-4]
                    seqName = os.path.basename(os.path.dirname(output['imageName'][0]))
                    fileNameOnly = f"{seqName}_{fileNameOnly}"
                    # fileNameOnly = (output['imageName'][0])[:-4].replace("/","-")

                    sampleIdx = output['sampleIdx'][0].item()
                    if self.options.bExemplar_dataLoaderStart>=0:
                        sampleIdx +=self.options.bExemplar_dataLoaderStart

                    fileName = '{}_{}.pkl'.format(fileNameOnly,sampleIdx)
                    outputPath = os.path.join(exemplarOutputPath,fileName)

                    print("Saved:{}".format(outputPath))
                    with open(outputPath,'wb') as f:
                        pickle.dump(output,f)       
                        f.close()

                else:
                    fileNameOnly = os.path.basename(output['imageName'][0])[:-4]
                    # fileNameOnly = (output['imageName'][0])[:-4].replace("/","-")

                    sampleIdx = output['sampleIdx'][0].item()
                    if self.options.bExemplar_dataLoaderStart>=0:
                        sampleIdx +=self.options.bExemplar_dataLoaderStart

                    fileName = '{}_{}.pkl'.format(fileNameOnly,sampleIdx)
                    outputPath = os.path.join(exemplarOutputPath,fileName)

                    print("Saved:{}".format(outputPath))
                    with open(outputPath,'wb') as f:
                        pickle.dump(output,f)       
                        f.close()
                        
            # # # Tensorboard logging every summary_steps steps
            # if self.step_count % self.options.summary_steps == 0:
            #     self.train_summaries(batch, *out)

