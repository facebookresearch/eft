import submitit
import torch
import torch.nn as nn
import numpy as np
from torchgeometry import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
import cv2

from fairmocap.datasets import MixedDataset, BaseDataset
from fairmocap.models import hmr, SMPL, SMPLX


from fairmocap.smplify import SMPLify
from fairmocap.utils.geometry import batch_rodrigues, perspective_projection, estimate_translation, weakProjection_gpu
from fairmocap.core import BaseTrainer

from fairmocap.core import config
from fairmocap.core import constants
from .fits_dict import FitsDict

from renderer import viewer2D
from renderer import glViewer

from torch.utils.tensorboard import SummaryWriter


# from fairmocap.apps.eval import run_evaluation #For test
from fairmocap.apps.eval_multicrop import run_evaluation #For test
from fairmocap.utils.timer import Timer
g_timer = Timer()

from fairmocap.utils import CheckpointDataLoader, CheckpointSaver

# g_smplx = True      #TODO: option should be aded in the parameters

# g_debugVisualize = True

#(N,57) (N,) (N,2)
# def weakProjection_gpu(skel3D, scale, trans2D ):
#     # if len(skel3D.shape)==1:
#     #     skel3D = np.reshape(skel3D, (-1,3))

#     skel3D = skel3D.view((skel3D.shape[0],-1,3))
#     trans2D = trans2D.view((trans2D.shape[0],1,2))
#     scale = scale.view((scale.shape[0],1,1))
#     skel3D_proj = scale* skel3D[:,:,:2] + trans2D

#     return skel3D_proj#skel3D_proj.view((skel3D.shape[0],-1))       #(N, 19*2) o

def normalize_2dvector(gt_boneOri_leftLeg):
        gt_boneOri_leftLeg_norm =torch.norm(gt_boneOri_leftLeg,dim=1)          #(N)
        # gt_boneOri_leftLeg[:,0] = gt_boneOri_leftLeg[:,0]/gt_boneOri_leftLeg_norm
        # gt_boneOri_leftLeg[:,1] = gt_boneOri_leftLeg[:,1]/gt_boneOri_leftLeg_norm
        gt_boneOri_leftLeg = gt_boneOri_leftLeg /gt_boneOri_leftLeg_norm

        return gt_boneOri_leftLeg, gt_boneOri_leftLeg_norm

class DDP_Trainer(BaseTrainer):
    
    def __init__(self, options):
        self.options = options
        self.endtime = time.time() + self.options.time_to_run
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # override this function to define your model, optimizers etc.
        
        self._setup_process_group() #For DDP

        self.init_fn()
        self.saver = CheckpointSaver(save_dir=options.checkpoint_dir)
        self.summary_writer = SummaryWriter(self.options.summary_dir)

        self.checkpoint = None
        if self.options.resume and self.saver.exists_checkpoint():
            self.checkpoint = self.saver.load_checkpoint(self.models_dict, self.optimizers_dict, checkpoint_file=self.options.checkpoint)

        if self.checkpoint is None:
            self.epoch_count = 0
            self.step_count = 0
        else:
            self.epoch_count = self.checkpoint['epoch']
            self.step_count = self.checkpoint['total_step_count']


    def _setup_process_group(self) -> None:
        job_env = submitit.JobEnvironment()
        torch.cuda.set_device(job_env.local_rank)
        torch.distributed.init_process_group(
            backend=self._cluster_cfg.dist_backend,
            init_method=self._cluster_cfg.dist_url,
            world_size=job_env.num_tasks,
            rank=job_env.global_rank,
        )
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")

    def init_fn(self):

        job_env = submitit.JobEnvironment()

        self.train_ds = MixedDataset(self.options, ignore_3d=self.options.ignore_3d, is_train=True)

        self.model = hmr(config.SMPL_MEAN_PARAMS, pretrained=True).to(self.device)
        self.model.cuda(job_env.local_rank)
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model, device_ids=[job_env.local_rank], output_device=job_env.local_rank
        )

        if self.options.bExemplarMode:
            lr = 5e-5 * 0.2
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
        # if torch.cuda.device_count() > 1:
        assert torch.cuda.device_count() > 1
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        
        # self.model = torch.nn.DataParallel(self.model)      #Failed... 
        # self.model.cuda(job_env.local_rank)


        # Load dictionary of fits
        self.fits_dict = FitsDict(self.options, self.train_ds)

        # Create renderer
        self.renderer = None# Renderer(focal_length=self.focal_length, img_res=self.options.img_res, faces=self.smpl.faces)

        #debug
        from torchvision.transforms import Normalize
        self.de_normalize_img = Normalize(mean=[ -constants.IMG_NORM_MEAN[0]/constants.IMG_NORM_STD[0]    , -constants.IMG_NORM_MEAN[1]/constants.IMG_NORM_STD[1], -constants.IMG_NORM_MEAN[2]/constants.IMG_NORM_STD[2]], std=[1/constants.IMG_NORM_STD[0], 1/constants.IMG_NORM_STD[1], 1/constants.IMG_NORM_STD[2]])

    def finalize(self):
        pass
        # self.fits_dict.save()

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight):
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        conf[:, 25:] *= gt_weight
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d):
        """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence.
        """
        pred_keypoints_3d = pred_keypoints_3d[:, 25:, :]
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
        conf = conf[has_pose_3d == 1]
        pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def shape_loss(self, pred_vertices, gt_vertices, has_smpl):
        """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
        pred_vertices_with_shape = pred_vertices[has_smpl == 1]
        gt_vertices_with_shape = gt_vertices[has_smpl == 1]
        if len(gt_vertices_with_shape) > 0:
            return self.criterion_shape(pred_vertices_with_shape, gt_vertices_with_shape)
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl):
        pred_rotmat_valid = pred_rotmat[has_smpl == 1]
        gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1,3)).view(-1, 24, 3, 3)[has_smpl == 1]
        pred_betas_valid = pred_betas[has_smpl == 1]
        gt_betas_valid = gt_betas[has_smpl == 1]
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas

    def train_step(self, input_batch):
        self.model.train()
        
        # Get data from the batch
        images = input_batch['img'] # input image
        gt_keypoints_2d = input_batch['keypoints'] # 2D keypoints           #[N,49,3]
        gt_pose = input_batch['pose'] # SMPL pose parameters                #[N,72]
        gt_betas = input_batch['betas'] # SMPL beta parameters              #[N,10]
        gt_joints = input_batch['pose_3d'] # 3D pose                        #[N,24,4]
        has_smpl = input_batch['has_smpl'].byte() ==1 # flag that indicates whether SMPL parameters are valid
        has_pose_3d = input_batch['has_pose_3d'].byte()==1 # flag that indicates whether 3D pose is valid
        is_flipped = input_batch['is_flipped'] # flag that indicates whether image was flipped during data augmentation
        rot_angle = input_batch['rot_angle'] # rotation angle used for data augmentation
        dataset_name = input_batch['dataset_name'] # name of the dataset the image comes from
        indices = input_batch['sample_index'] # index of example inside its dataset
        batch_size = images.shape[0]


        #Debug temporary scaling for h36m
        # Get GT vertices and model joints
        # Note that gt_model_joints is different from gt_joints as it comes from SMPL
        gt_out = self.smpl(betas=gt_betas, body_pose=gt_pose[:,3:], global_orient=gt_pose[:,:3])
        
        gt_model_joints = gt_out.joints.detach()             #[N, 49, 3]
        gt_vertices = gt_out.vertices

        # else:
        #     gt_out = self.smpl(betas=gt_betas, body_pose=gt_pose[:,3:-6], global_orient=gt_pose[:,:3])

        #     gt_model_joints = gt_out.joints.detach()             #[N, 49, 3]
        #     gt_vertices = gt_out.vertices

        # Get current best fits from the dictionary

        opt_pose, opt_betas, opt_validity = self.fits_dict[(dataset_name, indices.cpu(), rot_angle.cpu(), is_flipped.cpu())]
        opt_pose = opt_pose.to(self.device)
        opt_betas = opt_betas.to(self.device)
        # if g_smplx == False:
        opt_output = self.smpl(betas=opt_betas, body_pose=opt_pose[:,3:], global_orient=opt_pose[:,:3])

        opt_vertices = opt_output.vertices
        opt_joints = opt_output.joints.detach()

        # else:
        #     opt_output = self.smpl(betas=opt_betas, body_pose=opt_pose[:,3:-6], global_orient=opt_pose[:,:3])

        #     opt_vertices = opt_output.vertices
        #     opt_joints = opt_output.joints.detach()
        

        #assuer that non valid opt has GT values
        if len(has_smpl[opt_validity==0])>0:
            assert min(has_smpl[opt_validity==0])  #All should be True


        #assuer that non valid opt has GT values
        if len(has_smpl[opt_validity==0])>0:
            assert min(has_smpl[opt_validity==0])  #All should be True

        # De-normalize 2D keypoints from [-1,1] to pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()              
        gt_keypoints_2d_orig[:, :, :-1] = 0.5 * self.options.img_res * (gt_keypoints_2d_orig[:, :, :-1] + 1)

        # Estimate camera translation given the model joints and 2D keypoints
        # by minimizing a weighted least squares loss
        gt_cam_t = estimate_translation(gt_model_joints, gt_keypoints_2d_orig, focal_length=self.focal_length, img_size=self.options.img_res)

        opt_cam_t = estimate_translation(opt_joints, gt_keypoints_2d_orig, focal_length=self.focal_length, img_size=self.options.img_res)


        opt_joint_loss = self.smplify.get_fitting_loss(opt_pose, opt_betas, opt_cam_t,          #opt_pose (N,72)  (N,10)  opt_cam_t: (N,3)
                                                       0.5 * self.options.img_res * torch.ones(batch_size, 2, device=self.device),      #(N,2)   (112, 112)
                                                       gt_keypoints_2d_orig).mean(dim=-1)

        # Feed images in the network to predict camera and SMPL parameters
        pred_rotmat, pred_betas, pred_camera = self.model(images)

        # if g_smplx == False: #Original
        pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        # else:
        #     pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:-2], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)

        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints


        # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
        # This camera translation can be used in a full perspective projection
        pred_cam_t = torch.stack([pred_camera[:,1],
                                  pred_camera[:,2],
                                  2*self.focal_length/(self.options.img_res * pred_camera[:,0] +1e-9)],dim=-1)


        camera_center = torch.zeros(batch_size, 2, device=self.device)
        pred_keypoints_2d = perspective_projection(pred_joints,
                                                   rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
                                                   translation=pred_cam_t,
                                                   focal_length=self.focal_length,
                                                   camera_center=camera_center)
        # Normalize keypoints to [-1,1]
        pred_keypoints_2d = pred_keypoints_2d / (self.options.img_res / 2.)

        #Weak Projection
        if self.options.bUseWeakProj:
            pred_keypoints_2d = weakProjection_gpu(pred_joints, pred_camera[:,0], pred_camera[:,1:] )           #N, 49, 2

        bFootOriLoss = False
        if bFootOriLoss:    #Ignore hips and hip centers, foot
            # LENGTH_THRESHOLD = 0.0089 #1/112.0     #at least it should be 5 pixel
            #Disable parts
            gt_keypoints_2d[:,2+25,2]=0
            gt_keypoints_2d[:,3+25,2]=0
            gt_keypoints_2d[:,14+25,2]=0

            #Disable Foots
            gt_keypoints_2d[:,5+25,2]=0     #Left foot
            gt_keypoints_2d[:,0+25,2]=0     #Right foot


        if self.options.run_smplify:

            # Convert predicted rotation matrices to axis-angle
            pred_rotmat_hom = torch.cat([pred_rotmat.detach().view(-1, 3, 3).detach(), torch.tensor([0,0,1], dtype=torch.float32,
                device=self.device).view(1, 3, 1).expand(batch_size * 24, -1, -1)], dim=-1)
            pred_pose = rotation_matrix_to_angle_axis(pred_rotmat_hom).contiguous().view(batch_size, -1)
            # tgm.rotation_matrix_to_angle_axis returns NaN for 0 rotation, so manually hack it
            pred_pose[torch.isnan(pred_pose)] = 0.0

            # Run SMPLify optimization starting from the network prediction
            new_opt_vertices, new_opt_joints,\
            new_opt_pose, new_opt_betas,\
            new_opt_cam_t, new_opt_joint_loss = self.smplify(
                                        pred_pose.detach(), pred_betas.detach(),
                                        pred_cam_t.detach(),
                                        0.5 * self.options.img_res * torch.ones(batch_size, 2, device=self.device),
                                        gt_keypoints_2d_orig)
            new_opt_joint_loss = new_opt_joint_loss.mean(dim=-1)

            # Will update the dictionary for the examples where the new loss is less than the current one
            update = (new_opt_joint_loss < opt_joint_loss)
            # print("new_opt_joint_loss{} vs opt_joint_loss{}".format(new_opt_joint_loss))

            if True:   #Visualize opt
                for b in range(batch_size):

                    curImgVis = images[b]     #3,224,224
                    curImgVis = self.de_normalize_img(curImgVis).cpu().numpy()
                    curImgVis = np.transpose( curImgVis , (1,2,0) )*255.0
                    curImgVis =curImgVis[:,:,[2,1,0]] 

                    #Denormalize image
                    curImgVis = np.ascontiguousarray(curImgVis, dtype=np.uint8)
                    viewer2D.ImShow(curImgVis, name='rawIm')
                    originalImg = curImgVis.copy()

                    pred_camera_vis = pred_camera.detach().cpu().numpy()


                    opt_vert_vis = opt_vertices[b].detach().cpu().numpy() 
                    opt_vert_vis *=pred_camera_vis[b,0]
                    opt_vert_vis[:,0] += pred_camera_vis[b,1]        #no need +1 (or  112). Rendernig has this offset already
                    opt_vert_vis[:,1] += pred_camera_vis[b,2]        #no need +1 (or  112). Rendernig has this offset already
                    opt_vert_vis*=112
                    opt_meshes = {'ver': opt_vert_vis, 'f': self.smpl.faces}


                    gt_vert_vis = gt_vertices[b].detach().cpu().numpy() 
                    gt_vert_vis *=pred_camera_vis[b,0]
                    gt_vert_vis[:,0] += pred_camera_vis[b,1]        #no need +1 (or  112). Rendernig has this offset already
                    gt_vert_vis[:,1] += pred_camera_vis[b,2]        #no need +1 (or  112). Rendernig has this offset already
                    gt_vert_vis*=112
                    gt_meshes = {'ver': gt_vert_vis, 'f': self.smpl.faces}

                    new_opt_output = self.smpl(betas=new_opt_betas, body_pose=new_opt_pose[:,3:], global_orient=new_opt_pose[:,:3])
                    new_opt_vertices = new_opt_output.vertices
                    new_opt_joints = new_opt_output.joints
                    new_opt_vert_vis = new_opt_vertices[b].detach().cpu().numpy() 
                    new_opt_vert_vis *=pred_camera_vis[b,0]
                    new_opt_vert_vis[:,0] += pred_camera_vis[b,1]        #no need +1 (or  112). Rendernig has this offset already
                    new_opt_vert_vis[:,1] += pred_camera_vis[b,2]        #no need +1 (or  112). Rendernig has this offset already
                    new_opt_vert_vis*=112
                    new_opt_meshes = {'ver': new_opt_vert_vis, 'f': self.smpl.faces}
                    
                    glViewer.setMeshData([new_opt_meshes, gt_meshes, new_opt_meshes], bComputeNormal= True)

                    glViewer.setBackgroundTexture(originalImg)
                    glViewer.setWindowSize(curImgVis.shape[1], curImgVis.shape[0])
                    glViewer.SetOrthoCamera(True)

                    print(has_smpl[b])
                    glViewer.show()
                

            opt_joint_loss[update] = new_opt_joint_loss[update]
            opt_vertices[update, :] = new_opt_vertices[update, :]
            opt_joints[update, :] = new_opt_joints[update, :]
            opt_pose[update, :] = new_opt_pose[update, :]
            opt_betas[update, :] = new_opt_betas[update, :]
            opt_cam_t[update, :] = new_opt_cam_t[update, :]

            self.fits_dict[(dataset_name, indices.cpu(), rot_angle.cpu(), is_flipped.cpu(), update.cpu())] = (opt_pose.cpu(), opt_betas.cpu())

        else:
            update = torch.zeros(batch_size, device=self.device).byte()

        # Replace the optimized parameters with the ground truth parameters, if available
        opt_vertices[has_smpl, :, :] = gt_vertices[has_smpl, :, :]
        opt_cam_t[has_smpl, :] = gt_cam_t[has_smpl, :]
        opt_joints[has_smpl, :, :] = gt_model_joints[has_smpl, :, :]
        opt_pose[has_smpl, :] = gt_pose[has_smpl, :]
        opt_betas[has_smpl, :] = gt_betas[has_smpl, :]

        # Assert whether a fit is valid by comparing the joint loss with the threshold
        valid_fit = (opt_joint_loss < self.options.smplify_threshold).to(self.device)


        if self.options.ablation_no_pseudoGT:  
            valid_fit[:] =False       #Disable all pseudoGT


        # Add the examples with GT parameters to the list of valid fits
        valid_fit = valid_fit | has_smpl    

        # if len(valid_fit) > sum(valid_fit):
        #     print(">> Rejected fit: {}/{}".format(len(valid_fit) - sum(valid_fit), len(valid_fit) ))

        opt_keypoints_2d = perspective_projection(opt_joints,
                                                  rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
                                                  translation=opt_cam_t,
                                                  focal_length=self.focal_length,
                                                  camera_center=camera_center)


        opt_keypoints_2d = opt_keypoints_2d / (self.options.img_res / 2.)


        # Compute loss on SMPL parameters
        loss_regr_pose, loss_regr_betas = self.smpl_losses(pred_rotmat, pred_betas, opt_pose, opt_betas, valid_fit)

        # Compute 2D reprojection loss for the keypoints
        loss_keypoints = self.keypoint_loss(pred_keypoints_2d, gt_keypoints_2d,
                                            self.options.openpose_train_weight,
                                            self.options.gt_train_weight)

        # Compute 3D keypoint loss
        loss_keypoints_3d = self.keypoint_3d_loss(pred_joints, gt_joints, has_pose_3d)

        # Per-vertex loss for the shape
        loss_shape = self.shape_loss(pred_vertices, opt_vertices, valid_fit)


        #Regularization term for shape
        loss_regr_betas_noReject = torch.mean(pred_betas**2)

        
        # Compute total loss
        # The last component is a loss that forces the network to predict positive depth values
        if self.options.ablation_loss_2dkeyonly:        #2D keypoint only
            loss = self.options.keypoint_loss_weight * loss_keypoints +\
                ((torch.exp(-pred_camera[:,0]*10)) ** 2 ).mean() +\
                    self.options.beta_loss_weight * loss_regr_betas_noReject        #Beta regularization

        elif self.options.ablation_loss_noSMPLloss:     #2D no Pose parameter
            loss = self.options.keypoint_loss_weight * loss_keypoints +\
                self.options.keypoint_loss_weight * loss_keypoints_3d +\
                ((torch.exp(-pred_camera[:,0]*10)) ** 2 ).mean() +\
                self.options.beta_loss_weight * loss_regr_betas_noReject        #Beta regularization

        else:
            loss = self.options.shape_loss_weight * loss_shape +\
                self.options.keypoint_loss_weight * loss_keypoints +\
                self.options.keypoint_loss_weight * loss_keypoints_3d +\
                loss_regr_pose + self.options.beta_loss_weight * loss_regr_betas +\
                ((torch.exp(-pred_camera[:,0]*10)) ** 2 ).mean()


        # loss = self.options.keypoint_loss_weight * loss_keypoints #Debug: 2d error only
        # print("DEBUG: 2donly loss")
        loss *= 60


        # Do backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Pack output arguments for tensorboard logging
        output = {'pred_vertices': pred_vertices.detach(),
                  'opt_vertices': opt_vertices,
                  'pred_cam_t': pred_cam_t.detach(),
                  'opt_cam_t': opt_cam_t}
        losses = {'loss': loss.detach().item(),
                  'loss_keypoints': loss_keypoints.detach().item(),
                  'loss_keypoints_3d': loss_keypoints_3d.detach().item(),
                  'loss_regr_pose': loss_regr_pose.detach().item(),
                  'loss_regr_betas': loss_regr_betas.detach().item(),
                  'loss_shape': loss_shape.detach().item()}



        if self.options.bDebug_visEFT:#g_debugVisualize:    #Debug Visualize input
            for b in range(batch_size):
                #denormalizeImg
                curImgVis = images[b]     #3,224,224
                curImgVis = self.de_normalize_img(curImgVis).cpu().numpy()
                curImgVis = np.transpose( curImgVis , (1,2,0) )*255.0
                curImgVis =curImgVis[:,:,[2,1,0]] 

                #Denormalize image
                curImgVis = np.ascontiguousarray(curImgVis, dtype=np.uint8)
                viewer2D.ImShow(curImgVis, name='rawIm')
                originalImg = curImgVis.copy()

                # curImgVis = viewer2D.Vis_Skeleton_2D_general(gt_keypoints_2d_orig[b,:,:2].cpu().numpy(), gt_keypoints_2d_orig[b,:,2], bVis= False, image=curImgVis)


                pred_keypoints_2d_vis = pred_keypoints_2d[b,:,:2].detach().cpu().numpy()
                pred_keypoints_2d_vis = 0.5 * self.options.img_res * (pred_keypoints_2d_vis + 1)        #49: (25+24) x 3 

                curImgVis = viewer2D.Vis_Skeleton_2D_general(pred_keypoints_2d_vis, bVis= False, image=curImgVis)
                viewer2D.ImShow(curImgVis, scale=2.0, waitTime=1)

                #Get camera pred_params
                pred_camera_vis = pred_camera.detach().cpu().numpy()

                ############### Visualize Mesh ############### 
                pred_vert_vis = pred_vertices[b].detach().cpu().numpy() 
                # meshVertVis = gt_vertices[b].detach().cpu().numpy() 
                # meshVertVis = meshVertVis-pelvis        #centering
                pred_vert_vis *=pred_camera_vis[b,0]
                pred_vert_vis[:,0] += pred_camera_vis[b,1]        #no need +1 (or  112). Rendernig has this offset already
                pred_vert_vis[:,1] += pred_camera_vis[b,2]        #no need +1 (or  112). Rendernig has this offset already
                pred_vert_vis*=112
                pred_meshes = {'ver': pred_vert_vis, 'f': self.smpl.faces}


                opt_vert_vis = opt_vertices[b].detach().cpu().numpy() 
                opt_vert_vis *=pred_camera_vis[b,0]
                opt_vert_vis[:,0] += pred_camera_vis[b,1]        #no need +1 (or  112). Rendernig has this offset already
                opt_vert_vis[:,1] += pred_camera_vis[b,2]        #no need +1 (or  112). Rendernig has this offset already
                opt_vert_vis*=112
                opt_meshes = {'ver': opt_vert_vis, 'f': self.smpl.faces}


                # glViewer.setMeshData([pred_meshes, opt_meshes], bComputeNormal= True)
                glViewer.setMeshData([pred_meshes, opt_meshes], bComputeNormal= True)
                # glViewer.setMeshData([opt_meshes], bComputeNormal= True)


                ############### Visualize Skeletons ############### 
                #Vis pred-SMPL joint
                pred_joints_vis = pred_joints[b,:,:3].detach().cpu().numpy()  #[N,49,3]
                pred_joints_vis = pred_joints_vis.ravel()[:,np.newaxis]
                #Weak-perspective projection
                pred_joints_vis*=pred_camera_vis[b,0]
                pred_joints_vis[::3] += pred_camera_vis[b,1]
                pred_joints_vis[1::3] += pred_camera_vis[b,2]
                pred_joints_vis *=112           #112 == 0.5*224
                glViewer.setSkeleton( [pred_joints_vis])

                # #GT joint
                gt_jointsVis = gt_joints[b,:,:3].cpu().numpy()        #[N,49,3]
                # gt_pelvis = (gt_smpljointsVis[ 25+2,:] + gt_smpljointsVis[ 25+3,:]) / 2
                # gt_smpljointsVis = gt_smpljointsVis- gt_pelvis
                gt_jointsVis = gt_jointsVis.ravel()[:,np.newaxis]
                gt_jointsVis*=pred_camera_vis[b,0]
                gt_jointsVis[::3] += pred_camera_vis[b,1]
                gt_jointsVis[1::3] += pred_camera_vis[b,2]
                gt_jointsVis*=112 
                glViewer.addSkeleton( [gt_jointsVis],jointType='spin')



                # #Vis SMPL's Skeleton
                # gt_smpljointsVis = gt_model_joints[b,:,:3].cpu().numpy()        #[N,49,3]
                # # gt_pelvis = (gt_smpljointsVis[ 25+2,:] + gt_smpljointsVis[ 25+3,:]) / 2
                # # gt_smpljointsVis = gt_smpljointsVis- gt_pelvis
                # gt_smpljointsVis = gt_smpljointsVis.ravel()[:,np.newaxis]
                # gt_smpljointsVis*=pred_camera_vis[b,0]
                # gt_smpljointsVis[::3] += pred_camera_vis[b,1]
                # gt_smpljointsVis[1::3] += pred_camera_vis[b,2]
                # gt_smpljointsVis*=112
                # glViewer.addSkeleton( [gt_smpljointsVis])


                # #Vis GT  joint  (not model (SMPL) joint!!)
                # if has_pose_3d[b]:
                #     gt_jointsVis = gt_model_joints[b,:,:3].cpu().numpy()        #[N,49,3]
                #     # gt_jointsVis = gt_joints[b,:,:3].cpu().numpy()        #[N,49,3]
                #     # gt_pelvis = (gt_jointsVis[ 25+2,:] + gt_jointsVis[ 25+3,:]) / 2
                #     # gt_jointsVis = gt_jointsVis- gt_pelvis

                #     gt_jointsVis = gt_jointsVis.ravel()[:,np.newaxis]
                #     gt_jointsVis*=pred_camera_vis[b,0]
                #     gt_jointsVis[::3] += pred_camera_vis[b,1]
                #     gt_jointsVis[1::3] += pred_camera_vis[b,2]
                #     gt_jointsVis*=112

                #     glViewer.addSkeleton( [gt_jointsVis])
                # # glViewer.show()


                glViewer.setBackgroundTexture(originalImg)
                glViewer.setWindowSize(curImgVis.shape[1], curImgVis.shape[0])
                glViewer.SetOrthoCamera(True)
                glViewer.show(0)

                # continue


        return output, losses

    def train_summaries(self, input_batch, output, losses):
        images = input_batch['img']
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)

        # pred_vertices = output['pred_vertices']
        # opt_vertices = output['opt_vertices']
        # pred_cam_t = output['pred_cam_t']
        # opt_cam_t = output['opt_cam_t']
        # images_pred = self.renderer.visualize_tb(pred_vertices, pred_cam_t, images)
        # images_opt = self.renderer.visualize_tb(opt_vertices, opt_cam_t, images)
        # self.summary_writer.add_image('pred_shape', images_pred, self.step_count)
        # self.summary_writer.add_image('opt_shape', images_opt, self.step_count)
        for loss_name, val in losses.items():
            self.summary_writer.add_scalar(loss_name, val, self.step_count)
    
    def test(self, dataset_test, datasetName):

        self.model.eval()

     
        print(">>> Run test code on Test DB: {}".format(datasetName))
        
        # Setup evaluation dataset
        # datasetName = '3dpw'
        # dataset = BaseDataset(None, datasetName, is_train=False,bMiniTest =True)
        # Run evaluation
        evalLog =  run_evaluation(self.model, datasetName, dataset_test, result_file =None,
                    batch_size=self.options.batch_size,
                    shuffle=False,
                    log_freq=20, num_workers=self.options.num_workers, bVerbose=False)

        mpjpe_mm_3dpw  = evalLog['quant_mpjpe_avg_mm']
        recon_error_mm_3dpw = evalLog['quant_recon_error_avg_mm']

        return recon_error_mm_3dpw
