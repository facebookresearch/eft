import torch
import os

# import sys
# sys.path.append('/home/hjoo/codes/SPIN/smplify')
from eft.models.smpl import SMPL
from .losses import camera_fitting_loss, camera_fitting_loss_weakperspective, body_fitting_loss, body_fitting_loss_weakperspective, body_prior_loss
from eft.cores import config
from eft.cores import constants

from eft.utils.geometry import weakProjection_gpu

# from eft.utils.timer import Timer
# g_timer = Timer()

# For the GMM prior, we use the GMM implementation of SMPLify-X
# https://github.com/vchoutas/smplify-x/blob/master/smplifyx/prior.py
from .prior import MaxMixturePrior

from eft.utils.smpl_utils import visSMPLoutput_bboxSpace, getSMPLoutput_imgSpace, renderSMPLoutput, renderSMPLoutput_merge

class SMPLify():
    """Implementation of single-stage SMPLify.""" 
    def __init__(self, 
                 step_size=1e-2,
                 batch_size=66,
                 num_iters=100,
                 focal_length=5000,
                 device=torch.device('cuda')):

        # Store options
        self.device = device
        self.focal_length = focal_length
        self.step_size = step_size

        # Ignore the the following joints for the fitting process
        ign_joints = ['OP Neck', 'OP RHip', 'OP LHip', 'Right Hip', 'Left Hip']
        self.ign_joints = [constants.JOINT_IDS[i] for i in ign_joints]
        self.num_iters = num_iters
        # GMM pose prior
        
        self.pose_prior = MaxMixturePrior(prior_folder='./extradata/spin',
                                          num_gaussians=8,
                                          dtype=torch.float32).to(device)
        # Load SMPL model
        self.smpl = SMPL(config.SMPL_MODEL_DIR,
                         batch_size=batch_size,
                         create_transl=False).to(self.device)

    def __call__(self, init_pose, init_betas, init_cam_t, camera_center, keypoints_2d):
        """Perform body fitting.
        Input:
            init_pose: SMPL pose estimate
            init_betas: SMPL betas estimate
            init_cam_t: Camera translation estimate
            camera_center: Camera center location
            keypoints_2d: Keypoints used for the optimization
        Returns:
            vertices: Vertices of optimized shape
            joints: 3D joints of optimized shape
            pose: SMPL pose parameters of optimized shape
            betas: SMPL beta parameters of optimized shape
            camera_translation: Camera translation
            reprojection_loss: Final joint reprojection loss
        """

        batch_size = init_pose.shape[0]

        # Make camera translation a learnable parameter
        camera_translation = init_cam_t.clone()

        # Get joint confidence
        joints_2d = keypoints_2d[:, :, :2]
        joints_conf = keypoints_2d[:, :, -1]

        # Split SMPL pose to body pose and global orientation
        body_pose = init_pose[:, 3:].detach().clone()
        global_orient = init_pose[:, :3].detach().clone()
        betas = init_betas.detach().clone()

        # Step 1: Optimize camera translation and body orientation
        # Optimize only camera translation and body orientation
        body_pose.requires_grad=False
        betas.requires_grad=False
        global_orient.requires_grad=True
        camera_translation.requires_grad = True

        camera_opt_params = [global_orient, camera_translation]
        camera_optimizer = torch.optim.Adam(camera_opt_params, lr=self.step_size, betas=(0.9, 0.999))

        bSMPLify_noCamOptFirst  = False
        if bSMPLify_noCamOptFirst==False:       #No cam optimization
            for i in range(self.num_iters):
                smpl_output = self.smpl(global_orient=global_orient,
                                        body_pose=body_pose,
                                        betas=betas)
                model_joints = smpl_output.joints
                loss = camera_fitting_loss(model_joints, camera_translation,
                                        init_cam_t, camera_center,
                                        joints_2d, joints_conf, focal_length=self.focal_length)
                camera_optimizer.zero_grad()
                loss.backward()
                camera_optimizer.step()
                # print(loss)

                # visSMPLoutput(self.smpl, {"pred_pose":body_pose, "pred_shape":betas, "pred_camera":pred_camera })

        # Fix camera translation after optimizing camera
        camera_translation.requires_grad = False

        # Step 2: Optimize body joints
        # Optimize only the body pose and global orientation of the body
        body_pose.requires_grad=True
        betas.requires_grad=True
        global_orient.requires_grad=True
        
        if bSMPLify_noCamOptFirst:
            camera_translation.requires_grad = True
        else:
            camera_translation.requires_grad = False

        body_opt_params = [body_pose, betas, global_orient]

        # For joints ignored during fitting, set the confidence to 0
        joints_conf[:, self.ign_joints] = 0.
        
        body_optimizer = torch.optim.Adam(body_opt_params, lr=self.step_size, betas=(0.9, 0.999))
        for i in range(self.num_iters):
            smpl_output = self.smpl(global_orient=global_orient,
                                    body_pose=body_pose,
                                    betas=betas)
            model_joints = smpl_output.joints
            loss = body_fitting_loss(body_pose, betas, model_joints, camera_translation, camera_center,
                                     joints_2d, joints_conf, self.pose_prior,
                                     focal_length=self.focal_length)
            body_optimizer.zero_grad()
            loss.backward()
            body_optimizer.step()

        # Get final loss value
        with torch.no_grad():
            smpl_output = self.smpl(global_orient=global_orient,
                                    body_pose=body_pose,
                                    betas=betas, return_full_pose=True)
            model_joints = smpl_output.joints
            reprojection_loss = body_fitting_loss(body_pose, betas, model_joints, camera_translation, camera_center,
                                                  joints_2d, joints_conf, self.pose_prior,
                                                  focal_length=self.focal_length,
                                                  output='reprojection')

        vertices = smpl_output.vertices.detach()
        joints = smpl_output.joints.detach()
        pose = torch.cat([global_orient, body_pose], dim=-1).detach()
        betas = betas.detach()

        return vertices, joints, pose, betas, camera_translation, reprojection_loss


    def run_withWeakProj(self, init_pose, init_betas, init_cameras, camera_center, keypoints_2d, bDebugVis= False, bboxInfo=None, imagevis = None, 
                        ablation_smplify_noCamOptFirst= False, ablation_smplify_noPrior = False
                        ):
        """Perform body fitting.
        Input:
            init_pose: SMPL pose estimate
            init_betas: SMPL betas estimate
            init_cam_t: Camera translation estimate
            camera_center: Camera center location
            keypoints_2d: Keypoints used for the optimization
        Returns:
            vertices: Vertices of optimized shape
            joints: 3D joints of optimized shape
            pose: SMPL pose parameters of optimized shape
            betas: SMPL beta parameters of optimized shape
            camera_translation: Camera translation
            reprojection_loss: Final joint reprojection loss
        """

        # batch_size = init_pose.shape[0]

        # Make camera translation a learnable parameter
        # camera_translation = init_cam_t.clone()

        # Get joint confidence
        joints_2d = keypoints_2d[:, :, :2]
        joints_conf = keypoints_2d[:, :, -1]

        # Split SMPL pose to body pose and global orientation
        body_pose = init_pose[:, 3:].detach().clone()
        global_orient = init_pose[:, :3].detach().clone()
        betas = init_betas.detach().clone()
        camera = init_cameras.detach().clone()

        # Step 1: Optimize camera translation and body orientation
        # Optimize only camera translation and body orientation
        body_pose.requires_grad=False
        betas.requires_grad=False
        global_orient.requires_grad=True
        camera.requires_grad = True

        camera_opt_params = [global_orient, camera]
        camera_optimizer = torch.optim.Adam(camera_opt_params, lr=self.step_size, betas=(0.9, 0.999))

        # joints_2d
        # for i in range(self.num_iters*10):
        # g_timer.tic()    
        if ablation_smplify_noCamOptFirst==False:
            for i in range(self.num_iters):
                smpl_output = self.smpl(global_orient=global_orient,
                                        body_pose=body_pose,
                                        betas=betas)
                model_joints = smpl_output.joints
                loss = camera_fitting_loss_weakperspective(model_joints, camera,
                                        init_cameras, joints_2d, joints_conf)
                camera_optimizer.zero_grad()
                loss.backward()
                camera_optimizer.step()
                # print(loss)

                #Render
                if False:   
                    from renderer import glViewer
                    body_pose_all = torch.cat([global_orient, body_pose], dim=-1)       #[N,72]
                    smpl_output, smpl_output_bbox  = visSMPLoutput_bboxSpace(self.smpl, {"pred_pose":body_pose_all, "pred_shape":betas, "pred_camera":camera }, color= glViewer.g_colorSet['spin'], image = imagevis)
                    glViewer.show(1)

                    #Render
                    if False:
                        root_imgname = os.path.basename(bboxInfo['imgname'])[:-4]
                        renderRoot=f'/home/hjoo/temp/render_eft/smplify_{root_imgname}'
                        imgname='{:04d}'.format(i)
                        renderSMPLoutput(renderRoot,'overlaid','mesh',imgname=imgname)
                        renderSMPLoutput(renderRoot,'overlaid','skeleton',imgname=imgname)
                        renderSMPLoutput(renderRoot,'side','mesh',imgname=imgname)

        # g_timer.toc(average =True, bPrint=True,title="Single Camera Optimization")

        # Fix camera translation after optimizing camera
        # camera.requires_grad = False

        # Step 2: Optimize body joints
        # Optimize only the body pose and global orientation of the body
        body_pose.requires_grad=True
        betas.requires_grad=True
        global_orient.requires_grad=True
        
        if ablation_smplify_noCamOptFirst==False:       #Original from SPIN
            camera.requires_grad = False
            body_opt_params = [body_pose, betas, global_orient]
        else:       #New
            camera.requires_grad = True
            body_opt_params = [body_pose, betas, global_orient, camera]

        # For joints ignored during fitting, set the confidence to 0
        joints_conf[:, self.ign_joints] = 0.
        
        # g_timer.tic()    
        
        body_optimizer = torch.optim.Adam(body_opt_params, lr=self.step_size, betas=(0.9, 0.999))
        for i in range(self.num_iters):
            smpl_output = self.smpl(global_orient=global_orient,
                                    body_pose=body_pose,
                                    betas=betas)

            g_timer.tic()
            model_joints = smpl_output.joints
            # loss = body_fitting_loss(body_pose, betas, model_joints, camera, camera_center,
            #                          joints_2d, joints_conf, self.pose_prior,
            #                          focal_length=self.focal_length)

            if ablation_smplify_noPrior:
                # print('ablation_smplify_noPrior')
                loss, reprojection_loss = body_fitting_loss_weakperspective(body_pose, betas, model_joints, camera,
                                         joints_2d, joints_conf, self.pose_prior,angle_prior_weight=0)#,pose_prior_weight=0) # pose_prior_weight=0)#, angle_prior_weight=0)
            else:
                loss, reprojection_loss = body_fitting_loss_weakperspective(body_pose, betas, model_joints, camera,
                                         joints_2d, joints_conf, self.pose_prior)

            #Stop with sufficiently small pixel error
            if reprojection_loss.mean()<2.0:
                break

            body_optimizer.zero_grad()
            loss.backward()
            body_optimizer.step()

            g_timer.toc(average =False, bPrint=True,title="SMPLify iter")


            if bDebugVis:
                from renderer import glViewer
                body_pose_all = torch.cat([global_orient, body_pose], dim=-1)       #[N,72]
                smpl_output, smpl_output_bbox  = visSMPLoutput_bboxSpace(self.smpl, {"pred_pose":body_pose_all, "pred_shape":betas, "pred_camera":camera }, waittime =1, color= glViewer.g_colorSet['spin'], image = imagevis)

                # #Render
                if False:
                    root_imgname = os.path.basename(bboxInfo['imgname'])[:-4]
                    renderRoot=f'/home/hjoo/temp/render_eft/smplify_{root_imgname}'
                    imgname='{:04d}'.format(i+ self.num_iters)
                    renderSMPLoutput(renderRoot,'overlaid','mesh',imgname=imgname)
                    renderSMPLoutput(renderRoot,'overlaid','skeleton',imgname=imgname)
                    renderSMPLoutput(renderRoot,'side','mesh',imgname=imgname)

        # g_timer.toc(average =True, bPrint=True,title="Whole body optimization")

        # Get final loss value
        with torch.no_grad():
            smpl_output = self.smpl(global_orient=global_orient,
                                    body_pose=body_pose,
                                    betas=betas, return_full_pose=True)
            model_joints = smpl_output.joints
            # reprojection_loss = body_fitting_loss(body_pose, betas, model_joints, camera_translation, camera_center,
            #                                       joints_2d, joints_conf, self.pose_prior,
            #                                       focal_length=self.focal_length,
            #                                       output='reprojection')

            if ablation_smplify_noPrior:
                reprojection_loss = body_fitting_loss_weakperspective(body_pose, betas, model_joints, camera,
                                        joints_2d, joints_conf, self.pose_prior,angle_prior_weight=0,#pose_prior_weight=0, # pose_prior_weight=0 angle_prior_weight=0,
                                                    output='reprojection')
                                                    
            else: #Original
                reprojection_loss = body_fitting_loss_weakperspective(body_pose, betas, model_joints, camera,
                                        joints_2d, joints_conf, self.pose_prior,
                                                    output='reprojection')

             

        vertices = smpl_output.vertices.detach()
        joints = smpl_output.joints.detach()
        pose = torch.cat([global_orient, body_pose], dim=-1).detach()
        betas = betas.detach()


        if bDebugVis:
            from renderer import glViewer
            body_pose_all = torch.cat([global_orient, body_pose], dim=-1)       #[N,72]
            visSMPLoutput_bboxSpace(self.smpl, {"pred_pose":body_pose_all, "pred_shape":betas, "pred_camera":camera },color=glViewer.g_colorSet['spin'])
            
            if False:
                glViewer.show()
            elif False:   #Render to Files in original image space
                    bboxCenter = bboxInfo['bboxCenter']
                    bboxScale = bboxInfo['bboxScale']
                    imgname = bboxInfo['imgname']

                    #Get Skeletons
                    import cv2
                    img_original = cv2.imread( imgname )
                    # viewer2D.ImShow(img_original, waitTime=0)
                    imgShape = img_original.shape[:2]
                    smpl_output, smpl_output_bbox, smpl_output_imgspace  = getSMPLoutput_imgSpace(self.smpl, {"pred_pose":body_pose_all, "pred_shape":betas, "pred_camera":camera },
                                                                    bboxCenter, bboxScale, imgShape)

                    glViewer.setBackgroundTexture(img_original)       #Vis raw video as background
                    glViewer.setWindowSize(img_original.shape[1]*2, img_original.shape[0]*2)       #Vis raw video as background
                    smpl_output_imgspace['body_mesh']['color'] = glViewer.g_colorSet['spin']
                    glViewer.setMeshData([smpl_output_imgspace['body_mesh']], bComputeNormal = True )       #Vis raw video as background
                    glViewer.setSkeleton([])

                    imgname = os.path.basename(imgname)[:-4]
                    fileName = "smplify_{0}_{1:04d}".format(imgname, 0)

                    # rawImg = cv2.putText(rawImg,data['subjectId'],(100,100), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0),2)
                    glViewer.render_on_image('/home/hjoo/temp/render_eft', fileName, img_original, scaleFactor=2)

                    glViewer.show()
            else:
                root_imgname = os.path.basename(bboxInfo['imgname'])[:-4]
                # renderRoot=f'/home/hjoo/temp/render_eft/smplify_{root_imgname}'
                renderRoot=f'/home/hjoo/temp/render_rebuttal/smplify_{root_imgname}'
                
                imgname='{:04d}'.format(i+ self.num_iters)
                renderSMPLoutput(renderRoot,'overlaid','mesh',imgname=imgname)
                renderSMPLoutput(renderRoot,'overlaid','skeleton',imgname=imgname)
                renderSMPLoutput(renderRoot,'side','mesh',imgname=imgname)
                renderSMPLoutput_merge(renderRoot)

        return vertices, joints, pose, betas, camera, reprojection_loss
        

    def get_fitting_loss(self, pose, betas, cam_t, camera_center, keypoints_2d):
        """Given body and camera parameters, compute reprojection loss value.
        Input:
            pose: SMPL pose parameters
            betas: SMPL beta parameters
            cam_t: Camera translation
            camera_center: Camera center location
            keypoints_2d: Keypoints used for the optimization
        Returns:
            reprojection_loss: Final joint reprojection loss
        """

        batch_size = pose.shape[0]

        # Get joint confidence
        joints_2d = keypoints_2d[:, :, :2]
        joints_conf = keypoints_2d[:, :, -1]
        # For joints ignored during fitting, set the confidence to 0
        joints_conf[:, self.ign_joints] = 0.

        # Split SMPL pose to body pose and global orientation
        body_pose = pose[:, 3:]
        global_orient = pose[:, :3]

        with torch.no_grad():
            smpl_output = self.smpl(global_orient=global_orient,
                                    body_pose=body_pose,
                                    betas=betas, return_full_pose=True)
            model_joints = smpl_output.joints
            reprojection_loss = body_fitting_loss(body_pose, betas, model_joints, cam_t, camera_center,
                                                  joints_2d, joints_conf, self.pose_prior,
                                                  focal_length=self.focal_length,
                                                  output='reprojection')

        return reprojection_loss

    
    def get_prior_loss(self, pose, betas):
        """Given body and camera parameters, compute reprojection loss value.
        Input:
            pose: SMPL pose parameters
            betas: SMPL beta parameters
            cam_t: Camera translation
            camera_center: Camera center location
            keypoints_2d: Keypoints used for the optimization
        Returns:
            reprojection_loss: Final joint reprojection loss
        """

        batch_size = pose.shape[0]

        # Split SMPL pose to body pose and global orientation
        body_pose = pose[:, 3:]
        global_orient = pose[:, :3]

        reprojection_loss = body_prior_loss(body_pose, betas, self.pose_prior)

        return reprojection_loss

