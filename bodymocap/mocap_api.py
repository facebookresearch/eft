# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import numpy as np
from bodymocap.core import constants
from torchvision.transforms import Normalize

from bodymocap.models import hmr, SMPL, SMPLX
from bodymocap.core import config
from bodymocap.utils.imutils import crop,crop_bboxInfo, process_image_bbox, process_image_keypoints, bbox_from_keypoints
from bodymocap.utils.imutils import convert_smpl_to_bbox, convert_bbox_to_oriIm

from renderer import viewer2D

class BodyMocap:

    def __init__(self, regressor_checkpoint, smpl_dir, device = torch.device('cuda') , bUseSMPLX = False):

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        #Load parametric model (SMPLX or SMPL)
        if bUseSMPLX:
            self.smpl = SMPLX(smpl_dir,
                    batch_size=1,
                    create_transl=False).to(self.device)
        else:
            smplModelPath = smpl_dir + '/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
            self.smpl = SMPL(smplModelPath, batch_size=1, create_transl=False).to(self.device)

        #Load pre-trained neural network 
        self.model_regressor = hmr(config.SMPL_MEAN_PARAMS).to(self.device)
        checkpoint = torch.load(regressor_checkpoint)
        self.model_regressor.load_state_dict(checkpoint['model'], strict=False)
        self.model_regressor.eval()


        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        self.de_normalize_img =  Normalize(mean=[ -constants.IMG_NORM_MEAN[0]/constants.IMG_NORM_STD[0],
                                     -constants.IMG_NORM_MEAN[1]/constants.IMG_NORM_STD[1], -constants.IMG_NORM_MEAN[2]/constants.IMG_NORM_STD[2]],
                                     std=[1/constants.IMG_NORM_STD[0], 1/constants.IMG_NORM_STD[1], 1/constants.IMG_NORM_STD[2]])

    def regress(self, img_original, bbox_XYWH, bExport=True):
        """
            args: 
                img_original: original raw image (BGR order by using cv2.imread)
                bbox_XYWH: bounding box around the target: (minX,minY,width, height)
            outputs:
                Default output:
                    pred_vertices_img:
                    pred_joints_vis_img:
                if bExport==True
                    pred_rotmat
                    pred_betas
                    pred_camera
                    bbox: [bbr[0], bbr[1],bbr[0]+bbr[2], bbr[1]+bbr[3]])
                    bboxTopLeft:  bbox top left (redundant)
                    boxScale_o2n: bbox scaling factor (redundant) 
        """
        img, norm_img, boxScale_o2n, bboxTopLeft, bbox = process_image_bbox(img_original, bbox_XYWH, input_res=constants.IMG_RES)
        if img is None:
            return None

        with torch.no_grad():
            pred_rotmat, pred_betas, pred_camera = self.model_regressor(norm_img.to(self.device))
            pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices
            pred_joints_3d = pred_output.joints

            # img_original = img
            if False:
                #show cropped image
                # curImgVis = img
                curImgVis = self.de_normalize_img(img).cpu().numpy()
                curImgVis = np.transpose( curImgVis , (1,2,0) )*255.0
                curImgVis =curImgVis[:,:,[2,1,0]]
                curImgVis = np.ascontiguousarray(curImgVis, dtype=np.uint8)

                viewer2D.ImShow(curImgVis,name='input_{}'.format(idx))

            pred_vertices = pred_vertices[0].cpu().numpy()
            img =img[:,:,[2,1,0]]
            img = np.ascontiguousarray(img*255, dtype=np.uint8)

            pred_camera = pred_camera.cpu().numpy().ravel()
            camScale = pred_camera[0]# *1.15
            camTrans = pred_camera[1:]

            #Convert mesh to original image space (X,Y are aligned to image)
            pred_vertices_bbox = convert_smpl_to_bbox(pred_vertices, camScale, camTrans)  #SMPL -> 2D bbox
            pred_vertices_img = convert_bbox_to_oriIm(pred_vertices_bbox, boxScale_o2n, bboxTopLeft, img_original.shape[1], img_original.shape[0])       #2D bbox -> original 2D image

            #Convert joint to original image space (X,Y are aligned to image)
            pred_joints_3d = pred_joints_3d[0].cpu().numpy()       #(1,49,3)
            pred_joints_vis = pred_joints_3d[:,:3]    #(49,3)
            pred_joints_vis_bbox = convert_smpl_to_bbox(pred_joints_vis, camScale, camTrans)  #SMPL -> 2D bbox
            pred_joints_vis_img = convert_bbox_to_oriIm(pred_joints_vis_bbox, boxScale_o2n, bboxTopLeft, img_original.shape[1], img_original.shape[0])       #2D bbox -> original 2D image

            ##Output
            predoutput ={}
            predoutput['pred_vertices_img'] = pred_vertices_img #SMPL vertex in image space
            predoutput['pred_joints_img'] = pred_joints_vis_img #SMPL joints in image space
            if bExport:
                predoutput['pred_rotmat'] = pred_rotmat.detach().cpu().numpy()
                predoutput['pred_betas'] = pred_betas.detach().cpu().numpy()
                predoutput['pred_camera'] = pred_camera
                predoutput['bbox_xyxy'] = [bbox_XYWH[0], bbox_XYWH[1], bbox_XYWH[0]+bbox_XYWH[2], bbox_XYWH[1]+bbox_XYWH[3] ]
                predoutput['bboxTopLeft'] = bboxTopLeft
                predoutput['boxScale_o2n'] = boxScale_o2n
         
        return predoutput