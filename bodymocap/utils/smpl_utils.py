# Copyright (c) Facebook, Inc. and its affiliates.

import os
import sys

import torch
from torchvision.transforms import Normalize
import numpy as np
import cv2

from fairmocap.models import SMPL, SMPLX
# from hands.smplx_hand import SMPLX as SMPLX_HAND
from fairmocap.utils.imutils import (conv_bboxinfo_center2topleft, convert_smpl_to_bbox, convert_bbox_to_oriIm)

from renderer import viewer2D,glViewer

from fairmocap.core import constants

de_normalize_img =  Normalize(mean=[ -constants.IMG_NORM_MEAN[0]/constants.IMG_NORM_STD[0]    , -constants.IMG_NORM_MEAN[1]/constants.IMG_NORM_STD[1], -constants.IMG_NORM_MEAN[2]/constants.IMG_NORM_STD[2]], std=[1/constants.IMG_NORM_STD[0], 1/constants.IMG_NORM_STD[1], 1/constants.IMG_NORM_STD[2]])
def denormImg(image_tensor):
    image_np = de_normalize_img(image_tensor).cpu().numpy()
    image_np = np.transpose( image_np , (1,2,0) )*255.0
    image_np =image_np[:,:,[2,1,0]] 

    #Denormalize image
    image_np = np.ascontiguousarray(image_np, dtype=np.uint8)
    # originalImgVis = curImgVis.copy()
    viewer2D.ImShow(image_np, name='denormImg')

    return image_np

    
def visSMPLoutput_bboxSpace(smpl, pred_output, image= None, bUseSMPLX=False, waittime =-1, winsizeScale=4, color=None):
    """
    From prediction output, obtain smpl mesh and joint
    TODO: Currently just assume single batch

    Input:
        pred_output['pred_shape']
        pred_output['pred_rotmat'] or  pred_output['pred_pose']
        pred_output['pred_camera']
        
        if waittime <0, do not call glViwer.show()

    Example: 
        visSMPLoutput(self.smpl, {"pred_rotmat":pred_rotmat, "pred_shape":pred_betas, "pred_camera":pred_camera }, image = images[0])

    """
    smpl_output, smpl_output_bbox = getSMPLoutput_bboxSpace(smpl, pred_output, bUseSMPLX)

    if color is not None:
        smpl_output_bbox['body_mesh']['color'] = color
        
    glViewer.setMeshData([smpl_output_bbox['body_mesh']],bComputeNormal=True)

    glViewer.setSkeleton([np.reshape(smpl_output_bbox['body_joints'], (-1,1)) ], colorRGB =glViewer.g_colorSet['spin'])

    if image is not None:
        if type(image) == torch.Tensor:
            image = denormImg(image)
        smpl_output_bbox['img']= image
        glViewer.setBackgroundTexture(image)
        glViewer.setWindowSize(image.shape[1]*winsizeScale, image.shape[0]*winsizeScale)
        glViewer.SetOrthoCamera(True)

    if waittime>=0:
        glViewer.show(waittime)

    return smpl_output, smpl_output_bbox 

def getSMPLoutput_bboxSpace(smpl, pred_output, bUseSMPLX=False):

    """
    From prediction output, obtain smpl mesh and joint
    Visualize image in bbox space (224x224)

    TODO: Currently just assume single batch

    Input:
        pred_output['pred_shape']
        pred_output['pred_rotmat'] or  pred_output['pred_pose']
        pred_output['pred_camera']
    Output:
        smpl_output_bbox['body_mesh']
        smpl_output_bbox['body_joints']
        smpl_output_bbox['body_joints_vis']         #For glViewer visualization glViewer.setSkeleton( [smpl_output_bbox['body_joints_vis'] ])
        smpl_output_bbox['body_rhand_joints']
        smpl_output_bbox['body_lhand_joints']

    """
    if "pred_rotmat" in pred_output:
        smpl_output = smpl(betas=pred_output['pred_shape'], body_pose=pred_output['pred_rotmat'][:,1:], global_orient=pred_output['pred_rotmat'][:,0].unsqueeze(1), pose2rot=False )
    elif "pred_pose" in pred_output:
        smpl_output = smpl(betas=pred_output['pred_shape'], body_pose=pred_output['pred_pose'][:,3:], global_orient=pred_output['pred_pose'][:,:3])
    else:
        assert False

    pred_camera = pred_output['pred_camera'].detach().cpu().numpy().ravel()
    camScale = pred_camera[0]# *1.15
    camTrans = pred_camera[1:]

    smpl_output_bbox ={}        #Bbox space

    pred_vertices = smpl_output.vertices.detach().cpu().numpy()[0]      #Assume single batch
    pred_vertices = convert_smpl_to_bbox(pred_vertices, camScale, camTrans)  #SMPL -> 2D bbox

    #Visualize Vertices
    body_meshes = {'ver': pred_vertices, 'f': smpl.faces}
    # glViewer.setMeshData([pred_meshes], bComputeNormal= True)
    smpl_output_bbox['body_mesh'] = body_meshes


    #Visualize Vertices
    ours_body_joints_3d = smpl_output.joints.detach().cpu().numpy()[0]
    ours_body_joints_3d_bbox = convert_smpl_to_bbox(ours_body_joints_3d, camScale, camTrans)
    smpl_output_bbox['body_joints'] = ours_body_joints_3d_bbox          #(J=49,3)
    smpl_output_bbox['body_joints_vis'] = ours_body_joints_3d_bbox.ravel()[:,np.newaxis]  #(147,1)          #glViewer form
    

    # smpl_output_img ={}        #Img space
    if bUseSMPLX:
        #Process joint to bbox space
        #Assume single element
        ours_rhand_joints_3d = smpl_output.right_hand_joints.detach().cpu().numpy()[0]
        ours_lhand_joints_3d = smpl_output.left_hand_joints.detach().cpu().numpy()[0]

        rhand_joints_3d_bbox = convert_smpl_to_bbox(ours_rhand_joints_3d, camScale, camTrans)
        lhand_joints_3d_bbox = convert_smpl_to_bbox(ours_lhand_joints_3d, camScale, camTrans)

        smpl_output_bbox['body_rhand_joints'] = rhand_joints_3d_bbox
        smpl_output_bbox['body_lhand_joints'] = lhand_joints_3d_bbox
    
    return smpl_output, smpl_output_bbox

def getSMPLoutput_imgSpace(smpl, pred_output, bboxCenter, bboxScale, imgShape, bUseSMPLX=False):
    """
    From prediction output, obtain smpl mesh and joint
    Aditionally, converting smpl output (vert and joint) to original image space

    TODO: Currently just assume single batch

    Input:
        pred_output['pred_shape']
        pred_output['pred_rotmat']
        pred_output['pred_camera']
    """
    smpl_output, smpl_output_bbox = getSMPLoutput_bboxSpace(smpl, pred_output, bUseSMPLX)

    #Bbox space to image space
    if len(bboxScale.shape)==2:
        bboxScale = bboxScale[0] 
        bboxCenter = bboxCenter[0] 
    bboxScale_o2n, bboxTopLeft_inOriginal = conv_bboxinfo_center2topleft(bboxScale, bboxCenter)
    smpl_output_imgspace ={}
    for k in smpl_output_bbox.keys():
        if "mesh" in k:
            mesh_data = smpl_output_bbox[k]
            newMesh ={}
            newMesh['f'] = mesh_data['f']
            newMesh['ver'] = convert_bbox_to_oriIm(mesh_data['ver'].copy(), bboxScale_o2n, bboxTopLeft_inOriginal, imgShape[1], imgShape[0])       #2D bbox -> original 2D image
            smpl_output_imgspace[k] = newMesh
        else:
            print(k)
            data3D = smpl_output_bbox[k]
            if data3D.shape[1]==1:
                data3D = np.reshape(data3D,(-1,3))
            smpl_output_imgspace[k] = convert_bbox_to_oriIm(data3D, bboxScale_o2n, bboxTopLeft_inOriginal, imgShape[1], imgShape[0])       #2D bbox -> original 2D image\
            # smpl_output_imgspace[k] = np.shape(smpl_output_imgspace[k], (-1,1) )

    return smpl_output, smpl_output_bbox, smpl_output_imgspace

def renderSMPLoutput(rootDir='/home/hjoo/temp/render_general',rendermode='overlaid',rendertype='mesh', imgname='render'):

    if os.path.exists(rootDir)==False:
        os.mkdir(rootDir)

    if rendermode=='side':
        targetFolder =rootDir +'/side'

        if rendertype=='mesh':
            glViewer.g_bShowSkeleton = False
            glViewer.g_bShowMesh = True
            
            targetFolder +="_mesh"
            
        elif rendertype=='skeleton':
            glViewer.g_bShowSkeleton = True
            glViewer.g_bShowMesh = False

            targetFolder +="_skel"
        else:
            assert(False)

        if os.path.exists(targetFolder)==False:
            os.mkdir(targetFolder)

        glViewer.show(1)
        glViewer.setSaveFolderName(targetFolder)
        glViewer.setSaveImgName(imgname)

        glViewer.show_SMPL(bSaveToFile = True, bResetSaveImgCnt = False, countImg = True, mode = 'side', zoom=1600, bShowBG= False)
        # glViewer.show(0)
    elif rendermode=='overlaid':
        targetFolder =rootDir +'/overlaid'
        
        if rendertype=='mesh':
            glViewer.g_bShowSkeleton = False
            glViewer.g_bShowMesh = True
            
            targetFolder +="_mesh"
            
        elif rendertype=='skeleton':
            glViewer.g_bShowSkeleton = True
            glViewer.g_bShowMesh = False

            targetFolder +="_skel"
        elif rendertype=='raw':
            glViewer.g_bShowSkeleton = False
            glViewer.g_bShowMesh = False

            targetFolder +="_raw"
        else:
            assert(False)

        if os.path.exists(targetFolder)==False:
            os.mkdir(targetFolder)
    
        glViewer.setSaveFolderName(targetFolder)
        glViewer.setSaveImgName(imgname)

        glViewer.show_SMPL(bSaveToFile = True, bResetSaveImgCnt = False, countImg = True, mode = 'camera')
    else:
        assert(False)

from tqdm import tqdm
def renderSMPLoutput_merge(rootDir):

    if os.path.exists(rootDir)==False:
        os.mkdir(rootDir)
    
    outputdir = os.path.join(rootDir,'merged')
    if os.path.exists(outputdir)==False:
        os.mkdir(outputdir)

    #Merge videos
    skelDir = os.path.join(rootDir,'overlaid_skel')
    meshDir = os.path.join(rootDir,'overlaid_mesh')
    sideDir = os.path.join(rootDir,'side_mesh')

    fnames = os.listdir(skelDir)
    if True:
        for f in tqdm(sorted(fnames)):
            skelfile = os.path.join(skelDir, f)
            meshfile = os.path.join(meshDir, f)
            sidefile = os.path.join(sideDir, f)

            skel_img = cv2.imread(skelfile)
            mesh_img = cv2.imread(meshfile)
            side_img = cv2.imread(sidefile)

            if True:        #Add iteration number 
                side_img = cv2.putText(side_img, 'Iter={}'.format(f[:-4]), (640,850), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,0,0),2)

            # render_img = cv2.resize(render_img,  )

            # sidefile = os.path.join(inputInfo['sideviewFolder'], f)
            # side_img = cv2.imread(sidefile)
            # side_img = cv2.resize(side_img, (input_img.shape[1], input_img.shape[0]) )

            mergedImg = np.concatenate( (skel_img, mesh_img, side_img), axis=1)
            # viewer2D.ImShow(mergedImg)

            outFileName = os.path.join(outputdir,f)
            cv2.imwrite(outFileName, mergedImg)

    outVideo_fileName = os.path.join(rootDir, os.path.basename(rootDir)) +".mp4"
    # if os.path.exists(outVideo_fileName):
    #     os.remove(outVideo_fileName)
    ffmpeg_cmd = 'ffmpeg -y -f  image2 -framerate 10 -pattern_type glob -i "{0}/*.jpg" -pix_fmt yuv420p  -c:v libx264 {1}'.format(outputdir, outVideo_fileName)
    os.system(ffmpeg_cmd)