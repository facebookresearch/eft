# Copyright (c) Facebook, Inc. and its affiliates.

import os
from os.path import join
from os import listdir
# import json
import numpy as np

import cv2
import pickle
import torch
from eft.models import SMPL

from eft.utils.imutils import crop, crop_bboxInfo
from eft.utils.imutils import convert_smpl_to_bbox, convert_bbox_to_oriIm, conv_bboxinfo_bboxXYXY
from renderer import viewer2D#, glViewer, glRenderer
from renderer import meshRenderer #glRenderer
from renderer import denseposeRenderer #glRenderer
# from renderer import torch3dRenderer #glRenderer

import argparse
import json

## Constant
BBOX_IMG_RES = 224

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir',default="/run/media/hjoo/disk/data/coco/train2014", type=str , help='Folder path where input image files exist')
parser.add_argument('--fit_data',default="eft_fit/COCO2014-Part-ver01.json", type=str, help='EFT data json fortmat')
parser.add_argument('--smpl_dir',default="./extradata/smpl", type=str , help='Folder path where smpl pkl files exist')
# parser.add_argument('--onbbox',action="store_true", help="Show the 3D pose on bbox space")
parser.add_argument('--rendermode',default="geo", help="Choose among geo, normal, densepose")
parser.add_argument('--render_dir',default="render_eft", help="Folder to save rendered images")
parser.add_argument('--waitforkeys',action="store_true", help="If true, it will pasue after each visualizing each sample, waiting for any key pressed")
# parser.add_argument('--turntable',action="store_true", help="If true, show turn table views")
# parser.add_argument('--bShowMultiSub',action="store_true", help='If True, show multi-person outputs at each time. Default, visualize a single person at each time')
parser.add_argument('--cocoAnnotFile',default='/run/media/hjoo/disk/data/coco/annotations/person_keypoints_train2014.json', type=str , help='COCO 2014 annotation file path')

args = parser.parse_args()



def getRenderer(ren_type='geo'):
    """
    Choose renderer type
    geo: phong-shading
    denspose: densepose IUV
    normal: normal map
    torch3d: via pytorch3d TODO
    """

    if ren_type=='geo':
        renderer = meshRenderer.meshRenderer()
        renderer.setRenderMode('geo')
        
    elif ren_type=='normal':
        renderer = meshRenderer.meshRenderer()
        renderer.setRenderMode('normal')

    elif ren_type=='densepose':
        renderer = denseposeRenderer.denseposeRenderer()

    elif  ren_type=='torch3d':
        renderer = torch3dRenderer.torch3dRenderer()
    else:
        assert False

    renderer.offscreenMode(True)
    renderer.bAntiAliasing= False
    return renderer

def conv_3djoint_2djoint(smpl_joints_3d_vis, imgshape):

    smpl_joints_2d_vis = smpl_joints_3d_vis[:,:2]       #3D is in camera comaera coordinate with origin on the image center
    smpl_joints_2d_vis[:,0] += imgshape[1]*0.5      #Offset to move the origin on the top left
    smpl_joints_2d_vis[:,1] += imgshape[0]*0.5

    return smpl_joints_2d_vis
    

def visEFT_singleSubject(renderer):

    bStopForEachSample = args.waitforkeys      #if True, it will wait for any key pressed to move to the next sample
    # bShowTurnTable = args.turntable

    inputData = args.fit_data
    imgDir = args.img_dir

    #Load SMPL model
    smplModelPath = args.smpl_dir + '/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
    smpl = SMPL(smplModelPath, batch_size=1, create_transl=False)

    print("Loading coco annotation from:{}".format(args.cocoAnnotFile))
    assert os.path.exists(args.cocoAnnotFile)
    cocoAnnotDic = loadCOCOAnnot(args.cocoAnnotFile)
    
    #Load EFT fitting data
    print(f"Loading EFT data from {inputData}")
    if os.path.exists(inputData):
        with open(inputData,'r') as f:
            eft_data = json.load(f)
            print("EFT data: ver {}".format(eft_data['ver']))
            eft_data_all = eft_data['data']
    else:
        print(f"ERROR:: Cannot find EFT data: {inputData}")
        assert False


    #Visualize each EFT Fitting output
    for idx, eft_data in enumerate(eft_data_all):

        #Get raw image path
        imgFullPath = eft_data['imageName']
        imgName = os.path.basename(imgFullPath)
        imgFullPath =os.path.join(imgDir, imgName)
        if os.path.exists(imgFullPath) ==False:
            print(f"Img path is not valid: {imgFullPath}")
            assert False
        rawImg = cv2.imread(imgFullPath)
        print(f'Input image: {imgFullPath}')

        #EFT data
        bbox_scale = eft_data['bbox_scale']
        bbox_center = eft_data['bbox_center']

        pred_camera = np.array(eft_data['parm_cam'])
        pred_betas = np.reshape(np.array( eft_data['parm_shape'], dtype=np.float32), (1,10) )     #(10,)
        pred_betas = torch.from_numpy(pred_betas)

        pred_pose_rotmat = np.reshape( np.array( eft_data['parm_pose'], dtype=np.float32), (1,24,3,3)  )        #(24,3,3)
        pred_pose_rotmat = torch.from_numpy(pred_pose_rotmat)

        keypoint_2d_validity = eft_data['joint_validity_openpose18']

        #COCO only. Annotation index
        print("COCO annotId: {}".format(eft_data['annotId']))
        annot = cocoAnnotDic[eft_data['annotId']]
        print(annot['bbox'])

        ########################
        #Visualize COCO annotation
        annot_keypoint = np.reshape(np.array(annot['keypoints'], dtype=np.float32), (-1,3))     #17,3
        rawImg = viewer2D.Vis_Skeleton_2D_coco(annot_keypoint[:,:2],annot_keypoint[:,2], image=rawImg)
        rawImg = viewer2D.Vis_Bbox(rawImg, annot['bbox'],color=(0,255,0))

        #Get SMPL mesh and joints from SMPL parameters
        smpl_output = smpl(betas=pred_betas, body_pose=pred_pose_rotmat[:,1:], global_orient=pred_pose_rotmat[:,[0]], pose2rot=False)
        smpl_vertices = smpl_output.vertices.detach().cpu().numpy()[0]
        smpl_joints_3d = smpl_output.joints.detach().cpu().numpy()[0]

        #Crop image using cropping information
        croppedImg, boxScale_o2n, bboxTopLeft = crop_bboxInfo(rawImg, bbox_center, bbox_scale, (BBOX_IMG_RES, BBOX_IMG_RES) )

        ########################
        # Visualization of EFT 
        ########################

        # Visualize 2D image
        if True:
            viewer2D.ImShow(rawImg, name='rawImg', waitTime=1)      #You should press any key 
            viewer2D.ImShow(croppedImg, name='croppedImg', waitTime=1)

            #Convert bbox_center, bbox_scale --> bbox_xyxy
            bbox_xyxy = conv_bboxinfo_bboxXYXY(bbox_scale,bbox_center)
            img_bbox = viewer2D.Vis_Bbox_minmaxPt(rawImg.copy(),bbox_xyxy[:2], bbox_xyxy[2:])
            viewer2D.ImShow(img_bbox, name='img_bbox', waitTime=1)

        # Visualization Mesh
        if True:    
            camParam_scale = pred_camera[0]
            camParam_trans = pred_camera[1:]
            pred_vert_vis = smpl_vertices
            smpl_joints_3d_vis = smpl_joints_3d

            if True:#args.onbbox:
                pred_vert_vis = convert_smpl_to_bbox(pred_vert_vis, camParam_scale, camParam_trans)
                smpl_joints_3d_vis = convert_smpl_to_bbox(smpl_joints_3d_vis, camParam_scale, camParam_trans)
                renderer.setBackgroundTexture(croppedImg)
                renderer.setViewportSize(croppedImg.shape[1], croppedImg.shape[0])
           
            pred_meshes = {'ver': pred_vert_vis, 'f': smpl.faces}
            v = pred_meshes['ver'] 
            f = pred_meshes['f']

            #Visualize in the original image space
            renderer.set_mesh(v,f)
            renderer.showBackground(True)
            renderer.setWorldCenterBySceneCenter()
            renderer.setCameraViewMode("cam")

            renderer.setViewportSize(croppedImg.shape[1], croppedImg.shape[0])
            renderer.display()
            renderImg = renderer.get_screen_color_ibgr()
            viewer2D.ImShow(renderImg,waitTime=1)

        # Visualization Mesh on side view
        if True:
            renderer.showBackground(False)
            renderer.setWorldCenterBySceneCenter()
            renderer.setCameraViewMode("side")

            renderer.setViewportSize(croppedImg.shape[1], croppedImg.shape[0])
            renderer.display()
            sideImg = renderer.get_screen_color_ibgr()        #Overwite on rawImg
            viewer2D.ImShow(sideImg,waitTime=1)
            
            sideImg = cv2.resize(sideImg, (renderImg.shape[1], renderImg.shape[0]) )

        #Visualize camera view and side view
        saveImg = np.concatenate( (renderImg,sideImg), axis =1)

        if bStopForEachSample:
            viewer2D.ImShow(saveImg,waitTime=0) #waitTime=0 means that it will wait for any key pressed
        else:
            viewer2D.ImShow(saveImg,waitTime=1)
        
        #Save the rendered image to files
        if False:    
            if os.path.exists(args.render_dir) == False:
                os.mkdir(args.render_dir)
            render_output_path = args.render_dir + '/render_{:08d}.jpg'.format(idx)
            print(f"Save to {render_output_path}")
            cv2.imwrite(render_output_path, saveImg)


def loadCOCOAnnot(cocoAnnotFile):
    
    with open(cocoAnnotFile,'rb' ) as f:
        json_data = json.load(f)

    imgs = {}
    for img in json_data['images']:
        imgs[img['id']] = img

    cocoAnnot={}
    for annot in json_data['annotations']:
        annot_id = annot['id']
        cocoAnnot[annot_id] = annot

    return cocoAnnot
    

if __name__ == '__main__':

    renderer = getRenderer(args.rendermode)

    #visualize single human only
    visEFT_singleSubject(renderer)