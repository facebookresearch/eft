# Copyright (c) Facebook, Inc. and its affiliates.

import os
from os.path import join
import json
import numpy as np

import cv2
from os import listdir
import pickle

from fairmocap.core import constants 
from fairmocap.core import config 
import torch



from fairmocap.utils.imutils import crop, crop_bboxInfo
from fairmocap.models import SMPL

from fairmocap.utils.imutils import convert_smpl_to_bbox, convert_bbox_to_oriIm
from renderer import viewer2D,glViewer, denseposeRenderer #glRenderer


# g_bExportDPOut = True   #Save to PKL file

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir',default="/run/media/hjoo/disk/data/coco/train2014", type=str , help='dir path where input image files exist')
parser.add_argument('--fit_dir',default="/run/media/hjoo/disk/data/cvpr2020_eft_researchoutput/0_SPIN/0_exemplarOutput/01-23_coco_with8143_annotId", type=str, help='dir path where fitting pkl files exist')
parser.add_argument('--smpl_dir',default="./extradata/smpl", type=str , help='dir path where smpl pkl files exist')
parser.add_argument('--bNoVis',action="store_true", help="Set if you do not want to show the output on screen (e.g., headless rendering)")
parser.add_argument('--export',action="store_true", help="Set if you want to export densepose label output from SMPL fits")
parser.add_argument('--colormode',default="u", help="choose among u, v, seg")
args = parser.parse_args()

# g_bVis = True       #Visualize rendering output to screen


if __name__ == '__main__':

    #Set the following
    # inputDir = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-08_cocoplus_with8143'
    # inputDir = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/01-24_cocoall_with8143_annotId'
    # inputDir = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/01-23_coco_with8143_annotId'
    # imgDir = '/run/media/hjoo/disk/data/coco/train2014'
    # smplModelDir = './smpl/'
    inputDir = args.fit_dir
    imgDir = args.img_dir
    smplModelDir = args.smpl_dir

    outputFolder = os.path.basename(inputDir) + '_dpOut'
    outputFolder =os.path.join('/run/media/hjoo/disk/data/eftout/',outputFolder)

    # # devfair
    # inputDir = './eft/11-08_coco_with8143'
    # inputDir = '/private/home/hjoo/spinOut/01-23_coco_8143_annotId'
    # imgDir = '/private/home/hjoo/data/coco/train2014'

    smpl = SMPL(smplModelDir, batch_size=1, create_transl=False)
    fileList  = listdir(inputDir)       #Check all fitting files

    render = denseposeRenderer.denseposeRenderer()
    render.offscreenMode(True)
    render.bAntiAliasing= False


    print(">> Found {} files in the fitting folder {}".format(len(fileList), inputDir))
    totalCnt =0
    erroneousCnt =0
    for idx, f in enumerate(sorted(fileList)):
        
        if "_init" in f:
            continue
        #Load
        fileFullPath = join(inputDir, f)
        with open(fileFullPath,'rb') as f:
            data = pickle.load(f)
        imgFullPath = data['imageName'][0]
        imgFullPath =os.path.join(imgDir, os.path.basename(imgFullPath) )

        scale = data['scale'][0]
        center = data['center'][0]
        print("annotId: {}".format(data['annotId']))

        ours_betas = torch.from_numpy(data['pred_shape'])
        ours_pose_rotmat = torch.from_numpy(data['pred_pose_rotmat'])
        spin_betas = torch.from_numpy(data['opt_beta'])
        
        #Compute 2D reprojection error
        # if not (data['loss_keypoints_2d']<0.0001 or data['loss_keypoints_2d']>0.001 :
        #     continue
        maxBeta = abs(torch.max( abs(ours_betas)).item())

        if data['loss_keypoints_2d']>0.0005 or maxBeta>3:
            erroneousCnt +=1
        
        print(">>> loss2d: {}, maxBeta: {}".format( data['loss_keypoints_2d'],maxBeta) )

        spin_pose = torch.from_numpy(data['opt_pose'])
        pred_camera_vis = data['pred_camera']
 
        assert os.path.exists(imgFullPath)
        rawImg = cv2.imread(imgFullPath)
        print(imgFullPath)

        #Crop image
        # croppedImg = crop(rawImg, center, scale, 
        #               [constants.IMG_RES, constants.IMG_RES])

        croppedImg, boxScale_o2n, bboxTopLeft = crop_bboxInfo(rawImg, center, scale, (constants.IMG_RES, constants.IMG_RES) )


        #Visualize 2D image
        if True:
            viewer2D.ImShow(rawImg, name='rawImg', waitTime=10)      #You should press any key 
            viewer2D.ImShow(croppedImg, name='croppedImg', waitTime=10)

        ours_output = smpl(betas=ours_betas, body_pose=ours_pose_rotmat[:,1:], global_orient=ours_pose_rotmat[:,0].unsqueeze(1), pose2rot=False )
        ours_vertices = ours_output.vertices.detach().cpu().numpy() 
        ours_joints_3d = ours_output.joints.detach().cpu().numpy() 

        if args.bNoVis == False:    #Visualization via densepose renderer

            ############### Visualize Mesh ############### 
            b=0
            camParam_scale = pred_camera_vis[b,0]
            camParam_trans = pred_camera_vis[b,1:]

            pred_vert_vis = ours_vertices[b].copy()
            pred_vert_vis = convert_smpl_to_bbox(pred_vert_vis, camParam_scale, camParam_trans)

            #From cropped space to original
            pred_vert_vis = convert_bbox_to_oriIm(pred_vert_vis, boxScale_o2n, bboxTopLeft, rawImg.shape[1], rawImg.shape[0]) 
            pred_meshes = {'ver': pred_vert_vis, 'f': smpl.faces}
            # glViewer.setMeshData([pred_meshes], bComputeNormal= True)
            # v, f, color= loadSMPL()
            v = pred_meshes['ver'] 
            # f = pred_meshes['f']

            # v, f, color= glRenderer.loadSMPL()
            # v =v[0]
            
            import time
            start = time.time()
            # render.set_mesh(v,f)
            render.set_smpl_mesh(v, colormode=args.colormode)
            render.setBackgroundTexture(rawImg)

            # render.setWindowSize(rawImg.shape[1], rawImg.shape[0])
            render.setViewportSize(rawImg.shape[1], rawImg.shape[0])
            # render.show_once()
            render.display()
            
            out_all_f = render.get_screen_color()
            end = time.time()
            print("Time: {}".format(end-start))

            # out_all_f = render.get_z_value()
            viewer2D.ImShow(out_all_f,waitTime=0)
            # cv2.imwrite('/home/hjoo/temp/render_general/test_{:08d}.jpg'.format(idx), out_all_f*255.0)
            # cv2.imwrite('testout3/test_{:08d}_int.jpg'.format(idx), rawImg)
            # cv2.imwrite('tempout/test_{:08d}.jpg'.format(idx), out_all_f*255.0)

        if False:#args.export:    #Export to files

            
            if os.path.exists(outputFolder)==False:
                os.mkdir(outputFolder)

            dp_output ={} 

            ############### Visualize Mesh ############### 
            b=0
            camParam_scale = pred_camera_vis[b,0]
            camParam_trans = pred_camera_vis[b,1:]

            pred_vert_vis = ours_vertices[b].copy()
            pred_vert_vis = convert_smpl_to_bbox(pred_vert_vis, camParam_scale, camParam_trans)

            #From cropped space to original
            pred_vert_vis = convert_bbox_to_oriIm(pred_vert_vis, boxScale_o2n, bboxTopLeft, rawImg.shape[1], rawImg.shape[0]) 
            pred_meshes = {'ver': pred_vert_vis, 'f': smpl.faces}
            # glViewer.setMeshData([pred_meshes], bComputeNormal= True)
            # v, f, color= loadSMPL()
            v = pred_meshes['ver'] 
            # f = pred_meshes['f']

            # v, f, color= glRenderer.loadSMPL()
            # v =v[0]
            import time
            start = time.time()
            # render.set_mesh(v,f)
            # render.setWindowSize(rawImg.shape[1], rawImg.shape[0])
            render.setViewportSize(rawImg.shape[1], rawImg.shape[0])
            render.set_smpl_mesh(v,colormode='u')
            render.display()
            out_all_f = render.get_screen_color()
            dp_output['u'] = out_all_f[:,:,0]

            render.set_smpl_mesh(v,colormode='v')
            render.display()
            out_all_f = render.get_screen_color()
            dp_output['v'] = out_all_f[:,:,0]

            render.set_smpl_mesh(v,colormode='seg')
            render.display()
            out_all_f = render.get_screen_color()
            dp_output['seg'] = out_all_f[:,:,0]
            # dp_output['mask '] = dp_output['seg'][:,:,0]<0.99       #True: forground

            dp_output['imageName']  = os.path.basename(data['imageName'][0])
            dp_output['annotId']  = data['annotId'][0]

            outputPath = os.path.join(outputFolder, 'dp_{}_{}.pkl'.format(dp_output['imageName'][:-4], data['annotId'][0]) )
            print("Saved:{}".format(outputPath))
            with open(outputPath,'wb') as f:
                pickle.dump(dp_output,f)       
                f.close()
        
            # cv2.imwrite('/home/hjoo/temp/render_general/test_{:08d}.jpg'.format(idx), out_all_f*255.0)
            # cv2.imwrite('testout3/test_{:08d}_int.jpg'.format(idx), rawImg)
            # cv2.imwrite('tempout/test_{:08d}.jpg'.format(idx), out_all_f*255.0)

    print("erroneous Num : {}/{} ({} percent)".format(erroneousCnt,totalCnt, float(erroneousCnt)*100/totalCnt))
 