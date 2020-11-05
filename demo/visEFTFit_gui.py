# Copyright (c) Facebook, Inc. and its affiliates.

import os
from os.path import join
from os import listdir
import json
import numpy as np

import cv2
import pickle

from eft.utils.imutils import crop, crop_bboxInfo
from eft.models import SMPL
# from eft.models import SMPLX
# from smplx import SMPL

import torch
from eft.utils.imutils import convert_smpl_to_bbox, convert_bbox_to_oriIm
from renderer import viewer2D, glViewer
import argparse

from renderer import meshRenderer #glRenderer

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir',default="/run/media/hjoo/disk/data/coco/train2014", type=str , help='dir path where input image files exist')
parser.add_argument('--fit_dir',default="/run/media/hjoo/disk/data/cvpr2020_eft_researchoutput/0_SPIN/0_exemplarOutput/04-14_coco_with8143_annotId", type=str, help='dir path where fitting pkl files exist')
parser.add_argument('--smpl_dir',default="./extradata/smpl", type=str , help='dir path where smpl pkl files exist')
parser.add_argument('--multi',action="store_true", help='If True, show multi-person outputs at each time. Default, visualize a single person at each time')
parser.add_argument('--magnifyFactor',type=int,default=1, help='Rendering window size maginification factor')
parser.add_argument('--bRenderToFiles',action="store_true", help='Rendering the displayed output as files. Output Folder is specfied in render_dirName')
parser.add_argument('--displaytime',default=1, type=float , help='Display time for each output. Default==0 meaning that it will wait until q prossed')
parser.add_argument('--windowscale',default=3, type=float , help='scale factor for rendering window')
args = parser.parse_args()
render_dirName = "visEFT"

## Constant
BBOX_IMG_RES = 224

def getpath_level(imDir, imgFullPath,level=2):
    """
    Returns the path to the level of the given image

    Args:
        imDir: (str): write your description
        imgFullPath: (str): write your description
        level: (str): write your description
    """
    imgName = os.path.basename(imgFullPath)
    # seqname = os.path.basename( os.path.dirname( os.path.dirname(imgFullPath)))
    
    for i in range(level):
        imgFullPath = os.path.dirname(imgFullPath)
        foldername = os.path.basename(imgFullPath)
        imgName = os.path.join(foldername, imgName)
    # frmName =  os.path.basename( os.path.dirname(imgFullPath))
    return os.path.join(imDir, imgName)

def visEFT_singleSubject(inputDir, imDir, smplModelDir, bUseSMPLX):
    """
    Load a single model from a single directory.

    Args:
        inputDir: (str): write your description
        imDir: (str): write your description
        smplModelDir: (str): write your description
        bUseSMPLX: (todo): write your description
    """
    if bUseSMPLX:
        smpl = SMPLX(smplModelDir, batch_size=1, create_transl=False)
    else:
        smpl = SMPL(smplModelDir, batch_size=1, create_transl=False)
    fileList  = listdir(inputDir)       #Check all fitting files

    print(">> Found {} files in the fitting folder {}".format(len(fileList), inputDir))
    totalCnt =0
    erroneousCnt =0
    # fileList =['00_00_00008422_0.pkl', '00_00_00008422_1731.pkl', '00_00_00008422_3462.pkl']     #debug
    for f in sorted(fileList):
        
        #Load
        fileFullPath = join(inputDir, f)
        with open(fileFullPath,'rb') as f:
            dataDict = pickle.load(f)
        print(f"Loaded :{fileFullPath}")
        if 'imageName' in dataDict.keys():  #If this pkl has only one instance. Made this to hand panoptic output where pkl has multi instances
            dataDict = {0:dataDict}

        for jj, k in enumerate(dataDict):
            if jj%50 !=0:
                continue
            data = dataDict[k]
            # print(data['subjectId'])
            # continue
            if 'smpltype' in data:
                if (data['smpltype'] =='smpl' and bUseSMPLX) or (data['smpltype'] =='smplx' and bUseSMPLX==False):
                    print("SMPL type mismatch error")
                    assert False

            imgFullPathOri = data['imageName'][0]
            imgFullPath = os.path.join(imDir, os.path.basename(imgFullPathOri))


            data['subjectId'] =0 #TODO debug

            fileName = "{}_{}".format(data['subjectId'],  os.path.basename(imgFullPathOri)[:-4])
            if args.bRenderToFiles and os.path.exists(os.path.join(render_dirName, fileName+".jpg")):
                continue

            if True:    #Additional path checking, if not valid
                if os.path.exists(imgFullPath) == False:
                    imgFullPath =getpath_level(imDir, imgFullPathOri ,1)
                if os.path.exists(imgFullPath) == False:
                    imgFullPath =getpath_level(imDir, imgFullPathOri,2)
                if os.path.exists(imgFullPath) == False:
                    imgFullPath =getpath_level(imDir, imgFullPathOri, 3 )
                
            scale = data['scale'][0]
            center = data['center'][0]
            # print(data['annotId'])
            ours_betas = torch.from_numpy(data['pred_shape'])
            ours_pose_rotmat = torch.from_numpy(data['pred_pose_rotmat'])
            
            #Compute 2D reprojection error
            # if not (data['loss_keypoints_2d']<0.0001 or data['loss_keypoints_2d']>0.001 :
            #     continue
            maxBeta = abs(torch.max( abs(ours_betas)).item())

            if data['loss_keypoints_2d']>0.0005 or maxBeta>3:
                erroneousCnt +=1
            
            print(">>> loss2d: {}, maxBeta: {}".format( data['loss_keypoints_2d'],maxBeta) )

            # spin_pose = torch.from_numpy(data['opt_pose'])
            pred_camera_vis = data['pred_camera']
    
            if os.path.exists(imgFullPath) == False:
                print(imgFullPath)
                assert os.path.exists(imgFullPath)
            rawImg = cv2.imread(imgFullPath)
            print(imgFullPath)

            croppedImg, boxScale_o2n, bboxTopLeft = crop_bboxInfo(rawImg, center, scale, (BBOX_IMG_RES, BBOX_IMG_RES) )

            #Visualize 2D image
            if args.bRenderToFiles ==False:
                viewer2D.ImShow(rawImg, name='rawImg', waitTime=10)      #You should press any key 
                viewer2D.ImShow(croppedImg, name='croppedImg', waitTime=10)

            ours_output = smpl(betas=ours_betas, body_pose=ours_pose_rotmat[:,1:], global_orient=ours_pose_rotmat[:,0].unsqueeze(1), pose2rot=False )
            ours_vertices = ours_output.vertices.detach().cpu().numpy() 
            ours_joints_3d = ours_output.joints.detach().cpu().numpy() 

            #Visualize 3D mesh and 3D skeleton in BBox Space
            if True:
                b =0
                camParam_scale = pred_camera_vis[b,0]
                camParam_trans = pred_camera_vis[b,1:]

                ############### Visualize Mesh ############### 
                pred_vert_vis = ours_vertices[b].copy()
                pred_vert_vis = convert_smpl_to_bbox(pred_vert_vis, camParam_scale, camParam_trans)
                pred_meshes = {'ver': pred_vert_vis, 'f': smpl.faces}
                glViewer.setMeshData([pred_meshes], bComputeNormal= True)

                ################ Visualize Skeletons ############### 
                #Vis pred-SMPL joint
                pred_joints_vis = ours_joints_3d[b,:,:3].copy()     #(N,3)
                pred_joints_vis = convert_smpl_to_bbox(pred_joints_vis, camParam_scale, camParam_trans)
                glViewer.setSkeleton( [pred_joints_vis.ravel()[:,np.newaxis]])

                ################ Other 3D setup############### 
                glViewer.setBackgroundTexture(croppedImg)
                glViewer.setWindowSize(croppedImg.shape[1]*args.windowscale, croppedImg.shape[0]*args.windowscale)
                glViewer.SetOrthoCamera(True)

                print("Press 'q' in the 3D window to go to the next sample")
                glViewer.show(0)
            
            #Visualize 3D mesh and 3D skeleton on original image space
            if True:
                b =0
                camParam_scale = pred_camera_vis[b,0]
                camParam_trans = pred_camera_vis[b,1:]

                ############### Visualize Mesh ############### 
                pred_vert_vis = ours_vertices[b].copy()
                pred_vert_vis = convert_smpl_to_bbox(pred_vert_vis, camParam_scale, camParam_trans)

                #From cropped space to original
                pred_vert_vis = convert_bbox_to_oriIm(pred_vert_vis, boxScale_o2n, bboxTopLeft, rawImg.shape[1], rawImg.shape[0]) 
                pred_meshes = {'ver': pred_vert_vis, 'f': smpl.faces}
                glViewer.setMeshData([pred_meshes], bComputeNormal= True)

                # ################ Visualize Skeletons ############### 
                #Vis pred-SMPL joint
                pred_joints_vis = ours_joints_3d[b,:,:3].copy()     #(N,3)
                pred_joints_vis = convert_smpl_to_bbox(pred_joints_vis, camParam_scale, camParam_trans)
                pred_joints_vis = convert_bbox_to_oriIm(pred_joints_vis, boxScale_o2n, bboxTopLeft, rawImg.shape[1], rawImg.shape[0]) 


                glViewer.setSkeleton( [pred_joints_vis.ravel()[:,np.newaxis]])

                glViewer.setBackgroundTexture(rawImg)
                glViewer.setWindowSize(rawImg.shape[1]*args.magnifyFactor, rawImg.shape[0]*args.magnifyFactor)
                glViewer.SetOrthoCamera(True)

                print("Press 'q' in the 3D window to go to the next sample")

                if args.bRenderToFiles:        #Export rendered files
                    if os.path.exists(render_dirName) == False:     #make a output folder if necessary
                         os.mkdir(render_dirName)

                    # subjId = data['subjectId'][22:24]
                    fileName = "{}_{}".format(data['subjectId'],  os.path.basename(imgFullPathOri)[:-4])

                    # rawImg = cv2.putText(rawImg,data['subjectId'],(100,100), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0),2)
                    glViewer.render_on_image(render_dirName, fileName, rawImg)
                    print(f"Render to {fileName}")
                
 
def save_mesh_obj(verts, faces, obj_mesh_name):
    """
    Save a mesh as a mesh file.

    Args:
        verts: (str): write your description
        faces: (list): write your description
        obj_mesh_name: (str): write your description
    """
    with open(obj_mesh_name, "w") as fp:
        for v in verts:
            fp.write(f"v {v[0]:f} {v[1]:f} {v[2]:f}\n")

        for f in faces:
            o=1 #offset
            fp.write(f"f {f[0] + o:d} {f[1] + o:d} {f[2] + o:d}\n")


def visEFT_multiSubjects(inputDir, imDir, smplModelDir, bUseSMPLX = False):
    """
    Finds multiple images inimage images.

    Args:
        inputDir: (str): write your description
        imDir: (str): write your description
        smplModelDir: (str): write your description
        bUseSMPLX: (todo): write your description
    """

    if bUseSMPLX:
        smpl = SMPLX(smplModelDir, batch_size=1, create_transl=False)
    else:
        smpl = SMPL(smplModelDir, batch_size=1, create_transl=False)

    fileList  = listdir(inputDir)       #Check all fitting files

    print(">> Found {} files in the fitting folder {}".format(len(fileList), inputDir))
    totalCnt =0
    erroneousCnt =0
    #Merge sample from the same image

    data_perimage ={}
    for f in sorted(fileList):
        
        if "_init" in f:
            continue
        #Load
        imageName = f[:f.rfind('_')]
        if imageName not in data_perimage.keys():
            data_perimage[imageName] =[]

        data_perimage[imageName].append(f)

    for imgName in data_perimage:

        eftFileNames = data_perimage[imgName]

        meshData =[]
        skelData =[]
        for f in eftFileNames:
            fileFullPath = join(inputDir, f)
            with open(fileFullPath,'rb') as f:
                data = pickle.load(f)
            imgFullPathOri = data['imageName'][0]
            imgFullPath = os.path.join(imDir, os.path.basename(imgFullPathOri))

            
            if True:    #Additional path checking, if not valid
                if os.path.exists(imgFullPath) == False:
                    imgFullPath =getpath_level(imDir, imgFullPathOri ,1)
                if os.path.exists(imgFullPath) == False:
                    imgFullPath =getpath_level(imDir, imgFullPathOri,2)
                if os.path.exists(imgFullPath) == False:
                    imgFullPath =getpath_level(imDir, imgFullPathOri, 3 )

            scale = data['scale'][0]
            center = data['center'][0]

            ours_betas = torch.from_numpy(data['pred_shape'])
            ours_pose_rotmat = torch.from_numpy(data['pred_pose_rotmat'])
            #Compute 2D reprojection error
            # if not (data['loss_keypoints_2d']<0.0001 or data['loss_keypoints_2d']>0.001 :
            #     continue
            maxBeta = abs(torch.max( abs(ours_betas)).item())

            if data['loss_keypoints_2d']>0.0005 or maxBeta>3:
                erroneousCnt +=1
            
            print(">>> loss2d: {}, maxBeta: {}".format( data['loss_keypoints_2d'],maxBeta) )

            # spin_pose = torch.from_numpy(data['opt_pose'])
            pred_camera_vis = data['pred_camera']
    
            assert os.path.exists(imgFullPath)
            rawImg = cv2.imread(imgFullPath)
            print(imgFullPath)

            croppedImg, boxScale_o2n, bboxTopLeft = crop_bboxInfo(rawImg, center, scale, (constants.IMG_RES, constants.IMG_RES) )

            #Visualize 2D image
            if args.bRenderToFiles ==False:
                viewer2D.ImShow(rawImg, name='rawImg', waitTime=10)      #You should press any key 
                viewer2D.ImShow(croppedImg, name='croppedImg', waitTime=10)

            if bUseSMPLX:
                ours_output = smpl(betas=ours_betas, body_pose=ours_pose_rotmat[:,1:-2], global_orient=ours_pose_rotmat[:,0].unsqueeze(1), pose2rot=False )
                # ours_output = smpl()        #Default test
            else:
                ours_output = smpl(betas=ours_betas, body_pose=ours_pose_rotmat[:,1:], global_orient=ours_pose_rotmat[:,0].unsqueeze(1), pose2rot=False )
                # ours_output = smpl()        #Default test
            ours_vertices = ours_output.vertices.detach().cpu().numpy() 
            ours_joints_3d = ours_output.joints.detach().cpu().numpy() 

            if False:   #Debugging
                # ours_vertices = ours_vertices - ours_joints_3d[0,12,:]
                save_mesh_obj(ours_vertices[0], smpl.faces, 'test.obj')

            #Visualize 3D mesh and 3D skeleton on original image space
            if True:
                b =0
                camParam_scale = pred_camera_vis[b,0]
                camParam_trans = pred_camera_vis[b,1:]

                ############### Visualize Mesh ############### 
                pred_vert_vis = ours_vertices[b].copy()
                pred_vert_vis = convert_smpl_to_bbox(pred_vert_vis, camParam_scale, camParam_trans)

                #From cropped space to original
                pred_vert_vis = convert_bbox_to_oriIm(pred_vert_vis, boxScale_o2n, bboxTopLeft, rawImg.shape[1], rawImg.shape[0]) 
                pred_meshes = {'ver': pred_vert_vis, 'f': smpl.faces}

                # ################ Visualize Skeletons ############### 
                pred_joints_vis = ours_joints_3d[b,:,:3].copy()     #(N,3)
                pred_joints_vis = convert_smpl_to_bbox(pred_joints_vis, camParam_scale, camParam_trans)
                pred_joints_vis = convert_bbox_to_oriIm(pred_joints_vis, boxScale_o2n, bboxTopLeft, rawImg.shape[1], rawImg.shape[0]) 


                meshData.append(pred_meshes)
                skelData.append(pred_joints_vis.ravel()[:,np.newaxis])

                glViewer.setBackgroundTexture(rawImg)
                glViewer.setWindowSize(rawImg.shape[1]*args.magnifyFactor, rawImg.shape[0]*args.magnifyFactor)
                glViewer.SetOrthoCamera(True)

        glViewer.setSkeleton(skelData)
        glViewer.setMeshData(meshData, bComputeNormal= True)

        if args.bRenderToFiles:        #Export rendered files
            if os.path.exists(render_dirName) == False:     #make a output folder if necessary
                    os.mkdir(render_dirName)
            fileName = imgFullPathOri[:-4].replace("/","_")
            glViewer.render_on_image(render_dirName, fileName, rawImg)
            print(f"render to {fileName}")

        glViewer.show(args.displaytime)


if __name__ == '__main__':

    smplPath =  args.smpl_dir + '/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
    if args.multi:
        visEFT_multiSubjects(args.fit_dir, args.img_dir, smplPath, bUseSMPLX=False)        #SMPLX version
    else:
        visEFT_singleSubject(args.fit_dir, args.img_dir, smplPath, bUseSMPLX=False)