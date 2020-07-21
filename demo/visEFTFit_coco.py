# Copyright (c) Facebook, Inc. and its affiliates.

import os
from os.path import join
from os import listdir
# import json
import numpy as np

import cv2
import pickle
import torch
from smplx import SMPL
# from fairmocap.models import SMPL

from eft.utils.imutils import crop, crop_bboxInfo
from eft.utils.imutils import convert_smpl_to_bbox, convert_bbox_to_oriIm
from renderer import viewer2D#, glViewer, glRenderer
from renderer import meshRenderer #glRenderer
from renderer import denseposeRenderer #glRenderer
from renderer import torch3dRenderer #glRenderer

import argparse

## Constant
BBOX_IMG_RES = 224

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir',default="/run/media/hjoo/disk/data/coco/train2014", type=str , help='Folder path where input image files exist')
parser.add_argument('--fit_dir',default="/run/media/hjoo/disk/data/cvpr2020_eft_researchoutput/0_SPIN/0_exemplarOutput/04-14_cocoall_with8143_annotId", type=str, help='Folder path where EFT pkl files exist')
parser.add_argument('--smpl_dir',default="./extradata/smpl", type=str , help='Folder path where smpl pkl files exist')
# parser.add_argument('--export',action="store_true", help="Set if you want to export densepose label output from SMPL fits")
parser.add_argument('--onbbox',action="store_true", help="Show the 3D pose on bbox space")
parser.add_argument('--rendermode',default="geo", help="Choose among geo, normal, densepose")
parser.add_argument('--render_dir',default="render_eft", help="Folder to save rendered images")
parser.add_argument('--bShowMultiSub',action="store_true", help='If True, show multi-person outputs at each time. Default, visualize a single person at each time')
parser.add_argument('--cocoAnnotFile',default='/run/media/hjoo/disk/data/coco/annotations/person_keypoints_train2014.json', type=str , help='COCO 2014 annotation file path')

args = parser.parse_args()

#Change the path below. 
cocoAnnotFile = args.cocoAnnotFile
assert os.path.exists(cocoAnnotFile)

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


def visEFT_singleSubject(renderer):
    inputDir = args.fit_dir
    imgDir = args.img_dir

    smplModelDir = args.smpl_dir
    smpl = SMPL(smplModelDir, batch_size=1, create_transl=False)

    print("Loading coco annotation")
    cocoAnnotDic = loadCOCOAnnot()

    # outputFolder = os.path.basename(inputDir) + '_dpOut'
    # outputFolder =os.path.join('/run/media/hjoo/disk/data/eftout/',outputFolder)
    
    eft_fileList  = listdir(inputDir)       #Check all fitting files
    print(">> Found {} files in the fitting folder {}".format(len(eft_fileList), inputDir))
    totalCnt =0
    erroneousCnt =0

    for idx, f in enumerate(sorted(eft_fileList)):
        
        #Load EFT data
        fileFullPath = join(inputDir, f)
        with open(fileFullPath,'rb') as f:
            eft_data = pickle.load(f)

        #Get raw image path
        imgFullPath = eft_data['imageName'][0]
        imgName = os.path.basename(imgFullPath)
        imgFullPath =os.path.join(imgDir, os.path.basename(imgFullPath) )
        assert os.path.exists(imgFullPath)
        rawImg = cv2.imread(imgFullPath)
        print(f'Input image: {imgFullPath}')

        #EFT data
        bbox_scale = eft_data['scale'][0]
        bbox_center = eft_data['center'][0]

        pred_camera = eft_data['pred_camera']
        pred_betas = torch.from_numpy(eft_data['pred_shape'])
        pred_pose_rotmat = torch.from_numpy(eft_data['pred_pose_rotmat'])        

        #COCO only. Annotation index
        print("COCO annotId: {}".format(eft_data['annotId']))
        annot = cocoAnnotDic[eft_data['annotId'][0]]
        print(annot['bbox'])

        #Visualize COCO annotation
        annot_keypoint = np.reshape(np.array(annot['keypoints'], dtype=np.float32), (-1,3))     #17,3
        rawImg = viewer2D.Vis_Skeleton_2D_coco(annot_keypoint[:,:2],annot_keypoint[:,2], image=rawImg)
        rawImg = viewer2D.Vis_Bbox(rawImg, annot['bbox'])

        #Obtain skeleton and smpl data
        smpl_output = smpl(betas=pred_betas, body_pose=pred_pose_rotmat[:,1:], global_orient=pred_pose_rotmat[:,0].unsqueeze(1), pose2rot=False )
        smpl_vertices = smpl_output.vertices.detach().cpu().numpy() 
        smpl_joints_3d = smpl_output.joints.detach().cpu().numpy() 

        #Crop image
        croppedImg, boxScale_o2n, bboxTopLeft = crop_bboxInfo(rawImg, bbox_center, bbox_scale, (BBOX_IMG_RES, BBOX_IMG_RES) )

        ########################
        # Visualize
        if False:
            #Compute 2D reprojection error
            # if not (data['loss_keypoints_2d']<0.0001 or data['loss_keypoints_2d']>0.001 :
            #     continue
            maxBeta = abs(torch.max( abs(pred_betas)).item())
            if eft_data['loss_keypoints_2d']>0.0005 or maxBeta>3:
                erroneousCnt +=1
            print(">>> loss2d: {}, maxBeta: {}".format( eft_data['loss_keypoints_2d'],maxBeta) )
        
        # Visualize 2D image
        if True:
            viewer2D.ImShow(rawImg, name='rawImg', waitTime=1)      #You should press any key 
            viewer2D.ImShow(croppedImg, name='croppedImg', waitTime=1)

        # Visualization Mesh
        if True:    
            b=0
            camParam_scale = pred_camera[b,0]
            camParam_trans = pred_camera[b,1:]
            pred_vert_vis = smpl_vertices[b]
            smpl_joints_3d_vis = smpl_joints_3d[b]

            if args.onbbox:
                pred_vert_vis = convert_smpl_to_bbox(pred_vert_vis, camParam_scale, camParam_trans)
                smpl_joints_3d_vis = convert_smpl_to_bbox(smpl_joints_3d_vis, camParam_scale, camParam_trans)
                renderer.setBackgroundTexture(croppedImg)
                renderer.setViewportSize(croppedImg.shape[1], croppedImg.shape[0])
            else:
                #Covert SMPL to BBox first
                pred_vert_vis = convert_smpl_to_bbox(pred_vert_vis, camParam_scale, camParam_trans)
                smpl_joints_3d_vis = convert_smpl_to_bbox(smpl_joints_3d_vis, camParam_scale, camParam_trans)

                #From cropped space to original
                pred_vert_vis = convert_bbox_to_oriIm(pred_vert_vis, boxScale_o2n, bboxTopLeft, rawImg.shape[1], rawImg.shape[0]) 
                smpl_joints_3d_vis = convert_bbox_to_oriIm(smpl_joints_3d_vis, boxScale_o2n, bboxTopLeft, rawImg.shape[1], rawImg.shape[0])
                renderer.setBackgroundTexture(rawImg)
                renderer.setViewportSize(rawImg.shape[1], rawImg.shape[0])

            pred_meshes = {'ver': pred_vert_vis, 'f': smpl.faces}
            v = pred_meshes['ver'] 
            f = pred_meshes['f']

            #Visualize in the original image space
            renderer.showBackground(True)
            renderer.set_mesh(v,f)
            renderer.display()
            out_all_f = renderer.get_screen_color_ibgr()
            # out_all_f = render.get_z_value()
            viewer2D.ImShow(out_all_f,waitTime=0)

            if True:    #Save the rendered image to files
                if os.path.exists(args.render_dir) == False:
                    os.mkdir(args.render_dir)
                render_output_path = args.render_dir + '/render_{:08d}.jpg'.format(idx)
                print(f"Save to {render_output_path}")
                cv2.imwrite(render_output_path, out_all_f*255.0)

    print("erroneous Num : {}/{} ({} percent)".format(erroneousCnt,totalCnt, float(erroneousCnt)*100/totalCnt))
 

def visEFT_multiSubjects(renderer):
    inputDir = args.fit_dir
    imgDir = args.img_dir

    smplModelDir = args.smpl_dir
    smpl = SMPL(smplModelDir, batch_size=1, create_transl=False)
    
    eft_fileList  = listdir(inputDir)       #Check all fitting files
    print(">> Found {} files in the fitting folder {}".format(len(eft_fileList), inputDir))
    totalCnt =0
    erroneousCnt =0


    #Aggregate all efl per image
    eft_perimage ={}
    for f in sorted(eft_fileList):
        #Load
        imageName = f[:f.rfind('_')]
        if imageName not in eft_perimage.keys():
            eft_perimage[imageName] =[]

        eft_perimage[imageName].append(f)


    for imgName in eft_perimage:
        eftFiles_perimage = eft_perimage[imgName]
        for idx,f in enumerate(eftFiles_perimage):
            
            #Load EFT data
            fileFullPath = join(inputDir, f)
            with open(fileFullPath,'rb') as f:
                eft_data = pickle.load(f)

            #Get raw image path
            if idx==0:
                imgFullPath = eft_data['imageName'][0]
                imgFullPath =os.path.join(imgDir, os.path.basename(imgFullPath) )
                assert os.path.exists(imgFullPath)
                rawImg = cv2.imread(imgFullPath)
                print(f'Input image: {imgFullPath}')

            #EFT data
            bbox_scale = eft_data['scale'][0]
            bbox_center = eft_data['center'][0]

            pred_camera = eft_data['pred_camera']
            pred_betas = torch.from_numpy(eft_data['pred_shape'])
            pred_pose_rotmat = torch.from_numpy(eft_data['pred_pose_rotmat'])        

            #COCO only. Annotation index
            print("COCO annotId: {}".format(eft_data['annotId']))

            #Obtain skeleton and smpl data
            smpl_output = smpl(betas=pred_betas, body_pose=pred_pose_rotmat[:,1:], global_orient=pred_pose_rotmat[:,0].unsqueeze(1), pose2rot=False )
            smpl_vertices = smpl_output.vertices.detach().cpu().numpy() 
            smpl_joints_3d = smpl_output.joints.detach().cpu().numpy() 

            #Crop image
            croppedImg, boxScale_o2n, bboxTopLeft = crop_bboxInfo(rawImg.copy(), bbox_center, bbox_scale, (BBOX_IMG_RES, BBOX_IMG_RES) )

            ########################
            # Visualize
            # Visualize 2D image
            if False:
                viewer2D.ImShow(rawImg, name='rawImg', waitTime=1)      #You should press any key 
                viewer2D.ImShow(croppedImg, name='croppedImg', waitTime=0)

            # Visualization Mesh
            if True:    
                b=0
                camParam_scale = pred_camera[b,0]
                camParam_trans = pred_camera[b,1:]
                pred_vert_vis = smpl_vertices[b]
                smpl_joints_3d_vis = smpl_joints_3d[b]

                if False:#args.onbbox:      #Always in the original image
                    pred_vert_vis = convert_smpl_to_bbox(pred_vert_vis, camParam_scale, camParam_trans)
                    smpl_joints_3d_vis = convert_smpl_to_bbox(smpl_joints_3d_vis, camParam_scale, camParam_trans)
                    renderer.setBackgroundTexture(croppedImg)
                    renderer.setViewportSize(croppedImg.shape[1], croppedImg.shape[0])
                else:
                    #Covert SMPL to BBox first
                    pred_vert_vis = convert_smpl_to_bbox(pred_vert_vis, camParam_scale, camParam_trans)
                    smpl_joints_3d_vis = convert_smpl_to_bbox(smpl_joints_3d_vis, camParam_scale, camParam_trans)

                    #From cropped space to original
                    pred_vert_vis = convert_bbox_to_oriIm(pred_vert_vis, boxScale_o2n, bboxTopLeft, rawImg.shape[1], rawImg.shape[0]) 
                    smpl_joints_3d_vis = convert_bbox_to_oriIm(smpl_joints_3d_vis, boxScale_o2n, bboxTopLeft, rawImg.shape[1], rawImg.shape[0])
                    renderer.setBackgroundTexture(rawImg)
                    renderer.setViewportSize(rawImg.shape[1], rawImg.shape[0])

                pred_meshes = {'ver': pred_vert_vis, 'f': smpl.faces}
                v = pred_meshes['ver'] 
                f = pred_meshes['f']

                #Visualize in the original image spaceq
                renderer.set_mesh(v,f)
                renderer.display()
                rawImg = renderer.get_screen_color()        #Overwite on rawImg
                # out_all_f = render.get_z_value()
                #Convert to 0-255
                rawImg = (rawImg[:,:,:3]*255).astype(np.uint8)


        viewer2D.ImShow(rawImg,waitTime=1)
        if True:    #Save the rendered image to files
            if os.path.exists(args.render_dir) == False:
                os.mkdir(args.render_dir)
            render_output_path = args.render_dir + '/render_{}.jpg'.format(imgName)
            print(f"Save to {render_output_path}")
            cv2.imwrite(render_output_path, rawImg)


def loadCOCOAnnot():
    import json
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

    if args.bShowMultiSub:
        visEFT_multiSubjects(renderer)
    else:
        visEFT_singleSubject(renderer)