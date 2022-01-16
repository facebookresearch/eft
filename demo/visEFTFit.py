# Copyright (c) Facebook, Inc. and its affiliates.

import os
from os.path import join
from os import listdir
# import json
import numpy as np

import cv2
import pickle
import torch
# from smplx import SMPL
from eft.models import SMPL_19

from eft.utils.imutils import crop, crop_bboxInfo
from eft.utils.imutils import convert_smpl_to_bbox, convert_bbox_to_oriIm, conv_bboxinfo_bboxXYXY
from eft.utils.geometry import weakProjection
from renderer import viewer2D#, glViewer, glRenderer
from renderer import meshRenderer #screen less opengl renderer
from renderer import glViewer #gui mode opengl renderer
from renderer import denseposeRenderer #densepose renderer


from tqdm import tqdm
import argparse
import json

## Constant
BBOX_IMG_RES = 224

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir',default="/run/media/hjoo/disk/data/mpii_human_pose_v1/images", type=str , help='Folder path where input image files exist')
parser.add_argument('--fit_data',default="eft_fit/MPII_ver01.json", type=str, help='EFT data json fortmat')
parser.add_argument('--smpl_dir',default="./extradata/smpl", type=str , help='Folder path where smpl pkl files exist')
parser.add_argument('--onbbox',action="store_true", help="Show the 3D pose on bbox space")
parser.add_argument('--rendermode',default="geo", help="Choose among geo, normal, densepose")
parser.add_argument('--multi_bbox',default=True, help="If ture, show multi-bbox computed from mesh vertices")
parser.add_argument('--render_dir',default="render_eft", help="Folder to save rendered images")
parser.add_argument('--waitforkeys',action="store_true", help="If true, it will pasue after each visualizing each sample, waiting for any key pressed")
parser.add_argument('--turntable',action="store_true", help="If true, show turn table views")
parser.add_argument('--multi',action="store_true", help='If True, show all available fitting people per image. Default, visualize a single person at each time')
args = parser.parse_args()

def getRenderer(ren_type='geo'):
    """
    Choose renderer type
    geo: phong-shading (silver color)
    colorgeo: phong-shading with color (need color infor. Default silver color)
    denspose: densepose IUV
    normal: normal map
    torch3d: via pytorch3d TODO
    """
    if ren_type=='geo':
        renderer = meshRenderer.meshRenderer()
        renderer.setRenderMode('geo')

    elif ren_type=='colorgeo':
        renderer = meshRenderer.meshRenderer()
        renderer.setRenderMode('colorgeo')
        
    elif ren_type=='normal':
        renderer = meshRenderer.meshRenderer()
        renderer.setRenderMode('normal')

    elif ren_type=='densepose':
        renderer = denseposeRenderer.denseposeRenderer()

    # elif  ren_type=='torch3d':
    #     renderer = torch3dRenderer.torch3dRenderer()
    else:
        assert False

    renderer.offscreenMode(True)
    # renderer.bAntiAliasing= False
    return renderer

def conv_3djoint_2djoint(smpl_joints_3d_vis, imgshape):

    smpl_joints_2d_vis = smpl_joints_3d_vis[:,:2]       #3D is in camera comaera coordinate with origin on the image center
    smpl_joints_2d_vis[:,0] += imgshape[1]*0.5      #Offset to move the origin on the top left
    smpl_joints_2d_vis[:,1] += imgshape[0]*0.5

    return smpl_joints_2d_vis
    

def visEFT_singleSubject(renderer):

    MAGNIFY_RATIO = 3           #onbbox only. To magnify the rendered image size 

    bStopForEachSample = args.waitforkeys      #if True, it will wait for any key pressed to move to the next sample
    bShowTurnTable = args.turntable

    inputData = args.fit_data
    imgDir = args.img_dir

    #Load SMPL model
    smplModelPath = args.smpl_dir + '/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
    smpl = SMPL_19(smplModelPath, batch_size=1, create_transl=False)
    
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
    for idx, eft_data in enumerate(tqdm(eft_data_all)):
        
        #Get raw image path
        imgFullPath = eft_data['imageName']
        # imgName = os.path.basename(imgFullPath)
        imgName = imgFullPath
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
        if 'annotId' in eft_data.keys():
            print("COCO annotId: {}".format(eft_data['annotId']))


        #Get SMPL mesh and joints from SMPL parameters
        smpl_output = smpl(betas=pred_betas, body_pose=pred_pose_rotmat[:,1:], global_orient=pred_pose_rotmat[:,[0]], pose2rot=False)
        smpl_vertices = smpl_output.vertices.detach().cpu().numpy()[0]
        smpl_joints_3d = smpl_output.joints.detach().cpu().numpy()[0]

        #Crop image using cropping information
        croppedImg, boxScale_o2n, bboxTopLeft = crop_bboxInfo(rawImg, bbox_center, bbox_scale, (BBOX_IMG_RES, BBOX_IMG_RES) )


        if MAGNIFY_RATIO>1:
            croppedImg = cv2.resize(croppedImg, (croppedImg.shape[1]*MAGNIFY_RATIO, croppedImg.shape[0]*MAGNIFY_RATIO) )

        ########################
        # Visualization
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

            if args.onbbox:
                pred_vert_vis = convert_smpl_to_bbox(pred_vert_vis, camParam_scale, camParam_trans)
                smpl_joints_3d_vis = convert_smpl_to_bbox(smpl_joints_3d_vis, camParam_scale, camParam_trans)
                renderer.setBackgroundTexture(croppedImg)
                renderer.setViewportSize(croppedImg.shape[1], croppedImg.shape[0])

                pred_vert_vis *=MAGNIFY_RATIO
            else:
                #Covert SMPL to BBox first
                pred_vert_vis = convert_smpl_to_bbox(pred_vert_vis, camParam_scale, camParam_trans)
                smpl_joints_3d_vis = convert_smpl_to_bbox(smpl_joints_3d_vis, camParam_scale, camParam_trans)

                #From cropped space to original
                pred_vert_vis = convert_bbox_to_oriIm(pred_vert_vis, boxScale_o2n, bboxTopLeft, rawImg.shape[1], rawImg.shape[0]) 
                smpl_joints_3d_vis = convert_bbox_to_oriIm(smpl_joints_3d_vis, boxScale_o2n, bboxTopLeft, rawImg.shape[1], rawImg.shape[0])
                renderer.setBackgroundTexture(rawImg)
                renderer.setViewportSize(rawImg.shape[1], rawImg.shape[0])

                #In orthographic model. XY of 3D is just 2D projection
                smpl_joints_2d_vis = conv_3djoint_2djoint(smpl_joints_3d_vis,rawImg.shape )
                # image_2dkeypoint_pred = viewer2D.Vis_Skeleton_2D_smpl45(smpl_joints_2d_vis, image=rawImg.copy(),color=(0,255,255))
                image_2dkeypoint_pred = viewer2D.Vis_Skeleton_2D_Openpose18(smpl_joints_2d_vis, image=rawImg.copy(),color=(255,0,0))        #All 2D joint
                image_2dkeypoint_pred = viewer2D.Vis_Skeleton_2D_Openpose18(smpl_joints_2d_vis, pt2d_visibility=keypoint_2d_validity, image=image_2dkeypoint_pred,color=(0,255,255))        #Only valid
                viewer2D.ImShow(image_2dkeypoint_pred, name='keypoint_2d_pred', waitTime=1)

            pred_meshes = {'ver': pred_vert_vis, 'f': smpl.faces}
            v = pred_meshes['ver'] 
            f = pred_meshes['f']

            #Visualize in the original image space
            renderer.set_mesh(v,f)
            renderer.showBackground(True)
            renderer.setWorldCenterBySceneCenter()
            renderer.setCameraViewMode("cam")

            #Set image size for rendering
            if args.onbbox:
                renderer.setViewportSize(croppedImg.shape[1], croppedImg.shape[0])
            else:
                renderer.setViewportSize(rawImg.shape[1], rawImg.shape[0])
                
            renderer.display()
            renderImg = renderer.get_screen_color_ibgr()
            viewer2D.ImShow(renderImg,waitTime=1)
        
        # Visualize multi-level cropped bbox
        if args.multi_bbox:
            from demo.multi_bbox_gen import multilvel_bbox_crop_gen
            
            bbox_list = multilvel_bbox_crop_gen(rawImg, pred_vert_vis, bbox_center, bbox_scale)

            #Visualize BBox
            for b_idx, b in enumerate(bbox_list):
                # bbox_xyxy= conv_bboxinfo_centerscale_to_bboxXYXY(b['center'], b['scale'])
                bbox_xyxy= b['bbox_xyxy']
                if b_idx==0:
                    img_multi_bbox = viewer2D.Vis_Bbox_minmaxPt(rawImg,  bbox_xyxy[:2], bbox_xyxy[2:] ,color=(0,255,0))
                else:
                    img_multi_bbox = viewer2D.Vis_Bbox_minmaxPt(rawImg,  bbox_xyxy[:2], bbox_xyxy[2:] ,color=(0,255,255))
            viewer2D.ImShow(img_multi_bbox, name='multi_bbox', waitTime=1)
            # for bbox in bbox_list:


        # Visualization Mesh on side view
        if True:
            renderer.showBackground(False)
            renderer.setWorldCenterBySceneCenter()
            # renderer.setCameraViewMode("side")    #To show the object in side vie
            renderer.setCameraViewMode("free")     
            renderer.setViewAngle(90,20)

            #Set image size for rendering
            if args.onbbox:
                renderer.setViewportSize(croppedImg.shape[1], croppedImg.shape[0])
            else:
                renderer.setViewportSize(rawImg.shape[1], rawImg.shape[0])
            renderer.display()
            sideImg = renderer.get_screen_color_ibgr()        #Overwite on rawImg
            viewer2D.ImShow(sideImg,waitTime=1)
            
            sideImg = cv2.resize(sideImg, (renderImg.shape[1], renderImg.shape[0]) )
            # renderImg = cv2.resize(renderImg, (sideImg.shape[1], sideImg.shape[0]) )
        
        # Visualization Mesh on side view
        if True:
            renderer.showBackground(False)
            renderer.setWorldCenterBySceneCenter()
            # renderer.setCameraViewMode("side")    #To show the object in side vie
            renderer.setCameraViewMode("free")     
            renderer.setViewAngle(-60,50)

            #Set image size for rendering
            if args.onbbox:
                renderer.setViewportSize(croppedImg.shape[1], croppedImg.shape[0])
            else:
                renderer.setViewportSize(rawImg.shape[1], rawImg.shape[0])
            renderer.display()
            sideImg_2 = renderer.get_screen_color_ibgr()        #Overwite on rawImg
            viewer2D.ImShow(sideImg_2,waitTime=1)
            
            sideImg_2 = cv2.resize(sideImg_2, (renderImg.shape[1], renderImg.shape[0]) )
            # renderImg = cv2.resize(renderImg, (sideImg.shape[1], sideImg.shape[0]) )


        #Visualize camera view and side view
        saveImg = np.concatenate( (renderImg,sideImg), axis =1)
        # saveImg = np.concatenate( (croppedImg, renderImg,sideImg, sideImg_2), axis =1)

        if bStopForEachSample:
            viewer2D.ImShow(saveImg,waitTime=0) #waitTime=0 means that it will wait for any key pressed
        else:
            viewer2D.ImShow(saveImg,waitTime=1)
        
        #Render Mesh on the rotating view
        if bShowTurnTable:
            renderer.showBackground(False)
            renderer.setWorldCenterBySceneCenter()
            renderer.setCameraViewMode("free")
            for i in range(90):
                renderer.setViewAngle(i*4,0)
                renderer.display()
                sideImg = renderer.get_screen_color_ibgr()        #Overwite on rawImg
                viewer2D.ImShow(sideImg,waitTime=1,name="turn_table")

                if False:       #If you want to save this into files
                    render_output_path = args.render_dir + '/turntable_{}_{:08d}.jpg'.format(os.path.basename(imgName),i)
                    cv2.imwrite(render_output_path, sideImg)

        #Save the rendered image to files
        if True:    
            if os.path.exists(args.render_dir) == False:
                os.mkdir(args.render_dir)
            render_output_path = args.render_dir + '/render_{}_eft{:08d}.jpg'.format(imgName[:-4],idx)
            print(f"Save to {render_output_path}")
            cv2.imwrite(render_output_path, saveImg)


def visEFT_multiSubjects(renderer):

    bStopForEachSample = args.waitforkeys      #if True, it will wait for any key pressed to move to the next sample
    bShowTurnTable = args.turntable
    
    # inputDir = args.fit_dir
    inputData = args.fit_data
    imgDir = args.img_dir
    smplModelPath = args.smpl_dir + '/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
    smpl = SMPL(smplModelPath, batch_size=1, create_transl=False)

    if os.path.exists(inputData):
        with open(inputData,'r') as f:
            eft_data = json.load(f)
            print("EFT data: ver {}".format(eft_data['ver']))
            eft_data_all = eft_data['data']
    else:
        print(f"ERROR:: Cannot find EFT data: {inputData}")
        assert False

    #Aggregate all efl per image
    eft_perimage ={}
    for idx, eft_data in enumerate(eft_data_all):
        #Load
        imageName = eft_data['imageName']
        if imageName not in eft_perimage.keys():
            eft_perimage[imageName] =[]

        eft_perimage[imageName].append(eft_data)


    for imgName in tqdm(eft_perimage):
        eft_data_perimage = eft_perimage[imgName]
        
        renderer.clear_mesh()

        for idx,eft_data in enumerate(eft_data_perimage):
            
            #Get raw image path
            imgFullPath = eft_data['imageName']
            imgName = os.path.basename(imgFullPath)
            imgFullPath =os.path.join(imgDir, imgName)
            if os.path.exists(imgFullPath) ==False:
                print(f"Img path is not valid: {imgFullPath}")
                assert False
            rawImg = cv2.imread(imgFullPath)
            print(f'Input image: {imgFullPath}')

            bbox_scale = eft_data['bbox_scale']
            bbox_center = eft_data['bbox_center']

            pred_camera = np.array(eft_data['parm_cam'])
            pred_betas = np.reshape(np.array( eft_data['parm_shape'], dtype=np.float32), (1,10) )     #(10,)
            pred_betas = torch.from_numpy(pred_betas)

            pred_pose_rotmat = np.reshape( np.array( eft_data['parm_pose'], dtype=np.float32), (1,24,3,3)  )        #(24,3,3)
            pred_pose_rotmat = torch.from_numpy(pred_pose_rotmat)
        
            # gt_keypoint_2d = np.reshape( np.array(eft_data['gt_keypoint_2d']), (-1,3))    #(49,3)
            keypoint_2d_validity = eft_data['joint_validity_openpose18']

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
                viewer2D.ImShow(croppedImg, name='croppedImg', waitTime=1)

            # Visualization Mesh on raw images
            if True:    
                camParam_scale = pred_camera[0]
                camParam_trans = pred_camera[1:]
                pred_vert_vis = smpl_vertices[0]
                smpl_joints_3d_vis = smpl_joints_3d[0]

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
                # renderer.set_mesh(v,f)
                renderer.add_mesh(v,f)

        #Render Mesh on the camera view
        renderer.showBackground(True)
        renderer.setWorldCenterBySceneCenter()
        renderer.setCameraViewMode("cam")
        renderer.display()
        overlaid = renderer.get_screen_color_ibgr()        #Overwite on rawImg
        # viewer2D.ImShow(overlaid,waitTime=1,name="overlaid")

        if bStopForEachSample:
            viewer2D.ImShow(overlaid,waitTime=0,name="overlaid") #waitTime=0 means that it will wait for any key pressed
        else:
            viewer2D.ImShow(overlaid,waitTime=1,name="overlaid")

        #Render Mesh on the rotating view
        if bShowTurnTable:
            renderer.showBackground(False)
            renderer.setWorldCenterBySceneCenter()
            renderer.setCameraViewMode("free")
            for i in range(90):
                renderer.setViewAngle(i*4,0)
                renderer.display()
                sideImg = renderer.get_screen_color_ibgr()        #Overwite on rawImg
                viewer2D.ImShow(sideImg,waitTime=1,name="turn_table")
            
        if True:    #Save the rendered image to files
            if os.path.exists(args.render_dir) == False:
                os.mkdir(args.render_dir)
            render_output_path = args.render_dir + '/render_{}.jpg'.format(imgName)
            print(f"Save to {render_output_path}")
            cv2.imwrite(render_output_path, rawImg)

if __name__ == '__main__':
    renderer = getRenderer(args.rendermode)

    if args.multi:
        visEFT_multiSubjects(renderer)
    else:
        visEFT_singleSubject(renderer)