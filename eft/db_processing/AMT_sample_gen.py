import os
from os.path import join
import sys
import json
import numpy as np
# from .read_openpose import read_openpose

import cv2
from renderer import viewer2D
from renderer import glViewer

# from read_openpose import read_openpose

from os import listdir
from os.path import isfile, join
import pickle

sys.path.append('/home/hjoo/codes/SPIN/utils')
sys.path.append('/home/hjoo/codes/SPIN')
from imutils import crop, flip_img, flip_pose, flip_kp, transform, rot_aa
from fairmocap.core import constants 
from fairmocap.core import config 
from models import hmr, SMPL
import torch
import pickle

import subprocess


g_outputName = 'blue2'
g_renderDir = '/home/hjoo/temp/render_general_' + g_outputName
glViewer.setSaveFoldeName(g_renderDir)

# outputFolderRoot = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN'
# overlaidImageFolder= os.path.join(outputFolderRoot, g_outputName)

# if os.path.exists(overlaidImageFolder)==False:
#     os.mkdir(overlaidImageFolder)

from exemplar_exportToDB import IsValid
import json

smpl = SMPL(config.SMPL_MODEL_DIR,
                         batch_size=1,
                         create_transl=False)
from exemplar_exportToDB import IsValid


def IsValid_strict(data):
    ours_betas = torch.from_numpy(data['pred_shape'])
    if abs(torch.max( abs(ours_betas)).item()) >1.0:
        return False
    
    if 'loss_keypoints_2d_init' in data.keys()  and data['loss_keypoints_2d_init']>0.04:
        print("Rejected: loss_keypoints_2d_init: {}>0.03".format(data['loss_keypoints_2d_init']))
        return False


    if data['loss_keypoints_2d']>0.0003:
        print("Rejected: loss_keypoints_2d: {}>0.001".format(data['loss_keypoints_2d']))

        return False
    
    return True




# g_rootDirName = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_quant/AMT_1Kset_fromChallenging'
# g_rootDirName = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_quant/11-04_lspet_originalCode_weak_meta'
# g_rootDirName = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_quant/11-04_mpii_originalCode_weak_meta'

g_rootDirName = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_quant/11-04_visualizeDBs'
# g_rootDirName = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_quant/11-04_visualizeDBs/mpii_comparison'


# g_rootDirName ='/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_quant/11-06-AMT_5Kset_easy'
# g_rootDirName ='/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_quant/11-06-AMT_5Kset_hard'

g_rootDirName = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_quant/0_FinalVersion/11-06-AMT_5Kset_easy'
# g_rootDirName = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_quant/0_FinalVersion/11-06-AMT_5Kset_hard'
# g_rootDirName = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_quant/0_FinalVersion/11-08-cocoplus'
# g_rootDirName = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_quant/0_FinalVersion/11-08-posetrackWith8143'


if os.path.exists(g_rootDirName)==False:
    os.mkdir(g_rootDirName)

def RenderFitting(inputDir_list, bVisSpinOriginal=False, bAlwaysBlue = False, bSampling= True, bRenderInit = False):

    #1K random selection file
    # selectionFileDir ='/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_quant/AMT_1Kset_fromAll'
    # selectionFileDir ='/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_quant/11-04_visualizeDBs/mpii_comparison'

    # selectionFileDir ='/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_quant/11-06-AMT_5Kset_easy'
    selectionFileDir = g_rootDirName

    
    # selectionFileDir ='/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_quant/AMT_1Kset_fromChallenging'
    selectionOutoutTxt = os.path.join(selectionFileDir, 'selected500.json')


    # metaDir_list ={}
    # metaDir_list['coco'] ='/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/meta_coco_2dskeletons'
    # metaDir_list['mpii'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/meta_mpii_2dskeletons'
    # metaDir_list['lspet'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/meta_lspet_2dskeletons'
    # # metaDir_list['pennaction'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/meta_lspet_2dskeletons'

    imgDir ={}
    imgDir['mpii'] = '/run/media/hjoo/disk/data/mpii_human_pose_v1/images'
    imgDir['lspet'] = '/run/media/hjoo/disk/data/lspet_dataset/images_highres'
    imgDir['coco'] =  '/run/media/hjoo/disk/data/coco/train2014'
    imgDir['cocoplus'] =  '/run/media/hjoo/disk/data/coco/train2014'
    imgDir['pennaction'] =  '/run/media/hjoo/disk/data/Penn_Action/frames'
    imgDir['panoptic'] =  '/run/media/hjoo/disk/data/panoptic_mtc/a4_release/hdImgs'
    imgDir['posetrack'] =  '/run/media/hjoo/disk/data/posetrack/images/train'

    if bVisSpinOriginal:
        if bAlwaysBlue:
            rootDirName = os.path.join(g_rootDirName,'SPIN_original_blue')
            glViewer.SetMeshColor('blue')
        else:
            rootDirName = os.path.join(g_rootDirName,'SPIN_original_red')
            glViewer.SetMeshColor('red')
    else:

        dbVersion = os.path.basename( inputDir_list[ next(iter(inputDir_list))])
        if bRenderInit:
            rootDirName = os.path.join(g_rootDirName,dbVersion + '_init')
        else:
            rootDirName = os.path.join(g_rootDirName, dbVersion )
        glViewer.SetMeshColor('blue')


    if False== os.path.exists(rootDirName):
        os.mkdir(rootDirName)
        
     #Folder Name
    imageFolder = os.path.join(rootDirName, 'images')
    if not os.path.exists(imageFolder):
        os.mkdir(imageFolder)

    croppedimageFolder = os.path.join(rootDirName, 'croppedImage')
    if not os.path.exists(croppedimageFolder):
        os.mkdir(croppedimageFolder)

    skelCroppedimageFolder_ = os.path.join(rootDirName, 'skelCroppedImage')
    if not os.path.exists(skelCroppedimageFolder_):
        os.mkdir(skelCroppedimageFolder_)

    overlaidImageFolder = os.path.join(rootDirName, 'overlaid')
    if not os.path.exists(overlaidImageFolder):
        os.mkdir(overlaidImageFolder)
    
    noBG_overlaidImageFolder = os.path.join(rootDirName, 'overlaid_nobg')
    if not os.path.exists(noBG_overlaidImageFolder):
        os.mkdir(noBG_overlaidImageFolder)

    sideImageFolder = os.path.join(rootDirName, 'side')
    if not os.path.exists(sideImageFolder):
        os.mkdir(sideImageFolder)

    mergedImageFolder = os.path.join(rootDirName, 'merged')
    if not os.path.exists(mergedImageFolder):
        os.mkdir(mergedImageFolder)


    if bSampling:

        #Ignore wrong annotation cases (left,right flipped)
        blacklist =['092777110.jpg',
                'im06457.png',
                'im04109.png',
                'im03087.png',
                'im00153.png',
                'im01467.png',
                'im03756.png',
                'im03666.png', 
                'im03860.png',
                'im04101.png',
                'im04116.png',
                'im05634.png', 
                'im09286.png',
                'im09855.png',
                'im09957.png',
                'im08994.png',
                'im09683.png',
                'im00525.png',
                'im00563.png',
                'im03114.png',
                'im04469.png',
                'im04607.png',
                'im04805.png',
                'im06160.png',
                'im06551.png',
                'im08059.png',
                'im09849.png',
                'im08476.png',
                'COCO_train2014_000000040055.jpg'] 
                
        if os.path.exists(selectionOutoutTxt) == False:     #Generate new random 1K sample
            totalFileList =[]
            sampleNum ={}
            selectedSampleNum ={}


            for dbName in inputDir_list:
                inputDir = inputDir_list[dbName]
                fileList  = listdir(inputDir)
                sampleNum[dbName] = len(fileList)
                selectedSampleNum[dbName] =0
                for fileName in fileList:

                    if True:#'AMT_1Kset_fromChallenging' in rootDirName:
                        fileFullPath = join(inputDir, fileName)
                        with open(fileFullPath,'rb') as f:
                            data = pickle.load(f)

                        if os.path.basename(data['imageName'][0]) in blacklist:
                            continue

                      
        
                        # metaDir = metaDir_list[dbName]
                        # meta_fileFullPath = join(metaDir, fileName)
                        # with open(meta_fileFullPath,'rb') as f:
                        #     metadata = pickle.load(f)
                        #     data['opt_beta'] = metadata['opt_beta']
                        #     data['opt_pose'] = metadata['opt_pose']
                        if np.isnan(np.max(data['pred_pose_rotmat'])):
                            continue
                        spin_betas = torch.from_numpy(data['opt_beta'])

                        #guess bbox size 
                        validity = data['keypoint2d'][0,:,2] ==1
                        valid_keypoints = data['keypoint2d'][0,validity,:2]
                        min_pt = np.min(valid_keypoints, axis=0)
                        max_pt = np.max(valid_keypoints, axis=0)
                        bboxHeight = max_pt[1] - min_pt[1]
                        # if bboxHeight<100:
                        #     continue

                        if False: #Filtering #f samplingType is not None:
                            # if abs(torch.max( abs(spin_betas)) .item()) >=3:        #Easy example only
                            #     continue
                            if abs(torch.max( abs(spin_betas)).item()) <3:        #Hard example only
                                continue

                    totalFileList.append( (dbName,fileName))

            print("Total dataNum: {}\n".format(len(totalFileList)))
            print(sampleNum)


            #Random 1K samples
            expectedSampleNum = 500
            randIds = np.random.permutation(len(totalFileList))[:expectedSampleNum]
            selectedSampleList = [totalFileList[i]  for i  in randIds ]

            #Selected Stats
            for s in selectedSampleList:
                selectedSampleNum[s[0]] +=1
            print(selectedSampleNum)

            
            with open(selectionOutoutTxt, 'w') as f:
                json.dump(selectedSampleList, f, indent=4)
                f.close()

            statTxt = os.path.join(rootDirName, 'stats.json')
            with open(statTxt, 'w') as f:
                json.dump(sampleNum, f, indent=4)
                json.dump(selectedSampleNum, f, indent=4)
                f.close()
        else:
            with open(selectionOutoutTxt, 'r') as f:
                selectedSampleList= json.load(f)
                f.close()
        # print(selectedSampleList)
    else:
        totalFileList = []
        for dbName in inputDir_list:
            inputDir = inputDir_list[dbName]
            fileList  = listdir(inputDir)
            for fileName in fileList:
                
                if bRenderInit:
                    if ("_init" in fileName):
                        totalFileList.append( (dbName,fileName))
                else:
                    if ("_init" in fileName) == False:
                        totalFileList.append( (dbName,fileName))
        
        selectedSampleList = totalFileList

    # #Generate Output
    rejectedNum = 0
    
    loss_keypoints_2d_init_list=[]
    loss_keypoints_2d_list =[]
    maxShape_list =[]
    for idx, sample in enumerate(sorted(selectedSampleList)):

        dbName = sample[0]
        
          
                            
        fileName = sample[1]
        inputDir = inputDir_list[dbName]
        fileFullPath = join(inputDir, fileName)


        # if '246' in fileName:
        #     print(data['loss_keypoints_2d'])
        #     print(data['pred_shape'])
        #     here=0

        
        # print(fileFullPath)
        with open(fileFullPath,'rb') as f:
            data = pickle.load(f)

       

        if False:    #Cchkeing loss distribution
            loss_keypoints_2d_init_list.append(data['loss_keypoints_2d_init'])
            loss_keypoints_2d_list.append(data['loss_keypoints_2d'])
            maxShape_list.append( np.max(abs(data['pred_shape'])))
            continue

        if bRenderInit:
            fileFullPath_newName = join(inputDir, fileName.replace("_init",""))
            with open(fileFullPath_newName,'rb') as f:
                datameta = pickle.load(f)
                data['pred_camera'] = datameta['pred_camera']


        bValidityCheck = False
        if bValidityCheck:
            if IsValid(data) == False:
                rejectedNum+=1
                print("Rejected: so far{}".format(rejectedNum))
                continue

        # #Load meta data
        # metaDir = metaDir_list[dbName]
        # meta_fileFullPath = join(metaDir, fileName)
        # with open(meta_fileFullPath,'rb') as f:
        #     metadata = pickle.load(f)
        #     data['opt_beta'] = metadata['opt_beta']
        #     data['opt_pose'] = metadata['opt_pose']
        

        imagePath = data['imageName'][0]

        # imgFullPath =os.path.join(imgDir[dbName], os.path.basename(imagePath) )
        if dbName=='pennaction' or dbName =='posetrack':
            imgFullPath =os.path.join(imgDir[dbName], os.path.basename(os.path.dirname(imagePath)), os.path.basename(imagePath) )
        elif dbName=='panoptic':
                imgFullPath =os.path.join(imgDir[dbName], os.path.basename( os.path.dirname(os.path.dirname(imagePath))),  os.path.basename(os.path.dirname(imagePath)), os.path.basename(imagePath) )
        else:
            # imgFullPath =os.path.join(imgDir, os.path.basename(imgFullPath) )
            imgFullPath =os.path.join(imgDir[dbName], os.path.basename(imagePath) )
            
        print(imgFullPath)


        # if idx==362:
        #     here=9
        if False:
            if IsValid(data) == False:
                print(idx)
                outputFileName = '{0}-{1}-{2}.jpg'.format(dbName, os.path.basename(imgFullPath), idx )
                mergedImgFileName = '{0}/{1}'.format(mergedImageFolder,outputFileName )
                if os.path.exists(mergedImgFileName)==False:
                    continue
                mergedImg = cv2.imread(mergedImgFileName)
                viewer2D.ImShow(mergedImg,waitTime=0)
                continue
            # if os.path.exists(mergedImgFileName):


        outputFileName = '{0}-{1}-{2}.jpg'.format(dbName, os.path.basename(imgFullPath), idx )
        mergedImgFileName = '{0}/{1}'.format(mergedImageFolder,outputFileName )
        if os.path.exists(mergedImgFileName):
            continue


        assert os.path.exists(imgFullPath)
        rawImg = cv2.imread(imgFullPath)

        scale = data['scale'][0]
        center = data['center'][0]
        croppedImg_highRes = crop(rawImg, center, scale, 
                    [constants.IMG_RES*3, constants.IMG_RES*3])

        #Crop image
        croppedImg = crop(rawImg, center, scale, 
                    [constants.IMG_RES, constants.IMG_RES])

        if croppedImg is None:
            print("Warning: this sample is wrong\n")
            viewer2D.ImShow(rawImg)
            continue


        validity = data['keypoint2d'][0,:,2] ==1
        valid_keypoints = data['keypoint2d'][0,validity,:2]
        min_pt = np.min(valid_keypoints, axis=0)
        max_pt = np.max(valid_keypoints, axis=0)
        bbox= [ min_pt[0], min_pt[1], max_pt[0] - min_pt[0], max_pt[1] - min_pt[1]]                    

        bbox[0] = max(bbox[0]-20,0)
        bbox[1] = max(bbox[1]-20,0)
        bbox[2] = bbox[2]+20
        bbox[3] = bbox[3]+20

        rawImage_wSkel = viewer2D.Vis_Skeleton_2D_SPIN49(data['keypoint2d'][0,:,:2], pt2d_visibility=data['keypoint2d'][0,:,2], image= rawImg.copy())
        rawImage_wSkel = viewer2D.Vis_Bbox(rawImage_wSkel,bbox)
        croppedImg_highRes_withSkel = crop(rawImage_wSkel, center, scale, 
                    [constants.IMG_RES*3, constants.IMG_RES*3])

        if False:
            viewer2D.ImShow(croppedImg_highRes,name='croppedImg_highRes', waitTime=1)
            viewer2D.ImShow(croppedImg_highRes_withSkel,waitTime=1,name='croppedImg_highRes_withSkel')

            viewer2D.ImShow(croppedImg,name='croppedImg', waitTime=1)
            viewer2D.ImShow(rawImg,waitTime=0)


        #Export Image
        if True:
            imgFilePath = os.path.join(imageFolder,outputFileName)
            cv2.imwrite(imgFilePath,rawImg)

            imgFilePath = os.path.join(croppedimageFolder,outputFileName) 
            cv2.imwrite(imgFilePath,croppedImg_highRes)

            imgFilePath = os.path.join(skelCroppedimageFolder_,outputFileName)  
            cv2.imwrite(imgFilePath,croppedImg_highRes_withSkel)

        #Vis overlaid Img
        ours_betas = torch.from_numpy(data['pred_shape'])
        ours_pose_rotmat = torch.from_numpy(data['pred_pose_rotmat'])
        spin_betas = torch.from_numpy(data['opt_beta'])
        spin_pose = torch.from_numpy(data['opt_pose'])
        pred_camera_vis = data['pred_camera']


        #Visualize SMPL output
        # Note that gt_model_joints is different from gt_joints as it comes from SMPL
        ours_output = smpl(betas=ours_betas, body_pose=ours_pose_rotmat[:,1:], global_orient=ours_pose_rotmat[:,0].unsqueeze(1), pose2rot=False )
        ours_vertices = ours_output.vertices.detach().cpu().numpy() 
        ours_joints_3d = ours_output.joints.detach().cpu().numpy() 

        spin_output = smpl(betas=spin_betas, body_pose=spin_pose[:,3:], global_orient=spin_pose[:,:3])
        spin_vertices = spin_output.vertices.detach().cpu().numpy() 
        spin_joints_3d = spin_output.joints.detach().cpu().numpy() 

        #Centering
        if False:
            spin_pelvis = (ours_joints_3d[:, 27:28,:3] + ours_joints_3d[:,28:29,:3]) / 2
            ours_joints_3d[:,:,:3] = ours_joints_3d[:,:,:3] - spin_pelvis        #centering
            ours_vertices = ours_vertices - spin_pelvis

            spin_model_pelvis = (spin_joints_3d[:, 27:28,:3] + spin_joints_3d[:,28:29,:3]) / 2
            spin_joints_3d[:,:,:3] = spin_joints_3d[:,:,:3] - spin_model_pelvis        #centering
            spin_vertices = spin_vertices - spin_model_pelvis

        b =0
        ############### Visualize Mesh ############### 
        pred_vert_vis = ours_vertices[b]
        # meshVertVis = gt_vertices[b].detach().cpu().numpy() 
        # meshVertVis = meshVertVis-pelvis        #centering
        pred_vert_vis *=pred_camera_vis[b,0]
        pred_vert_vis[:,0] += pred_camera_vis[b,1]        #no need +1 (or  112). Rendernig has this offset already
        pred_vert_vis[:,1] += pred_camera_vis[b,2]        #no need +1 (or  112). Rendernig has this offset already
        pred_vert_vis*=112
        pred_meshes = {'ver': pred_vert_vis, 'f': smpl.faces}


        opt_vert_vis = spin_vertices[b]
        opt_vert_vis *=pred_camera_vis[b,0]
        opt_vert_vis[:,0] += pred_camera_vis[b,1]        #no need +1 (or  112). Rendernig has this offset already
        opt_vert_vis[:,1] += pred_camera_vis[b,2]        #no need +1 (or  112). Rendernig has this offset already
        opt_vert_vis*=112
        opt_meshes = {'ver': opt_vert_vis, 'f': smpl.faces}

        # glViewer.setMeshData([pred_meshes, opt_meshes], bComputeNormal= True)
        # glViewer.setMeshData([pred_meshes], bComputeNormal= True)
        # glViewer.setMeshData([pred_meshes, opt_meshes], bComputeNormal= True)


        # glViewer.setMeshData([pred_meshes, opt_meshes], bComputeNormal= True)

        if bVisSpinOriginal:
            glViewer.setMeshData([opt_meshes], bComputeNormal= True)
        else:
            glViewer.setMeshData([pred_meshes], bComputeNormal= True)

        glViewer.setBackgroundTexture(croppedImg)
        glViewer.setWindowSize(croppedImg.shape[1]*3, croppedImg.shape[0]*3)
        glViewer.SetOrthoCamera(True)

        # glViewer.show(0)
        # continue


        if True:#g_bRenderFile:   #Save to File
            glViewer.SetNearPlane(500)
            glViewer.setWindowSize(croppedImg.shape[1]*3, croppedImg.shape[0]*3)
            glViewer.show_SMPL_cameraView(True)


            overlaidImageFile = os.path.join(overlaidImageFolder,outputFileName)  
            rendered_img_source = g_renderDir + '/scene_00000000.jpg'
            cmd = "cp -v {0} {1}".format(rendered_img_source, overlaidImageFile)
            subprocess.call(cmd,shell=True)


            glViewer.SetNearPlane(500)
            glViewer.setWindowSize(croppedImg.shape[1]*3, croppedImg.shape[0]*3)
            glViewer.show_SMPL_cameraView(True, bShowBG =False)

            noBGoverlaidImageFile = os.path.join(noBG_overlaidImageFolder,outputFileName)  
            rendered_img_source = g_renderDir + '/scene_00000000.jpg'
            cmd = "cp -v {0} {1}".format(rendered_img_source, noBGoverlaidImageFile)
            subprocess.call(cmd,shell=True)




            glViewer.SetNearPlane(50)
            glViewer.setWindowSize(croppedImg.shape[1]*3, croppedImg.shape[0]*3)
            glViewer.show_SMPL_sideView(True)

            sideImageFileName = os.path.join(sideImageFolder,outputFileName)  
            rendered_img_source = g_renderDir + '/scene_00000000.jpg'
            cmd = "cp -v {0} {1}".format(rendered_img_source, sideImageFileName)
            subprocess.call(cmd,shell=True)

        else:
            glViewer.SetNearPlane(50)
            glViewer.setWindowSize(croppedImg.shape[1]*3, croppedImg.shape[0]*3)
            glViewer.show_SMPL_sideView(True)

            # glViewer.show(0)

        
         #Merge images
        if True:
            # croppedImg_highRes  + overlaidImageFile + sideImageFileName
            overlaidImg = cv2.imread(overlaidImageFile)          
            sideImg = cv2.imread(sideImageFileName)

            overlaidImg_vis = cv2.resize(overlaidImg, (croppedImg_highRes.shape[1], croppedImg_highRes.shape[0]))
            sideImg_vis = cv2.resize(sideImg, (croppedImg_highRes.shape[1], croppedImg_highRes.shape[0]))
            mergedImg = np.concatenate( (croppedImg_highRes, overlaidImg_vis, sideImg_vis), axis=1)

            # viewer2D.ImShow(overlaidImg)
            mergedImgFileName = '{0}/{1}'.format(mergedImageFolder,outputFileName )
            cv2.imwrite(mergedImgFileName, mergedImg)


    viewer2D.Plot(loss_keypoints_2d_init_list,title='loss_keypoints_2d_init_list')
    viewer2D.Plot(loss_keypoints_2d_list,title='loss_keypoints_2d_list')
    viewer2D.Plot(maxShape_list, title='maxShape_list')




def RenderFitting_allSamples(inputDir_list, bVisSpinOriginal=False, bSampling= True, bRenderInit = False):

    g_rootDirName = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_allSampleVis'

    # metaDir_list ={}
    # metaDir_list['coco'] ='/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/meta_coco_2dskeletons'
    # metaDir_list['mpii'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/meta_mpii_2dskeletons'
    # metaDir_list['lspet'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/meta_lspet_2dskeletons'
    # # metaDir_list['pennaction'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/meta_lspet_2dskeletons'

    imgDir ={}
    imgDir['mpii'] = '/run/media/hjoo/disk/data/mpii_human_pose_v1/images'
    imgDir['lsp'] = '/run/media/hjoo/disk/data/lsp_dataset_original/images/'
    imgDir['lspet'] = '/run/media/hjoo/disk/data/lspet_dataset/images_highres'
    imgDir['coco'] =  '/run/media/hjoo/disk/data/coco/train2014'
    imgDir['pennaction'] =  '/run/media/hjoo/disk/data/Penn_Action/frames'
    imgDir['panoptic'] =  '/run/media/hjoo/disk/data/panoptic_mtc/a4_release/hdImgs'

    
    totalFileList = []
    for dbName in inputDir_list:
        
        rootDirName_db = os.path.join(g_rootDirName, os.path.basename(inputDir_list[dbName]))
        if bVisSpinOriginal:
            glViewer.SetMeshColor('red')
        else:
            glViewer.SetMeshColor('blue')
        
        if False== os.path.exists(rootDirName_db):
            os.mkdir(rootDirName_db)
            
        #Folder Name
        imageFolder = os.path.join(rootDirName_db, 'images')
        if not os.path.exists(imageFolder):
            os.mkdir(imageFolder)

        croppedimageFolder = os.path.join(rootDirName_db, 'croppedImage')
        if not os.path.exists(croppedimageFolder):
            os.mkdir(croppedimageFolder)

        skelCroppedimageFolder_ = os.path.join(rootDirName_db, 'skelCroppedImage')
        if not os.path.exists(skelCroppedimageFolder_):
            os.mkdir(skelCroppedimageFolder_)

        overlaidImageFolder = os.path.join(rootDirName_db, 'overlaid')
        if not os.path.exists(overlaidImageFolder):
            os.mkdir(overlaidImageFolder)

        sideImageFolder = os.path.join(rootDirName_db, 'side')
        if not os.path.exists(sideImageFolder):
            os.mkdir(sideImageFolder)

        mergedImageFolder = os.path.join(rootDirName_db, 'merged')
        if not os.path.exists(mergedImageFolder):
            os.mkdir(mergedImageFolder)




        inputDir = inputDir_list[dbName]
        fileList  = listdir(inputDir)
        for fileName in fileList:
            
            if bRenderInit:
                if ("_init" in fileName):
                    totalFileList.append( (dbName,fileName))
            else:
                if ("_init" in fileName) == False:
                    totalFileList.append( (dbName,fileName))
    
        selectedSampleList = totalFileList

        # #Generate Output
        rejectedNum = 0
        
        loss_keypoints_2d_list =[]
        maxShape_list =[]
        for idx, sample in enumerate(sorted(selectedSampleList)):

            dbName = sample[0]
            fileName = sample[1]
            inputDir = inputDir_list[dbName]
            fileFullPath = join(inputDir, fileName)


            # if '098225941' in fileName:
            #     # print(data['loss_keypoints_2d'])
            #     # print(data['pred_shape'])
            #     here=0
            # else:
            #     continue


            
            print(fileFullPath)
            with open(fileFullPath,'rb') as f:
                data = pickle.load(f)


            # loss_keypoints_2d_list.append(data['loss_keypoints_2d'])
            # maxShape_list.append( np.max(abs(data['pred_shape'])))
            # continue

            if bRenderInit:
                fileFullPath_newName = join(inputDir, fileName.replace("_init",""))
                with open(fileFullPath_newName,'rb') as f:
                    datameta = pickle.load(f)
                    data['pred_camera'] = datameta['pred_camera']

            if dbName!='panoptic':
                if IsValid(data) == False:
                    rejectedNum+=1
                    print("Rejected: so far{}".format(rejectedNum))
                    continue

                if IsValid_strict(data) == False:
                    rejectedNum+=1
                    print("Rejected: so far{}".format(rejectedNum))
                    continue

            # #Load meta data
            # metaDir = metaDir_list[dbName]
            # meta_fileFullPath = join(metaDir, fileName)
            # with open(meta_fileFullPath,'rb') as f:
            #     metadata = pickle.load(f)
            #     data['keypoint2d'] = metadata['keypoint2d']

            imagePath = data['imageName'][0]
            if dbName=='pennaction':
                imgFullPath =os.path.join(imgDir[dbName], os.path.basename(os.path.dirname(imagePath)), os.path.basename(imagePath) )
            elif dbName=='panoptic':
                    imgFullPath =os.path.join(imgDir[dbName], os.path.basename( os.path.dirname(os.path.dirname(imagePath))),  os.path.basename(os.path.dirname(imagePath)), os.path.basename(imagePath) )
            else:
                # imgFullPath =os.path.join(imgDir, os.path.basename(imgFullPath) )
                imgFullPath =os.path.join(imgDir[dbName], os.path.basename(imagePath) )

            print(imgFullPath)

            # outputFileName = '{0}-{1}-{2}.jpg'.format(dbName, os.path.basename(imgFullPath), idx )
            outputFileName = imgFullPath.replace('/','-')
            mergedImgFileName = '{0}/{1}'.format(mergedImageFolder,outputFileName )
            if os.path.exists(mergedImgFileName):
                continue

            assert os.path.exists(imgFullPath)
            rawImg = cv2.imread(imgFullPath)

            scale = data['scale'][0]
            center = data['center'][0]
            croppedImg_highRes = crop(rawImg, center, scale, 
                        [constants.IMG_RES*3, constants.IMG_RES*3])

            #Crop image
            croppedImg = crop(rawImg, center, scale, 
                        [constants.IMG_RES, constants.IMG_RES])

            validity = data['keypoint2d'][0,:,2] ==1
            valid_keypoints = data['keypoint2d'][0,validity,:2]
            min_pt = np.min(valid_keypoints, axis=0)
            max_pt = np.max(valid_keypoints, axis=0)
            bbox= [ min_pt[0], min_pt[1], max_pt[0] - min_pt[0], max_pt[1] - min_pt[1]]                    

            bbox[0] = max(bbox[0]-20,0)
            bbox[1] = max(bbox[1]-20,0)
            bbox[2] = bbox[2]+20
            bbox[3] = bbox[3]+20

            rawImage_wSkel = viewer2D.Vis_Skeleton_2D_SPIN49(data['keypoint2d'][0,:,:2], pt2d_visibility=data['keypoint2d'][0,:,2], image= rawImg.copy())
            rawImage_wSkel = viewer2D.Vis_Bbox(rawImage_wSkel,bbox)
            croppedImg_highRes_withSkel = crop(rawImage_wSkel, center, scale, 
                        [constants.IMG_RES*3, constants.IMG_RES*3])

            if False:
                viewer2D.ImShow(croppedImg_highRes,name='croppedImg_highRes', waitTime=1)
                viewer2D.ImShow(croppedImg_highRes_withSkel,waitTime=1,name='croppedImg_highRes_withSkel')

                viewer2D.ImShow(croppedImg,name='croppedImg', waitTime=1)
                viewer2D.ImShow(rawImg,waitTime=0)


            #Export Image
            if True:
                imgFilePath = os.path.join(imageFolder,outputFileName)
                cv2.imwrite(imgFilePath,rawImg)

                imgFilePath = os.path.join(croppedimageFolder,outputFileName) 
                cv2.imwrite(imgFilePath,croppedImg_highRes)

                imgFilePath = os.path.join(skelCroppedimageFolder_,outputFileName)  
                cv2.imwrite(imgFilePath,croppedImg_highRes_withSkel)

            #Vis overlaid Img
            ours_betas = torch.from_numpy(data['pred_shape'])
            ours_pose_rotmat = torch.from_numpy(data['pred_pose_rotmat'])
            spin_betas = torch.from_numpy(data['opt_beta'])
            spin_pose = torch.from_numpy(data['opt_pose'])
            pred_camera_vis = data['pred_camera']


            #Visualize SMPL output
            # Note that gt_model_joints is different from gt_joints as it comes from SMPL
            ours_output = smpl(betas=ours_betas, body_pose=ours_pose_rotmat[:,1:], global_orient=ours_pose_rotmat[:,0].unsqueeze(1), pose2rot=False )
            ours_vertices = ours_output.vertices.detach().cpu().numpy() 
            ours_joints_3d = ours_output.joints.detach().cpu().numpy() 

            spin_output = smpl(betas=spin_betas, body_pose=spin_pose[:,3:], global_orient=spin_pose[:,:3])
            spin_vertices = spin_output.vertices.detach().cpu().numpy() 
            spin_joints_3d = spin_output.joints.detach().cpu().numpy() 

            #Centering
            if False:
                spin_pelvis = (ours_joints_3d[:, 27:28,:3] + ours_joints_3d[:,28:29,:3]) / 2
                ours_joints_3d[:,:,:3] = ours_joints_3d[:,:,:3] - spin_pelvis        #centering
                ours_vertices = ours_vertices - spin_pelvis

                spin_model_pelvis = (spin_joints_3d[:, 27:28,:3] + spin_joints_3d[:,28:29,:3]) / 2
                spin_joints_3d[:,:,:3] = spin_joints_3d[:,:,:3] - spin_model_pelvis        #centering
                spin_vertices = spin_vertices - spin_model_pelvis

            b =0
            ############### Visualize Mesh ############### 
            pred_vert_vis = ours_vertices[b]
            # meshVertVis = gt_vertices[b].detach().cpu().numpy() 
            # meshVertVis = meshVertVis-pelvis        #centering
            pred_vert_vis *=pred_camera_vis[b,0]
            pred_vert_vis[:,0] += pred_camera_vis[b,1]        #no need +1 (or  112). Rendernig has this offset already
            pred_vert_vis[:,1] += pred_camera_vis[b,2]        #no need +1 (or  112). Rendernig has this offset already
            pred_vert_vis*=112
            pred_meshes = {'ver': pred_vert_vis, 'f': smpl.faces}


            opt_vert_vis = spin_vertices[b]
            opt_vert_vis *=pred_camera_vis[b,0]
            opt_vert_vis[:,0] += pred_camera_vis[b,1]        #no need +1 (or  112). Rendernig has this offset already
            opt_vert_vis[:,1] += pred_camera_vis[b,2]        #no need +1 (or  112). Rendernig has this offset already
            opt_vert_vis*=112
            opt_meshes = {'ver': opt_vert_vis, 'f': smpl.faces}

            # glViewer.setMeshData([pred_meshes, opt_meshes], bComputeNormal= True)
            # glViewer.setMeshData([pred_meshes], bComputeNormal= True)
            # glViewer.setMeshData([pred_meshes, opt_meshes], bComputeNormal= True)

            if bVisSpinOriginal:
                glViewer.setMeshData([opt_meshes], bComputeNormal= True)
            else:
                glViewer.setMeshData([pred_meshes], bComputeNormal= True)

            glViewer.setBackgroundTexture(croppedImg)
            glViewer.setWindowSize(croppedImg.shape[1]*3, croppedImg.shape[0]*3)
            glViewer.SetOrthoCamera(True)

            # glViewer.show(0)
            # continue


            if True:#g_bRenderFile:   #Save to File
                glViewer.SetNearPlane(500)
                glViewer.setWindowSize(croppedImg.shape[1]*3, croppedImg.shape[0]*3)
                glViewer.show_SMPL_cameraView(True)


                overlaidImageFile = os.path.join(overlaidImageFolder,outputFileName)  
                rendered_img_source = g_renderDir + '/scene_00000000.jpg'
                cmd = "cp -v {0} {1}".format(rendered_img_source, overlaidImageFile)
                subprocess.call(cmd,shell=True)


                glViewer.SetNearPlane(50)
                glViewer.setWindowSize(croppedImg.shape[1]*3, croppedImg.shape[0]*3)
                glViewer.show_SMPL_sideView(True)

                sideImageFileName = os.path.join(sideImageFolder,outputFileName)  
                rendered_img_source = g_renderDir + '/scene_00000000.jpg'
                cmd = "cp -v {0} {1}".format(rendered_img_source, sideImageFileName)
                subprocess.call(cmd,shell=True)

            else:
                glViewer.SetNearPlane(50)
                glViewer.setWindowSize(croppedImg.shape[1]*3, croppedImg.shape[0]*3)
                glViewer.show_SMPL_sideView(True)

                # glViewer.show(0)

            
            #Merge images
            if True:
                # croppedImg_highRes  + overlaidImageFile + sideImageFileName
                overlaidImg = cv2.imread(overlaidImageFile)          
                sideImg = cv2.imread(sideImageFileName)

                overlaidImg_vis = cv2.resize(overlaidImg, (croppedImg_highRes.shape[1], croppedImg_highRes.shape[0]))
                sideImg_vis = cv2.resize(sideImg, (croppedImg_highRes.shape[1], croppedImg_highRes.shape[0]))
                mergedImg = np.concatenate( (croppedImg_highRes, overlaidImg_vis, sideImg_vis), axis=1)

                # viewer2D.ImShow(overlaidImg)
                mergedImgFileName = '{0}/{1}'.format(mergedImageFolder,outputFileName )


                #Write output here
                maxShapeBeta = abs(torch.max( abs(ours_betas)).item())
                finalKeyloss = data['loss_keypoints_2d']
                reconErrorStr_raw = "m_beta {:.2f} | l_2d {:.8f}".format(maxShapeBeta, finalKeyloss)
                mergedImg = cv2.putText(mergedImg,reconErrorStr_raw, (50,mergedImg.shape[0]-50), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0),2)

                
                cv2.imwrite(mergedImgFileName, mergedImg)

    viewer2D.Plot(loss_keypoints_2d_list)
    viewer2D.Plot(maxShape_list)


def RenderFitting_allSamples_panoptic(inputDir_list, bVisSpinOriginal=False, bSampling= True, bRenderInit = False):

    #1K random selection file
    g_rootDirName = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_allSampleVis'
    # selectionFileDir ='/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_quant/AMT_1Kset_fromChallenging'
    # selectionOutoutTxt = os.path.join(selectionFileDir, 'selected1K.json')


    metaDir_list ={}
    metaDir_list['coco'] ='/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/meta_coco_2dskeletons'
    metaDir_list['mpii'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/meta_mpii_2dskeletons'
    metaDir_list['lspet'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/meta_lspet_2dskeletons'
    # metaDir_list['pennaction'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/meta_lspet_2dskeletons'

    imgDir ={}
    imgDir['mpii'] = '/run/media/hjoo/disk/data/mpii_human_pose_v1/images'
    imgDir['lspet'] = '/run/media/hjoo/disk/data/lspet_dataset/images_highres'
    imgDir['coco'] =  '/run/media/hjoo/disk/data/coco/train2014'
    imgDir['pennaction'] =  '/run/media/hjoo/disk/data/Penn_Action/frames'
    imgDir['panoptic'] =  '/run/media/hjoo/disk/data/panoptic_mtc/a4_release/hdImgs'

    
    totalFileList = []
    for dbName in inputDir_list:
        
        rootDirName_db = os.path.join(g_rootDirName, os.path.basename(inputDir_list[dbName]))
        if bVisSpinOriginal:
            glViewer.SetMeshColor('red')
        else:
            glViewer.SetMeshColor('blue')
        
        if False== os.path.exists(rootDirName_db):
            os.mkdir(rootDirName_db)
            
        #Folder Name
        imageFolder = os.path.join(rootDirName_db, 'images')
        if not os.path.exists(imageFolder):
            os.mkdir(imageFolder)

        croppedimageFolder = os.path.join(rootDirName_db, 'croppedImage')
        if not os.path.exists(croppedimageFolder):
            os.mkdir(croppedimageFolder)

        skelCroppedimageFolder_ = os.path.join(rootDirName_db, 'skelCroppedImage')
        if not os.path.exists(skelCroppedimageFolder_):
            os.mkdir(skelCroppedimageFolder_)

        overlaidImageFolder = os.path.join(rootDirName_db, 'overlaid')
        if not os.path.exists(overlaidImageFolder):
            os.mkdir(overlaidImageFolder)

        sideImageFolder = os.path.join(rootDirName_db, 'side')
        if not os.path.exists(sideImageFolder):
            os.mkdir(sideImageFolder)

        mergedImageFolder = os.path.join(rootDirName_db, 'merged')
        if not os.path.exists(mergedImageFolder):
            os.mkdir(mergedImageFolder)




        inputDir = inputDir_list[dbName]
        fileList  = listdir(inputDir)
        for fileName in fileList:
            
            if bRenderInit:
                if ("_init" in fileName):
                    totalFileList.append( (dbName,fileName))
            else:
                if ("_init" in fileName) == False:
                    totalFileList.append( (dbName,fileName))
    
        selectedSampleList = totalFileList

        # #Generate Output
        rejectedNum = 0
        
        loss_keypoints_2d_list =[]
        maxShape_list =[]
        for idx, sample in enumerate(sorted(selectedSampleList)):

            dbName = sample[0]
            fileName = sample[1]
            inputDir = inputDir_list[dbName]
            fileFullPath = join(inputDir, fileName)


            # if '246' in fileName:
            #     print(data['loss_keypoints_2d'])
            #     print(data['pred_shape'])
            #     here=0

            print(fileFullPath)
            with open(fileFullPath,'rb') as f:
                dataList = pickle.load(f)

            # if not isinstance(dataList, dict):
            #     dataListDict ={}
            #     dataListDict['0'] = dataList
            #     dataList = dataListDict


            # dataList = dataList[5:6]
            for dataKey in dataList:
                data = dataList[dataKey]

                # loss_keypoints_2d_list.append(data['loss_keypoints_2d'])
                # maxShape_list.append( np.max(abs(data['pred_shape'])))
                # continue
                

                if bRenderInit:
                    fileFullPath_newName = join(inputDir, fileName.replace("_init",""))
                    with open(fileFullPath_newName,'rb') as f:
                        datameta = pickle.load(f)
                        data['pred_camera'] = datameta['pred_camera']


                if False:#IsValid(data) == False:
                    rejectedNum+=1
                    print("Rejected: so far{}".format(rejectedNum))
                    continue

                # #Load meta data
                # metaDir = metaDir_list[dbName]
                # meta_fileFullPath = join(metaDir, fileName)
                # with open(meta_fileFullPath,'rb') as f:
                #     metadata = pickle.load(f)
                #     data['keypoint2d'] = metadata['keypoint2d']

                imagePath = data['imageName'][0]
                if dbName=='pennaction':
                    imgFullPath =os.path.join(imgDir[dbName], os.path.basename(os.path.dirname(imagePath)), os.path.basename(imagePath) )
                elif dbName=='panoptic':
                    imgFullPath =os.path.join(imgDir[dbName], os.path.basename( os.path.dirname(os.path.dirname(imagePath))),  os.path.basename(os.path.dirname(imagePath)), os.path.basename(imagePath) )
                else:
                    # imgFullPath =os.path.join(imgDir, os.path.basename(imgFullPath) )
                    imgFullPath =os.path.join(imgDir[dbName], os.path.basename(imagePath) )

                print(imgFullPath)

                # outputFileName = '{0}-{1}-{2}.jpg'.format(dbName, os.path.basename(imgFullPath), idx )
                outputFileName = imgFullPath.replace('/','-')
                mergedImgFileName = '{0}/{1}'.format(mergedImageFolder,outputFileName )
                if os.path.exists(mergedImgFileName):
                    continue

                assert os.path.exists(imgFullPath)
                rawImg = cv2.imread(imgFullPath)

                scale = data['scale'][0]
                center = data['center'][0]
                croppedImg_highRes = crop(rawImg, center, scale, 
                            [constants.IMG_RES*3, constants.IMG_RES*3])

                #Crop image
                croppedImg = crop(rawImg, center, scale, 
                            [constants.IMG_RES, constants.IMG_RES])

                validity = data['keypoint2d'][0,:,2] ==1
                valid_keypoints = data['keypoint2d'][0,validity,:2]
                min_pt = np.min(valid_keypoints, axis=0)
                max_pt = np.max(valid_keypoints, axis=0)
                bbox= [ min_pt[0], min_pt[1], max_pt[0] - min_pt[0], max_pt[1] - min_pt[1]]                    

                bbox[0] = max(bbox[0]-20,0)
                bbox[1] = max(bbox[1]-20,0)
                bbox[2] = bbox[2]+20
                bbox[3] = bbox[3]+20

                rawImage_wSkel = viewer2D.Vis_Skeleton_2D_SPIN49(data['keypoint2d'][0,:,:2], pt2d_visibility=data['keypoint2d'][0,:,2], image= rawImg.copy())
                rawImage_wSkel = viewer2D.Vis_Bbox(rawImage_wSkel,bbox)
                croppedImg_highRes_withSkel = crop(rawImage_wSkel, center, scale, 
                            [constants.IMG_RES*3, constants.IMG_RES*3])

                if False:
                    viewer2D.ImShow(croppedImg_highRes,name='croppedImg_highRes', waitTime=1)
                    viewer2D.ImShow(croppedImg_highRes_withSkel,waitTime=1,name='croppedImg_highRes_withSkel')

                    viewer2D.ImShow(croppedImg,name='croppedImg', waitTime=1)
                    viewer2D.ImShow(rawImg,waitTime=0)


                #Export Image
                if True:
                    imgFilePath = os.path.join(imageFolder,outputFileName)
                    cv2.imwrite(imgFilePath,rawImg)

                    imgFilePath = os.path.join(croppedimageFolder,outputFileName) 
                    cv2.imwrite(imgFilePath,croppedImg_highRes)

                    imgFilePath = os.path.join(skelCroppedimageFolder_,outputFileName)  
                    cv2.imwrite(imgFilePath,croppedImg_highRes_withSkel)

                #Vis overlaid Img
                ours_betas = torch.from_numpy(data['pred_shape'])
                ours_pose_rotmat = torch.from_numpy(data['pred_pose_rotmat'])
                spin_betas = torch.from_numpy(data['opt_beta'])
                spin_pose = torch.from_numpy(data['opt_pose'])
                pred_camera_vis = data['pred_camera']


                #Visualize SMPL output
                # Note that gt_model_joints is different from gt_joints as it comes from SMPL
                ours_output = smpl(betas=ours_betas, body_pose=ours_pose_rotmat[:,1:], global_orient=ours_pose_rotmat[:,0].unsqueeze(1), pose2rot=False )
                ours_vertices = ours_output.vertices.detach().cpu().numpy() 
                ours_joints_3d = ours_output.joints.detach().cpu().numpy() 

                spin_output = smpl(betas=spin_betas, body_pose=spin_pose[:,3:], global_orient=spin_pose[:,:3])
                spin_vertices = spin_output.vertices.detach().cpu().numpy() 
                spin_joints_3d = spin_output.joints.detach().cpu().numpy() 

                #Centering
                if False:
                    spin_pelvis = (ours_joints_3d[:, 27:28,:3] + ours_joints_3d[:,28:29,:3]) / 2
                    ours_joints_3d[:,:,:3] = ours_joints_3d[:,:,:3] - spin_pelvis        #centering
                    ours_vertices = ours_vertices - spin_pelvis

                    spin_model_pelvis = (spin_joints_3d[:, 27:28,:3] + spin_joints_3d[:,28:29,:3]) / 2
                    spin_joints_3d[:,:,:3] = spin_joints_3d[:,:,:3] - spin_model_pelvis        #centering
                    spin_vertices = spin_vertices - spin_model_pelvis

                b =0
                ############### Visualize Mesh ############### 
                pred_vert_vis = ours_vertices[b]
                # meshVertVis = gt_vertices[b].detach().cpu().numpy() 
                # meshVertVis = meshVertVis-pelvis        #centering
                pred_vert_vis *=pred_camera_vis[b,0]
                pred_vert_vis[:,0] += pred_camera_vis[b,1]        #no need +1 (or  112). Rendernig has this offset already
                pred_vert_vis[:,1] += pred_camera_vis[b,2]        #no need +1 (or  112). Rendernig has this offset already
                pred_vert_vis*=112
                pred_meshes = {'ver': pred_vert_vis, 'f': smpl.faces}


                opt_vert_vis = spin_vertices[b]
                opt_vert_vis *=pred_camera_vis[b,0]
                opt_vert_vis[:,0] += pred_camera_vis[b,1]        #no need +1 (or  112). Rendernig has this offset already
                opt_vert_vis[:,1] += pred_camera_vis[b,2]        #no need +1 (or  112). Rendernig has this offset already
                opt_vert_vis*=112
                opt_meshes = {'ver': opt_vert_vis, 'f': smpl.faces}

                # glViewer.setMeshData([pred_meshes, opt_meshes], bComputeNormal= True)
                # glViewer.setMeshData([pred_meshes], bComputeNormal= True)
                # glViewer.setMeshData([pred_meshes, opt_meshes], bComputeNormal= True)

                if bVisSpinOriginal:
                    glViewer.setMeshData([opt_meshes], bComputeNormal= True)
                else:
                    glViewer.setMeshData([pred_meshes], bComputeNormal= True)

                glViewer.setBackgroundTexture(croppedImg)
                glViewer.setWindowSize(croppedImg.shape[1]*3, croppedImg.shape[0]*3)
                glViewer.SetOrthoCamera(True)

                # glViewer.show(0)
                # continue


                if True:#g_bRenderFile:   #Save to File
                    glViewer.SetNearPlane(500)
                    glViewer.setWindowSize(croppedImg.shape[1]*3, croppedImg.shape[0]*3)
                    glViewer.show_SMPL_cameraView(True)


                    overlaidImageFile = os.path.join(overlaidImageFolder,outputFileName)  
                    rendered_img_source = g_renderDir + '/scene_00000000.jpg'
                    cmd = "cp -v {0} {1}".format(rendered_img_source, overlaidImageFile)
                    subprocess.call(cmd,shell=True)


                    glViewer.SetNearPlane(50)
                    glViewer.setWindowSize(croppedImg.shape[1]*3, croppedImg.shape[0]*3)
                    glViewer.show_SMPL_sideView(True)

                    sideImageFileName = os.path.join(sideImageFolder,outputFileName)  
                    rendered_img_source = g_renderDir + '/scene_00000000.jpg'
                    cmd = "cp -v {0} {1}".format(rendered_img_source, sideImageFileName)
                    subprocess.call(cmd,shell=True)

                else:
                    glViewer.SetNearPlane(50)
                    glViewer.setWindowSize(croppedImg.shape[1]*3, croppedImg.shape[0]*3)
                    glViewer.show_SMPL_sideView(True)

                    # glViewer.show(0)

                
                #Merge images
                if True:
                    # croppedImg_highRes  + overlaidImageFile + sideImageFileName
                    overlaidImg = cv2.imread(overlaidImageFile)          
                    sideImg = cv2.imread(sideImageFileName)

                    overlaidImg_vis = cv2.resize(overlaidImg, (croppedImg_highRes.shape[1], croppedImg_highRes.shape[0]))
                    sideImg_vis = cv2.resize(sideImg, (croppedImg_highRes.shape[1], croppedImg_highRes.shape[0]))
                    mergedImg = np.concatenate( (croppedImg_highRes, overlaidImg_vis, sideImg_vis), axis=1)

                    # viewer2D.ImShow(overlaidImg)
                    mergedImgFileName = '{0}/{1}'.format(mergedImageFolder,outputFileName )
                    cv2.imwrite(mergedImgFileName, mergedImg)
                break
        # viewer2D.Plot(loss_keypoints_2d_list)
        # viewer2D.Plot(maxShape_list)




def RenderFitting_evalOutput(exportedFile_pkl, bVisSpinOriginal=False, bSampling= True, bRenderInit = False, bUpperMode= False):
    
    with open(exportedFile_pkl, 'rb') as f:
        data = pickle.load(f)
        f.close()

    imgDir ={}
    imgDir['mpii'] = '/run/media/hjoo/disk/data/mpii_human_pose_v1/images'
    imgDir['lspet'] = '/run/media/hjoo/disk/data/lspet_dataset/images_highres'
    imgDir['coco'] =  '/run/media/hjoo/disk/data/coco/train2014'
    imgDir['pennaction'] =  '/run/media/hjoo/disk/data/Penn_Action/frames'
    imgDir['panoptic'] =  '/run/media/hjoo/disk/data/panoptic_mtc/a4_release/hdImgs'
    imgDir['3dpw'] =  '/run/media/hjoo/disk/data/3dpw/imageFiles'

    
    totalFileList = []
    
    rootDirName_db = os.path.dirname(exportedFile_pkl)
    if bVisSpinOriginal:
        glViewer.SetMeshColor('red')
    else:
        glViewer.SetMeshColor('blue')
    
    if False== os.path.exists(rootDirName_db):
        os.mkdir(rootDirName_db)
        
    #Folder Name
    imageFolder = os.path.join(rootDirName_db, 'images')
    if not os.path.exists(imageFolder):
        os.mkdir(imageFolder)

    croppedimageFolder = os.path.join(rootDirName_db, 'croppedImage')
    if not os.path.exists(croppedimageFolder):
        os.mkdir(croppedimageFolder)

    skelCroppedimageFolder_ = os.path.join(rootDirName_db, 'skelCroppedImage')
    if not os.path.exists(skelCroppedimageFolder_):
        os.mkdir(skelCroppedimageFolder_)

    overlaidImageFolder = os.path.join(rootDirName_db, 'overlaid')
    if not os.path.exists(overlaidImageFolder):
        os.mkdir(overlaidImageFolder)

    sideImageFolder = os.path.join(rootDirName_db, 'side')
    if not os.path.exists(sideImageFolder):
        os.mkdir(sideImageFolder)

    mergedImageFolder = os.path.join(rootDirName_db, 'merged')
    if not os.path.exists(mergedImageFolder):
        os.mkdir(mergedImageFolder)

    imageDir = imgDir['3dpw']
    
    # #Generate Output
    rejectedNum = 0
    frameNum = data['pred_pose'].shape[0]
    for idx in range(frameNum):

        # data['pred_pose'][idx]
        # data['pred_betas'][idx]
        # data['pred_camera'][idx]
        # data['pred_joints'][idx]
        # data['gt_pose'][idx]
        # data['gt_betas'][idx]
        # data['gt_joints'][idx]
        # data['error_MPJPE'][idx]
        # data['error_recon'][idx]
        # data['cropScale'][idx]
        # data['cropCenter'][idx]
        # data['imageNames'][idx]
        
        if True:
            imagePath = data['imageNames'][idx]
            imgFullPath =os.path.join(imageDir, os.path.basename(os.path.dirname(imagePath)), os.path.basename(imagePath) )
            print(imgFullPath)

            # outputFileName = '{0}-{1}-{2}.jpg'.format(dbName, os.path.basename(imgFullPath), idx )
            outputFileName = imgFullPath.replace('/','-')
            mergedImgFileName = '{0}/{1}'.format(mergedImageFolder,outputFileName )
           
           
            #Skip if exists
            if os.path.exists(mergedImgFileName):
                continue



            assert os.path.exists(imgFullPath)
            rawImg = cv2.imread(imgFullPath)

            scale = data['cropScale'][idx]
            center = data['cropCenter'][idx]
            croppedImg_highRes = crop(rawImg, center, scale, 
                        [constants.IMG_RES*3, constants.IMG_RES*3])

            #Crop image
            croppedImg = crop(rawImg, center, scale, 
                        [constants.IMG_RES, constants.IMG_RES])

        if False:
            viewer2D.ImShow(croppedImg_highRes,name='croppedImg_highRes', waitTime=1)
            viewer2D.ImShow(croppedImg_highRes_withSkel,waitTime=1,name='croppedImg_highRes_withSkel')

            viewer2D.ImShow(croppedImg,name='croppedImg', waitTime=1)
            viewer2D.ImShow(rawImg,waitTime=0)


        #Export Image
        if True:
            imgFilePath = os.path.join(imageFolder,outputFileName)
            cv2.imwrite(imgFilePath,rawImg)

            imgFilePath = os.path.join(croppedimageFolder,outputFileName) 
            cv2.imwrite(imgFilePath,croppedImg_highRes)

        #Vis overlaid Img
        ours_betas = torch.from_numpy(data['pred_betas'][idx:idx+1].astype(np.float32))
        ours_pose = torch.from_numpy(data['pred_pose'][idx:idx+1].astype(np.float32))
        
        pred_camera_vis = data['pred_camera'][idx:idx+1]
        ours_output = smpl(betas=ours_betas, body_pose=ours_pose[:,3:], global_orient=ours_pose[:,:3])
        ours_vertices = ours_output.vertices.detach().cpu().numpy() 
        ours_joints_3d = ours_output.joints.detach().cpu().numpy() 


        if bVisSpinOriginal:
            gt_betas = torch.from_numpy(data['gt_betas'][idx:idx+1].astype(np.float32))
            gt_pose = torch.from_numpy(data['gt_pose'][idx:idx+1].astype(np.float32))
            gt_output = smpl(betas=gt_betas, body_pose=gt_pose[:,3:], global_orient=gt_pose[:,:3])
            gt_vertices = spin_output.vertices.detach().cpu().numpy() 
            gt_joints_3d = spin_output.joints.detach().cpu().numpy() 

        #Centering
        if False:
            spin_pelvis = (ours_joints_3d[:, 27:28,:3] + ours_joints_3d[:,28:29,:3]) / 2
            ours_joints_3d[:,:,:3] = ours_joints_3d[:,:,:3] - spin_pelvis        #centering
            ours_vertices = ours_vertices - spin_pelvis

            spin_model_pelvis = (spin_joints_3d[:, 27:28,:3] + spin_joints_3d[:,28:29,:3]) / 2
            spin_joints_3d[:,:,:3] = spin_joints_3d[:,:,:3] - spin_model_pelvis        #centering
            spin_vertices = spin_vertices - spin_model_pelvis

        b =0
        ############### Visualize Mesh ############### 
        pred_vert_vis = ours_vertices[b]
        # meshVertVis = gt_vertices[b].detach().cpu().numpy() 
        # meshVertVis = meshVertVis-pelvis        #centering
        pred_vert_vis *=pred_camera_vis[b,0]
        pred_vert_vis[:,0] += pred_camera_vis[b,1]        #no need +1 (or  112). Rendernig has this offset already

        if bUpperMode:
            pred_vert_vis[:,1] += pred_camera_vis[b,2] # +0.75  #UpperMode     #no need +1 (or  112). Rendernig has this offset already
        else:
            pred_vert_vis[:,1] += pred_camera_vis[b,2]   #UpperMode     #no need +1 (or  112). Rendernig has this offset already
        pred_vert_vis*=112
        pred_meshes = {'ver': pred_vert_vis, 'f': smpl.faces}
        glViewer.setMeshData([pred_meshes], bComputeNormal= True)


        if bVisSpinOriginal:
            opt_vert_vis = spin_vertices[b]
            opt_vert_vis *=pred_camera_vis[b,0]
            opt_vert_vis[:,0] += pred_camera_vis[b,1]        #no need +1 (or  112). Rendernig has this offset already
            opt_vert_vis[:,1] += pred_camera_vis[b,2]        #no need +1 (or  112). Rendernig has this offset already
            opt_vert_vis*=112
            opt_meshes = {'ver': opt_vert_vis, 'f': smpl.faces}
            glViewer.setMeshData([pred_meshes, opt_meshes], bComputeNormal= True)


        # glViewer.setMeshData([pred_meshes, opt_meshes], bComputeNormal= True)
        # glViewer.setMeshData([pred_meshes], bComputeNormal= True)

        # if bVisSpinOriginal:
        #     glViewer.setMeshData([opt_meshes], bComputeNormal= True)
        # else:
        #     glViewer.setMeshData([pred_meshes], bComputeNormal= True)

        print("Error: MPJPE {:.02f}, recon {:.02f}".format(data['error_MPJPE'][idx], data['error_recon'][idx]))
        # glViewer.show(1)
        # continue

        glViewer.setBackgroundTexture(croppedImg)
        glViewer.setWindowSize(croppedImg.shape[1]*3, croppedImg.shape[0]*3)
        glViewer.SetOrthoCamera(True)
        if True:#g_bRenderFile:   #Save to File
            glViewer.SetNearPlane(500)
            glViewer.setWindowSize(croppedImg.shape[1]*3, croppedImg.shape[0]*3)
            glViewer.show_SMPL_cameraView(True)


            overlaidImageFile = os.path.join(overlaidImageFolder,outputFileName)  
            rendered_img_source = g_renderDir + '/scene_00000000.jpg'
            cmd = "cp -v {0} {1}".format(rendered_img_source, overlaidImageFile)
            subprocess.call(cmd,shell=True)


            glViewer.SetNearPlane(50)
            glViewer.setWindowSize(croppedImg.shape[1]*3, croppedImg.shape[0]*3)
            if bUpperMode:
                glViewer.show_SMPL_sideView(True, zoom = 500)   #UpperMode
            else:
                glViewer.show_SMPL_sideView(True)   #UpperMode

            sideImageFileName = os.path.join(sideImageFolder,outputFileName)  
            rendered_img_source = g_renderDir + '/scene_00000000.jpg'
            cmd = "cp -v {0} {1}".format(rendered_img_source, sideImageFileName)
            subprocess.call(cmd,shell=True)

        else:
            glViewer.SetNearPlane(50)
            glViewer.setWindowSize(croppedImg.shape[1]*3, croppedImg.shape[0]*3)
            glViewer.show_SMPL_sideView(True)

            # glViewer.show(0)

        #Merge images
        if True:
            # croppedImg_highRes  + overlaidImageFile + sideImageFileName
            overlaidImg = cv2.imread(overlaidImageFile)          
            sideImg = cv2.imread(sideImageFileName)

            overlaidImg_vis = cv2.resize(overlaidImg, (croppedImg_highRes.shape[1], croppedImg_highRes.shape[0]))
            sideImg_vis = cv2.resize(sideImg, (croppedImg_highRes.shape[1], croppedImg_highRes.shape[0]))
            mergedImg = np.concatenate( (croppedImg_highRes, overlaidImg_vis, sideImg_vis), axis=1)

            # viewer2D.ImShow(overlaidImg)
            # reconErrorStr_raw = "MPJPE {:.2f} | Recon. {:.2f}".format(data['error_MPJPE'][idx], data['error_recon'][idx])
            reconErrorStr_raw = "{:06.2f} | {:06.2f}".format(data['error_MPJPE'][idx], data['error_recon'][idx])
            # reconErrorStr = "-{}.jpg".format(reconErrorStr_raw)
            # outputFileName = outputFileName.replace(".jpg",reconErrorStr)
            mergedImgFileName = '{0}/{1}'.format(mergedImageFolder,outputFileName)
            mergedImg = cv2.putText(mergedImg,reconErrorStr_raw, (50,mergedImg.shape[0]-50), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),2)
            viewer2D.ImShow(mergedImg,1)
            cv2.imwrite(mergedImgFileName, mergedImg)

        # viewer2D.Plot(loss_keypoints_2d_list)
        # viewer2D.Plot(maxShape_list)

if __name__ == '__main__':
    #Collect all samples
    inputDir_list ={}
    # inputDir_list['coco'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-01-51903_coco_naiveBeta'
    # inputDir_list['mpii'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-01-52066_mpii_naiveBeta'
    # inputDir_list['lspet'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-01-52093_lspet_naiveBeta'
    
    # inputDir_list['coco'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-03-68218_coco_ex_coco_original_naivebeta'
    # inputDir_list['mpii'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-03-68196_mpii_ex_mpii_original_naivebeta'
    # inputDir_list['lspet'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-03-68285_lspet_ex_lspet_original_naivebeta'
    # inputDir_list['lspet'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-04_lspet_originalCode_weak'
    # inputDir_list['lspet'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-04_lspet_originalCode_weak_meta'

    #500 sampling generation
    if False: 
        # inputDir_list['coco'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-05_coco_noHipFoot'
        # inputDir_list['mpii'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-05_mpii_noHipFoot'
        # inputDir_list['lspet'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-05_lspet_noHipFoot'

        # inputDir_list['coco'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-06_coco_legOriLoss'
        # inputDir_list['mpii'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-06_mpii_legOriLoss'
        # inputDir_list['lspet'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-06_lspet_legOriLoss'

        inputDir_list['coco'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-08_coco_with8143'
        inputDir_list['mpii'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-08_mpii_with8143'
        inputDir_list['lspet'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-08_lspet_with8143'
        # RenderFitting(inputDir_list, bVisSpinOriginal=True, bSampling = True)
        # RenderFitting(inputDir_list, bVisSpinOriginal=True, bAlwaysBlue=True, bSampling = True)
        RenderFitting(inputDir_list, bVisSpinOriginal=False, bSampling = True)

    
    #Visualize all samples
    if True: 
        # inputDir_list['lspet'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-08_lspet_with8143'
        # inputDir_list['coco'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-08_coco_with8143'
        # inputDir_list['mpii'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-08_mpii_with8143'
        # inputDir_list['coco'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-08_cocoplus_with8143'
        # inputDir_list['posetrack'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-08_posetrack_with8143'
        # inputDir_list['panoptic'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-05_panoptic_refit'
        inputDir_list['lsp'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-14_lsp_analysis_100'
        # inputDir_list['mpii'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-08_mpii_with8143'
        RenderFitting_allSamples(inputDir_list, bVisSpinOriginal=False, bSampling = True)


    # inputDir_list['pennaction'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-04_pennaction_originalCode_weak_meta'
    # inputDir_list['panoptic'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-05_panoptic_initFit'
    # inputDir_list['panoptic'] = '/home/hjoo/Dropbox (Facebook)/spinExemplar/11-05_panoptic_newtry'

    if False: #Panoptic
        # RenderFitting_allSamples(inputDir_list, bVisSpinOriginal=False, bSampling = False, bRenderInit=False)
        inputDir_list['panoptic'] = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/11-05_panoptic_refit'
        RenderFitting_allSamples_panoptic(inputDir_list, bVisSpinOriginal=False, bSampling = False, bRenderInit=False)
        # RenderFitting(inputDir_list, bVisSpinOriginal=True)

    #visualize export 
    if False:
        fileName = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_quant/SPIN_3dpw_original/result_3dpw_spin_original.pkl'      #SPIN original
        fileName = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_quant/SPIN_3dpw_upperOnly/result_3dpw_spin_upperOnly.pkl'
        fileName = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_quant/11-05-3dpw_ours_result_3dpw_ours_11_04_59961_4030_coco3d_all/result_3dpw_ours_11_04_59961_4030_coco3d_all.pkl'      #11-05 ours coco3d
        fileName = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_quant/spin_11-06-42861-upper0_2_ours_lc3d_all-8935/spin_11-06-42861-upper0_2_ours_lc3d_all-8935.pkl'
        fileName = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_quant/11-07-ours-upperOnlyTest-spin_11-06-42861-upper0_2_ours_lc3d_all-8935/spin_11-06-42861-upper0_2_ours_lc3d_all-8935.pkl'
        fileName = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_quant/11-07-ours-spin_11-06-42861-upper0_2_ours_lc3d_all-8935/spin_11-06-42861-upper0_2_ours_lc3d_all-8935.pkl'
        RenderFitting_evalOutput(fileName, bUpperMode = False)
