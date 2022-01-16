import os
from os.path import join
import sys
import json
import numpy as np
# from .read_openpose import read_openpose

import cv2
from renderer import viewer2D
from renderer import glViewer

from preprocessdb.read_openpose import read_openpose

def exportOursToSpin(cocoPose3DAll, out_path):

    scaleFactor = 1.2


    # structs we need
    imgnames_, scales_, centers_, parts_, openposes_ = [], [], [], [], []

    #additional 3D
    poses_ , shapes_, skel3D_, has_smpl_  = [], [] ,[], []


    # for imgSample in cocoPose3DAll:
    imgNum = len(cocoPose3DAll)
    totalSampleNum = [ len(cocoPose3DAll[imgSample]) for imgSample in cocoPose3DAll ]
    totalSampleNum = sum(totalSampleNum)

    print("\n\n### ImageNum: {}, SampleNum: {} ###".format(imgNum, totalSampleNum))
    # for imgSample in cocoPose3DAll:
    # for key_imgId, imgSample in sorted(cocoPose3DAll.items()):
    for key_imgId, imgSample in sorted(cocoPose3DAll.items()):
        #load image
        imgPathFull = imgSample[0]['imgId']           
        fileName = os.path.basename(imgPathFull)
        fileName_saved = os.path.join(os.path.basename(os.path.dirname(imgPathFull)), fileName) #start from train2014


        for sample in imgSample:

            validJointNum = np.sum(sample['pose2D_validity'][::2])

            if validJointNum<4:
                continue

            if np.isnan(sample['pose3DParam']['camScale']):     
                continue

            gt_skel = np.reshape(sample['pose2D_gt'],(26,-1))       #(26,2) This is from data
            gt_validity = np.reshape(sample['pose2D_validity'],(26,-1))     #(26,2)

            
            # Filtering ########################################################################################################
            if True:
                requiredJoints= [0,1,2, 3,4,5, 6,7,8, 9,10,11]      #In Total26
                if np.min(gt_validity[requiredJoints,0])== False:
                    continue

            min_pt = np.min(gt_skel[gt_validity[:,0]], axis=0)
            max_pt = np.max(gt_skel[gt_validity[:,0]], axis=0)
            # bbox= [ min_pt[0], min_pt[1], max_pt[0], max_pt[1] ]
            bbox= [ min_pt[0], min_pt[1], max_pt[0] - min_pt[0], max_pt[1] - min_pt[1]]


            center = [bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]
            scale = scaleFactor*max(bbox[2], bbox[3])/200


            #Save data
            # imgnames_.append(os.path.join('train',fileName_saved)) 
            imgnames_.append(os.path.join('train',fileName_saved)) 
            openposes_.append(np.zeros([25,3]))       #blank
            centers_.append(center)
            scales_.append(scale)
            has_smpl_.append(1)
            poses_.append(sample['pose3DParam']['pose'])        #(72,)
            shapes_.append(sample['pose3DParam']['shape'])       #(10,)


            #2D keypoints (total26 -> SPIN24)
            poseidx_spin24 = [0,1,2,  3,4,5, 6,7,8,  9,10,11, 19,20,21,22,23] 
            poseidx_total26 =  [0,1,2,  3,4,5,  6,7,8,  9,10,11,  14, 15, 16, 17, 18  ]
            part = np.zeros([24,3])
            part[poseidx_spin24,:2] = gt_skel[poseidx_total26] #(52,)  totalGT26 type
            part[poseidx_spin24,2] = 1*gt_validity[poseidx_total26,0]   
            parts_.append(part)

            #3D joint
            S = np.zeros([24,4])
            S[poseidx_spin24,:3] = sample['pose3D_pred'][poseidx_total26,:]  * 0.001     #Scaling skeleton 3D (currently mm) -> meter
            S[poseidx_spin24,3] = 1
            
            skel3D_.append(S)

            #Debug 2D Visualize
            if False:
                img = cv2.imread( os.path.join( '/run/media/hjoo/disk/data/mpii_human_pose_v1/images',imgnames_[-1]) )
                img = viewer2D.Vis_Skeleton_2D_smplCOCO(gt_skel, pt2d_visibility = gt_validity[:,0], image =img)
                img = viewer2D.Vis_Bbox_minmaxPt(img, min_pt, max_pt)
                viewer2D.ImShow(img, waitTime=0)  

            #Debug 3D Visualize smpl_coco
            if False:
                # data3D_coco_vis = np.reshape(data3D_coco, (data3D_coco.shape[0],-1)).transpose()   #(Dim, F)
                # data3D_coco_vis *=0.1   #mm to cm
                # glViewer.setSkeleton( [ data3D_coco_vis] ,jointType='smplcoco')
                # glViewer.show()

                #Debug 3D Visualize, h36m
                data3D_h36m_vis = np.reshape(data3D_h36m, (data3D_h36m.shape[0],-1)).transpose()   #(Dim, F)
                data3D_h36m_vis *=0.001   #meter to cm

                # data3D_smpl24 = np.reshape(data3D_smpl24, (data3D_smpl24.shape[0],-1)).transpose()   #(Dim, F)
                # data3D_smpl24 *=0.1

                glViewer.setSkeleton( [ data3D_h36m_vis]  ,jointType='smplcoco')
                glViewer.show()


            # keypoints

    # print("Final Img Num: {}, Final Sample Num: {}".format( len(set(imgnames_) , len(imgnames_)) ) )
    print("Final Sample Num: {}".format( len(imgnames_)))
    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    # out_file = os.path.join(out_path, '1031-mpii3D_train_44257_all.npz')
    out_file = os.path.join(out_path, '10-09-posetrack-train_validlimbs.npz')

    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=parts_,
                       openpose=openposes_,
                       pose=poses_,

                        shape=shapes_,
                        has_smpl=has_smpl_,
                        S=skel3D_)



def pennaction_extract(cocoPose3DAll,imgRootDir, out_path):

    # 0head
    # 1left_shoulder
    # 2right_shoulder
    # 3left_elbow
    # 4right_elbow
    # 5left_wrist
    # 6right_wrist
    # 7left_hip
    # 8right_hip
    # 9left_knee
    # 10right_knee
    # 11left_ankle
    # 12right_ankle

    # convert joints to global order
    # joints_idx = [ 18, 9, 8, 10 ,7, 11, 6, 3,2 ,4, 1, 5, 0]       #Left right were flipped...wrong
    joints_idx = [18 ,8 ,9 ,7 ,10 ,6 ,11 , 2 , 3 , 1, 4, 0 ,5]

    # bbox expansion factor
    scaleFactor = 1.2

    # structs we need
    imgnames_, scales_, centers_, parts_, openposes_, annot_ids_ = [], [], [], [], [], []
    subjectIds_ =[]
 
    for annot in cocoPose3DAll:
        # keypoints processing
        keypoints = annot['keypoints']
        keypoints[keypoints[:,2]>0,2] = 1

        #disable head 
        keypoints[0,2] = 0

        #Change the following to select a subset of coco
        # if sum(keypoints[5:,2]>0) < 12:   #Original: cases that all body limbs are annotated
        #     continue
        # if sum(keypoints[5:,2]>0) >= 12:   #If all parts are valid. skip. we already have this
        #     continue
        # if sum(keypoints[5:,2]>0) < 6:   #At least 6 joints should be there
        #     continue
        # image name
        img_name_full = annot['imgname']
        img_name_full = os.path.join( os.path.basename( os.path.dirname(img_name_full)) , os.path.basename(img_name_full) )

        # keypoints
        part = np.zeros([24,3])     
        part[joints_idx] = keypoints

        # scale and center
        bbox_xyxy = annot['bbox_xyxy']        #X,Y,W,H
        bbox_xywh = [ bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2]-bbox_xyxy[0], bbox_xyxy[3] -  bbox_xyxy[1]  ]
        center = [bbox_xywh[0] + bbox_xywh[2]/2, bbox_xywh[1] + bbox_xywh[3]/2]
        scale = scaleFactor*max(bbox_xywh[2], bbox_xywh[3])/200
        openpose = np.zeros([25,3])       #blank
  
        if False:    #visualize
            imgPath = os.path.join(imgRootDir, img_name_full)
            raw_img = cv2.imread(imgPath)
            raw_img = viewer2D.Vis_Skeleton_2D_SPIN24(part[:,:2],pt2d_visibility=part[:,2],  image=raw_img )
            # raw_img = viewer2D.Vis_Skeleton_2D_foot(foot_kp[:,:-1], foot_kp[:,-1] , image= raw_img)
            # raw_img = viewer2D.Vis_Skeleton_2D_Openpose25(openpose[:,:-1], openpose[:,-1] , image= raw_img)
            viewer2D.ImShow(raw_img, waitTime=1)
        
        subjectid = annot['subjectId']
        # subjectid = "{}-id{:03}".format(seqName,trackid)

        # store data
        subjectIds_.append(subjectid)
        imgnames_.append(img_name_full)
        annot_ids_.append(-1)
        centers_.append(center)
        scales_.append(scale)
        parts_.append(part)
        openposes_.append(openpose)

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'pennaction.npz')
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=parts_,
                       subjectIds = subjectIds_,
                       openpose=openposes_,
                       annotIds=annot_ids_)


import glob
import json
from pycocotools.coco import COCO




# Per person order. to show single person with temporal movement
def LoadMocap(dbDir,imgRoot):

    blackList =  ['0001', '0003', '0026', '0034', '1005', '1021', '1024', '1032']
  
    from scipy.io import loadmat

    annotations_all =[]

    # seqPathList = sorted(glob.glob('{0}/*.json'.format(dbDir)) )
    seqPathList = sorted(glob.glob('{0}/*.mat'.format(dbDir)) )

    #Iterate for each person
    for i, gtPath in enumerate(seqPathList):

        imageInfo =[]
        seqName = os.path.basename(gtPath)[:4] 

        if seqName in blackList:
            continue
        # seqName = os.path.basename(jsonData)[:-5]
        print('{}: {}'.format(seqName, i))

        mocap = loadmat(gtPath)
        # N x 13
        vis = mocap['visibility']
        
        x = mocap['x']
        y = mocap['y']
        skel2d_tracked = np.dstack((x, y, vis))    #(N, 13, 3)

        subjectId = 'pennaction_{}-id{:03d}'.format(seqName,0)     #only single person in pennaction

        frameLeng= len(skel2d_tracked)
        for idx in range(frameLeng):
            skel2d = skel2d_tracked[idx]      #13,3

            if idx >=len( mocap['bbox']):
                print("out of range for bbox")
                break
            bbox = mocap['bbox'][idx]


            annot = {}      #Current annotation

            annot['keypoints'] = skel2d #13,3
            annot['subjectId'] = subjectId
            imgPathFull = "{0}/{1}/{2:06d}.jpg".format(imgRoot, seqName, idx+1 )            

            annot['imgname'] = imgPathFull
            annot['bbox_xyxy'] = bbox

            if False:
                inputImg = cv2.imread(imgPathFull)
                inputImg = viewer2D.Vis_Skeleton_2D_pennaction(skel2d[:,:2],skel2d[:,2], image= inputImg)
                inputImg = viewer2D.Vis_Bbox_minmaxPt(inputImg, annot['bbox_xyxy'][:2], annot['bbox_xyxy'][2:])
                viewer2D.ImShow(inputImg,waitTime=0)

            annotations_all.append(annot)
        
        # if(np.sum(j2d_validity_coco19)==0):
        #     print("No valid annotations")
        #     continue
    

    # trackid = annot['track_id']
    # bbox = annot['bbox']
    # image_id = annot['image_id']
    # annot_id = annot['id']
    # keypoints = annot['keypoints']
    #Img path, 2D keypoint, bbox
    return annotations_all

import pickle
if __name__ == '__main__':

    annotRootDir = '/run/media/hjoo/disk/data/Penn_Action/labels'
    imgRootDir = '/run/media/hjoo/disk/data/Penn_Action/frames'     #Starting from 1

    annotData = LoadMocap(annotRootDir, imgRootDir)    #skel2DSeq[ (subId,action, '') ] = (N, 17, 2)

    pennaction_extract(annotData, imgRootDir,'0_ours_dbgeneration')
    # coco_extract('/run/media/hjoo/disk/data/coco', None, 'coco_3d_train')

   

