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



def posetrack_extract(cocoPose3DAll,imgRootDir, out_path):

    # posetrack_index = range(17)
    # posetrack_index = np.array([0,1,2,  3,4,5,6,7,8,9,10,11,12,13,14,15,16])   #no maching for head (posetrack(2))
    # posetrack_to_smplCOCO18 = np.array([14, 12,19,  16, 17, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0 ])      #no matching for head 

    # convert joints to global order
    # posetrack_index = np.array([0,1,2,  3,4,5,6,7,8,9,10,11,12,13,14,15,16])   #no maching for head (posetrack(2))
    joints_idx = [19, 12, 13, 23, 22,  9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0]

    # bbox expansion factor
    scaleFactor = 1.2

    # structs we need
    imgnames_, scales_, centers_, parts_, openposes_, annot_ids_ = [], [], [], [], [], []
    subjectIds_ =[]
 

    for annot in cocoPose3DAll:
        # keypoints processing
        keypoints = annot['keypoints']
        keypoints = np.reshape(keypoints, (17,3))
        keypoints[keypoints[:,2]>0,2] = 1
        # check if all major body joints are annotated

        #Change the following to select a subset of coco
        if sum(keypoints[5:,2]>0) < 12:   #Original: cases that all body limbs are annotated
            continue
        # if sum(keypoints[5:,2]>0) >= 12:   #If all parts are valid. skip. we already have this
        #     continue
        # if sum(keypoints[5:,2]>0) < 6:   #At least 6 joints should be there
        #     continue
        # image name
        image_id = annot['image_id']
        annot_id = annot['id']
            
        img_name = str(annot['file_name'])
        img_name_full = img_name#join(imgRootDir, img_name)
        # keypoints
        part = np.zeros([24,3])
        part[joints_idx] = keypoints
        # scale and center
        bbox = annot['bbox']        #X,Y,W,H
        center = [bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]
        scale = scaleFactor*max(bbox[2], bbox[3])/200
        openpose = np.zeros([25,3])       #blank
  
        if False:    #visualize
            imgPath = os.path.join(imgRootDir, img_name_full)
            raw_img = cv2.imread(imgPath)
            raw_img = viewer2D.Vis_Skeleton_2D_SPIN24(part[:,:2],pt2d_visibility=part[:,2],  image=raw_img )
            # raw_img = viewer2D.Vis_Skeleton_2D_foot(foot_kp[:,:-1], foot_kp[:,-1] , image= raw_img)
            # raw_img = viewer2D.Vis_Skeleton_2D_Openpose25(openpose[:,:-1], openpose[:,-1] , image= raw_img)
            viewer2D.ImShow(raw_img, waitTime=0)
        
        #Generate a unique human ID:
        seqName = os.path.dirname(img_name_full)
        trackid = annot['track_id']
        subjectid = "{}-id{:03}".format(seqName,trackid)

        # store data
        subjectIds_.append(subjectid)
        imgnames_.append(img_name_full)
        annot_ids_.append(annot_id)
        centers_.append(center)
        scales_.append(scale)
        parts_.append(part)
        openposes_.append(openpose)

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'posetrack.npz')
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



def Conv_posetrack_smplcocoTotal26(inputKp, inputKp_validity):
    
    # posetrack_index = range(17)
    posetrack_index = np.array([0,1,2,  3,4,5,6,7,8,9,10,11,12,13,14,15,16])   #no maching for head (posetrack(2))
    posetrack_to_smplCOCO18 = np.array([14, 12,19,  16, 17, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0 ])      #no matching for head 

    numSample = inputKp.shape[0]
    smplcoco19 = np.zeros( (numSample, 26,2)) #NoheadTop
    smplcoco19[:,posetrack_to_smplCOCO18,:] =  inputKp[:,posetrack_index,:]      #Convert COCO17 tp smplcoco18. neck of smplcoco18 (12) is not assigned.

    smplcoco19_validity = np.full( (numSample,26), False, dtype=bool) #NoheadTop
    smplcoco19_validity[:, posetrack_to_smplCOCO18]  = inputKp_validity[:,posetrack_index]


    #Don't trust neck from the annotation if shoulders are available
    # #interpolate neck (joint12 in smplcoco18)
    for i in range(numSample):
      if smplcoco19_validity[i,9] and smplcoco19_validity[i, 8]:       #only iff shoulders (8, and 9 are valid)
          smplcoco19[i,12,:] = (smplcoco19[i,9,:] + smplcoco19[i,8,:]) *0.5        #interpolate neck
          smplcoco19_validity[i,12] = True #set as valid

    return smplcoco19, smplcoco19_validity



# Per person order. to show single person with temporal movement
def LoadMocap(dbDir,imgRoot):

    # skel2DAll_coco_ori = {}
    # skel2DAll_coco_cropped = {}
    # skel2DAll_coco_validity = {}
    # imageInfoAll = {}   #Image path (seqName/image_%05d.jpg), bbox
    # cropInfoAll = {}
    # bbrInfoAll ={}
    annotations_all =[]

    seqPathList = sorted(glob.glob('{0}/*.json'.format(dbDir)) )

    #Iterate for each person
    for i, jsonData in enumerate(seqPathList):

        imageInfo =[]
        seqName = os.path.basename(jsonData)[:-5]
        print('{}: {}'.format(seqName, i))

        coco = COCO(jsonData)
        img_ids = coco.getImgIds()
        imgs = coco.loadImgs(img_ids)

        posetrack_images = []
        for img in imgs:
            if not img['is_labeled']:  # or img['vid_id'] != '000015':  # Uncomment to filter for a sequence.
                pass
            else:
                posetrack_images.append(img)
        
        print("valid img num: {}".format(len(posetrack_images)))

        #get track_id min, max (to track each person one by one)
        minTrackId = 10000
        maxTrackId = -10000
        for image_idx, selected_im in enumerate(posetrack_images):
            ann_ids = coco.getAnnIds(imgIds=selected_im['id'])
            anns = coco.loadAnns(ann_ids)

            for ann_idx, ann in enumerate(anns):
                minTrackId = min(minTrackId,ann['track_id'])
                maxTrackId = max(maxTrackId,ann['track_id'])
        # print("seq:{}, minId{}, maxId{}".format(seqName,minTrackId, maxTrackId))

        trackIds = range(minTrackId, maxTrackId+1)
        for trackId in trackIds:
            skel2dList =[]
            imageInfo =[]

            #Check all images and find the currently tracked one
            for image_idx, selected_im in enumerate(posetrack_images):
                imgName = selected_im['file_name']
                ann_ids = coco.getAnnIds(imgIds=selected_im['id'])
                anns = coco.loadAnns(ann_ids)

                #Find the currently tracked one
                for ann_idx, ann in enumerate(anns):
                    if(trackId !=ann['track_id']):
                        continue

                    if not ('keypoints' in ann and type(ann['keypoints']) == list):
                        continue

                    if 'bbox' not in ann.keys():
                        continue
                    
                    kp = np.array(ann['keypoints'])   #(51,)
                    kp = np.reshape(kp, (-1,3))     #(17,3)

                    validityCnt = np.sum(kp[:,-1])
                    if validityCnt<1:
                        continue
                    
                    assert kp.shape[0]==17
                    # bbox = np.array(ann['bbox'])
                    ann['file_name'] = selected_im['file_name']
                    annotations_all.append(ann)

                    # # Visualize Image and skeletons
                    if False:
                        filePath = os.path.join(imgRoot, selected_im['file_name'])
                        inputImg = cv2.imread(filePath)
                        img = viewer2D.Vis_Posetrack(kp[:,:2], image= inputImg)
                        viewer2D.ImShow(img,waitTime=10)
                    break       #No need to find anymore on current image
        # break
        # if len(annotations_all)==100:     #Debug
        #     break

    return annotations_all

import pickle
if __name__ == '__main__':

    annotRootDir = '/run/media/hjoo/disk/data/posetrack/annotations/train'
    imgRootDir = '/run/media/hjoo/disk/data/posetrack/'

    annotData = LoadMocap(annotRootDir, imgRootDir)    #skel2DSeq[ (subId,action, '') ] = (N, 17, 2)

    posetrack_extract(annotData, imgRootDir,'0_ours_dbgeneration')
    # coco_extract('/run/media/hjoo/disk/data/coco', None, 'coco_3d_train')

   

