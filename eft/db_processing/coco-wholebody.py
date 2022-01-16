import os
from os.path import join
import sys
import json
import numpy as np


#For debugging
from renderer import viewer2D, glViewer
import cv2

# from .read_openpose import read_openpose

# from read_openpose import read_openpose

from preprocessdb.read_openpose import read_openpose


bWithCOCOFoot = True



def loadFoot():
    cocoImgDir = '/run/media/hjoo/disk/data/coco2017/train2017'
    footAnnotFile = '/run/media/hjoo/disk/data/cmu_foot/person_keypoints_train2017_foot_v1.json'
    with open(footAnnotFile,'rb' ) as f:
        json_data = json.load(f)

    imgs = {}
    for img in json_data['images']:
        imgs[img['id']] = img

    footAnnot={}
    for annot in json_data['annotations']:

        keypoints =annot['keypoints']

        # keypoints = np.reshape(keypoints, (17,3))     #original coco
        keypoints = np.reshape(keypoints, (23,3))     #original coco
        keypoints[keypoints[:,2]>0,2] = 1

        footCnt = sum(keypoints[-6:,-1])        #Last 6 keypoints are for foot
        if footCnt <6:
            continue

        image_id = annot['image_id']
        annot_id = annot['id']
        imageName = imgs[image_id]['file_name']

        footAnnot[annot_id] = keypoints[-6:,]
        continue

        if True:
            imgPath = os.path.join(cocoImgDir, imageName)
            raw_img = cv2.imread(imgPath)
            raw_img = viewer2D.Vis_Skeleton_2D_foot(keypoints[-6:,:-1],keypoints[-6:,-1] , image= raw_img)
            viewer2D.ImShow(raw_img, waitTime=0)

        # 0: Left big toe.
        # 1: Left small toe.
        # 2: Left heel.
        # 3: Right big toe.
        # 4: Right small toe.
        # 5: Right heel.
        continue

    return footAnnot

def coco_extract(dataset_path, openpose_path, out_path):

    # if bWithCOCOFoot:
    #     footData = loadFoot()       #footData[annotId]

    # convert joints to global order
    joints_idx = [19, 20, 21, 22, 23, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0]

    # bbox expansion factor
    scaleFactor = 1.2

    # structs we need
    imgnames_, scales_, centers_, parts_, openposes_, annot_ids_ = [], [], [], [], [], []

    kp_leftHand_, kp_rightHand_, kp_face_ , kp_foot_  =  [], [] ,[] ,[]     #additional data from coco-wholebody

    # json annotation file
    json_path = os.path.join(dataset_path, 
                             'annotations', 
                             'coco_wholebody_train_v1.0.json')
    json_data = json.load(open(json_path, 'r'))

    imgs = {}
    for img in json_data['images']:
        imgs[img['id']] = img

    for annot in json_data['annotations']:

        """
        bboxes: face_box, lefthand_box, righthand_box and
        whole-body keypoints: face_kpts, lefthand_kpts, righthand_kpts, foot_kpts and
        validity: face_valid, lefthand_valid, righthand_valid, foot_valid.
        """
        # keypoints processing
        keypoints = annot['keypoints']
        keypoints = np.reshape(keypoints, (17,3))
        keypoints[keypoints[:,2]>0,2] = 1
        # check if all major body joints are annotated

        #Change the following to select a subset of coco
        # if sum(keypoints[5:,2]>0) < 6:   #Original: cases that all body limbs are annotated
        if sum(keypoints[5:,2]>0) < 12:   #Original: cases that all body limbs are annotated
            continue
            
        # image name
        image_id = annot['image_id']
        annot_id = annot['id']
            
        img_name = str(imgs[image_id]['file_name'])
        img_name_full = join('train2017', img_name)
        # keypoints
        part = np.zeros([24,3])
        part[joints_idx] = keypoints
        # scale and center
        bbox = annot['bbox']
        center = [bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]
        scale = scaleFactor*max(bbox[2], bbox[3])/200

        openpose = np.zeros([25,3])       #blank

        #New Annotation from https://github.com/jin-s13/COCO-WholeBody
        kp_leftHand = np.reshape(annot['lefthand_kpts'],(-1,3))            #21,3
        kp_rightHand = np.reshape(annot['righthand_kpts'],(-1,3))       #21,3
        kp_face = np.reshape(annot['face_kpts'],(-1,3))                #68,3
        kp_foot = np.reshape(annot['foot_kpts'],(-1,3))                #6,3

        bbox_leftHand = annot['lefthand_box']
        bbox_rightHand = annot['righthand_box']
        bbox_leftHand = annot['face_box']        

        valid_lefthand = annot['lefthand_valid']
        valid_righthand = annot['righthand_valid']
        valid_face = annot['face_valid']
        valid_foot = annot['foot_valid']

        if False:
            cocoImgDir = '/run/media/hjoo/disk/data/coco2017'
            imgPath = os.path.join(cocoImgDir, img_name_full)
            raw_img = cv2.imread(imgPath)
            # raw_img = viewer2D.Vis_Skeleton_2D_Openpose25(openpose[:,:-1], openpose[:,-1] , image= raw_img)

            #Visualize body
            raw_img = viewer2D.Vis_Skeleton_2D_SPIN24(part[:,:-1], part[:,-1] , image= raw_img)

            #Visualize face
            raw_img = viewer2D.Vis_Skeleton_2D_Openpose_face(kp_face[:,:-1],kp_face[:,-1], image= raw_img)

            #Visualize hand
            raw_img = viewer2D.Vis_Skeleton_2D_Openpose_hand(kp_leftHand[:,:-1],kp_leftHand[:,-1], image= raw_img)
            raw_img = viewer2D.Vis_Skeleton_2D_Openpose_hand(kp_rightHand[:,:-1],kp_rightHand[:,-1], image= raw_img)

            #Visualize feet
            raw_img = viewer2D.Vis_Skeleton_2D_foot(kp_foot[:,:-1],kp_foot[:,-1], image= raw_img)


            viewer2D.ImShow(raw_img, waitTime=0)

        # store data
        imgnames_.append(img_name_full)
        annot_ids_.append(annot_id)
        centers_.append(center)
        scales_.append(scale)
        parts_.append(part)
        openposes_.append(openpose)

        kp_leftHand_.append(kp_leftHand)
        kp_rightHand_.append(kp_rightHand)
        kp_face_.append(kp_face)
        kp_foot_.append(kp_foot)

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'coco2017_wholebody_train_v10_12kp.npz')
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=parts_,
                       openpose=openposes_,
                       annotIds=annot_ids_,
                       
                        kp_leftHand = kp_leftHand_,
                        kp_rightHand = kp_rightHand_,
                        kp_face =  kp_face_,
                        kp_foot = kp_foot_
                       )


if __name__ == '__main__':
  
    coco_extract('/run/media/hjoo/disk/data/coco2017', None, 'data_preprocessing')

   

