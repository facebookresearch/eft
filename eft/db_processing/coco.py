import os
from os.path import join
import sys
import json
import numpy as np

#For debugging
from renderer import viewer2D#, glViewer
import cv2
# from .read_openpose import read_openpose
# from read_openpose import read_openpose
from eft.db_processing.read_openpose import read_openpose


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

def coco_extract(dataset_path, openpose_path, out_path, train_val ='train', bWithCOCOFoot = False):

    if bWithCOCOFoot:
        footData = loadFoot()       #footData[annotId]

    # convert joints to global order
    joints_idx = [19, 20, 21, 22, 23, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0]

    # bbox expansion factor
    scaleFactor = 1.2

    # structs we need
    imgnames_, scales_, centers_, parts_, openposes_, annot_ids_ = [], [], [], [], [], []

    # json annotation file
    if train_val=='train':
        json_path = os.path.join(dataset_path, 
                                'annotations', 
                                'person_keypoints_train2014.json')
    else:
        json_path = os.path.join(dataset_path, 
                                'annotations', 
                                'person_keypoints_val2014.json')


    print("Processing COCO Dataset")
    # print(f"image dir: {cocoImgDir}")
    print(f"annot dir: {json_path}")


    json_data = json.load(open(json_path, 'r'))

    imgs = {}
    for img in json_data['images']:
        imgs[img['id']] = img

    cnt =0
    for annot in json_data['annotations']:
        # keypoints processing
        keypoints = annot['keypoints']
        keypoints = np.reshape(keypoints, (17,3))
        keypoints[keypoints[:,2]>0,2] = 1
        # check if all major body joints are annotated

        #Change the following to select a subset of coco
        if sum(keypoints[5:,2]>0) < 12:   #Original: cases that all body limbs are annotated
            continue

        # image name
        image_id = annot['image_id']
        annot_id = annot['id']

            
        img_name = str(imgs[image_id]['file_name'])
        if train_val=='train':
            img_name_full = join('train2014', img_name)
        else:
            img_name_full = join('val2014', img_name)
        # keypoints
        part = np.zeros([24,3])
        part[joints_idx] = keypoints
        # scale and center
        bbox = annot['bbox']        #X,Y,W,H
        center = [bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]
        scale = scaleFactor*max(bbox[2], bbox[3])/200

        # read openpose detections
        # json_file = os.path.join(openpose_path, 'coco',
        #     img_name.replace('.jpg', '_keypoints.json'))
        # openpose = read_openpose(json_file, part, 'coco')
        openpose = np.zeros([25,3])       #blank

        #If foot gt is availabe
        if bWithCOCOFoot and annot_id in footData.keys():
            foot_kp = footData[annot_id].astype(np.float32)     # # 0: Left big toe.# 1: Left small toe. # 2: Left heel. # 3: Right big toe. # 4: Right small toe. # 5: Right heel.
            openpose[19:] = foot_kp
            #Openpose: 19, "LBigToe", {20, "LSmallToe"}, {21, "LHeel"},{22, "RBigToe"}, {23, "RSmallToe"}, {24, "RHeel"},


            if False:    #visualize
                cocoImgDir = '/run/media/hjoo/disk/data/coco'
                imgPath = os.path.join(cocoImgDir, img_name_full)
                raw_img = cv2.imread(imgPath)
                # raw_img = viewer2D.Vis_Skeleton_2D_general(keypoints[:,:-1], image=raw_img )
                # raw_img = viewer2D.Vis_Skeleton_2D_foot(foot_kp[:,:-1], foot_kp[:,-1] , image= raw_img)
                raw_img = viewer2D.Vis_Skeleton_2D_Openpose25(openpose[:,:-1], openpose[:,-1] , image= raw_img)
                viewer2D.ImShow(raw_img, waitTime=0)

        # store data
        imgnames_.append(img_name_full)
        annot_ids_.append(annot_id)
        centers_.append(center)
        scales_.append(scale)
        parts_.append(part)
        openposes_.append(openpose)

    print(cnt)

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    if train_val=='train':
        out_file = os.path.join(out_path, 'coco_2014_train_12kp.npz')
    else:
        out_file = os.path.join(out_path, 'coco_2014_val_12kp.npz')
    
    print(f"Saving pre-processed db output: {out_file}")

    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=parts_,
                       openpose=openposes_,
                       annotIds=annot_ids_)


if __name__ == '__main__':
    coco_extract(dataset_path= './data_sets/coco/', openpose_path= None, out_path= './preprocessed_db/')         
    # coco_extract('/run/media/hjoo/disk/data/coco', None, 'coco_3d_val',train_val ='val') 

