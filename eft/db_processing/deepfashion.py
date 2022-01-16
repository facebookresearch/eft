import os
from os.path import join
import sys
import json
import numpy as np
# from .read_openpose import read_openpose


def read_openpose(json_file):
    # get only the arms/legs joints
    op_to_12 = [11, 10, 9, 12, 13, 14, 4, 3, 2, 5, 6, 7]
    # read the openpose detection
    json_data = json.load(open(json_file, 'r'))
    people = json_data['people']
    if len(people) == 0:
        # no openpose detection
        keyp25 = np.zeros([25,3])
    else:
        # size of person in pixels
        # scale = max(max(gt_part[:,0])-min(gt_part[:,0]),max(gt_part[:,1])-min(gt_part[:,1]))
        # go through all people and find a match
        dist_conf = np.inf*np.ones(len(people))
        for i, person in enumerate(people):
            # openpose keypoints
            op_keyp25 = np.reshape(person['pose_keypoints_2d'], [25,3])
            op_keyp12 = op_keyp25[op_to_12, :2]
            op_conf12 = op_keyp25[op_to_12, 2:3] > 0
            # all the relevant joints should be detected
            if min(op_conf12) > 0:
                # weighted distance of keypoints
                dist_conf[i] = np.mean(np.sqrt(np.sum(op_conf12*(op_keyp12 - gt_part[:12, :2])**2, axis=1)))
        # closest match
        p_sel = np.argmin(dist_conf)
        # the exact threshold is not super important but these are the values we used
        if dataset == 'mpii':
            thresh = 30
        elif dataset == 'coco':
            thresh = 10
        else:
            thresh = 0
        # dataset-specific thresholding based on pixel size of person
        if min(dist_conf)/scale > 0.1 and min(dist_conf) < thresh:
            keyp25 = np.zeros([25,3])
        else:
            keyp25 = np.reshape(people[p_sel]['pose_keypoints_2d'], [25,3])
    return keyp25




def dfs_extract(dataset_path, openpose_path, out_path):

    # convert joints to global order
    joints_idx = [19, 20, 21, 22, 23, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0]

    # bbox expansion factor
    scaleFactor = 1.2

    # structs we need
    imgnames_, scales_, centers_, parts_, openposes_ = [], [], [], [], []


    imgs = {}
    # for img in json_data['images']:
    #     imgs[img['id']] = img

    for idx in range(191961):

        image_id = '{:06d}.jpg'.format(idx)

        # read openpose detections
        json_file = os.path.join(openpose_path, image_id.replace('.jpg', '_keypoints.json'))
        openpose = read_openpose(json_file)


        # keypoints processing
        keypoints = annot['keypoints']
        keypoints = np.reshape(keypoints, (17,3))
        keypoints[keypoints[:,2]>0,2] = 1
        # check if all major body joints are annotated
        
        
        # if sum(keypoints[5:,2]>0) < 12:   #Original
        #     continue

        if sum(keypoints[5:,2]>0) >= 12:   #If all parts are valid. skip. we already have this
            continue

        if sum(keypoints[5:,2]>0) < 6:   #At least 6 joints should be there
            continue
            
        # image name
        image_id = annot['image_id']
        img_name = str(imgs[image_id]['file_name'])
        img_name_full = join('train2014', img_name)
        # keypoints
        part = np.zeros([24,3])
        part[joints_idx] = keypoints
        # scale and center
        bbox = annot['bbox']
        center = [bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]
        scale = scaleFactor*max(bbox[2], bbox[3])/200

    
        # read openpose detections
        # json_file = os.path.join(openpose_path, 'coco',
        #     img_name.replace('.jpg', '_keypoints.json'))
        # openpose = read_openpose(json_file, part, 'coco')
        openpose = np.zeros([25,3])       #blank
        
        # store data
        imgnames_.append(img_name_full)
        centers_.append(center)
        scales_.append(scale)
        parts_.append(part)
        openposes_.append(openpose)

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'coco_2014_train_missingLimbs.npz')
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=parts_,
                       openpose=openposes_)


if __name__ == '__main__':
  
    dfs_extract('/run/media/hjoo/disk/data/DeepFashion2/data/train/image','/run/media/hjoo/disk/data/DeepFashion2/data/train/openpose', 'deepfashion2')

   

