import os
from os.path import join
import sys
import json
import numpy as np
# from .read_openpose import read_openpose

from preprocessdb.read_openpose import read_openpose

def coco_extract(dataset_path, openpose_path, out_path):

    # convert joints to global order
    # joints_idx = [19, 20, 21, 22, 23, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0]
    joints_idx = [8,7,6,9,10,11,2,1,0,3,4,5, 13, 12, 23, 22,19,  21, 20]

    # bbox expansion factor
    scaleFactor = 1.2

    # structs we need
    imgnames_, scales_, centers_, parts_, openposes_, annot_ids_ = [], [], [], [], [], []

    # json annotation file
    json_path = os.path.join(dataset_path, 'ochuman.json')
    json_data = json.load(open(json_path, 'r'))

    #Cnt number
    n =0
    for tempData in json_data['images']:
        n+=len(tempData['annotations'])
    print(f"total cnt:{n}")

    json_path = os.path.join(dataset_path, 'ochuman_coco_format_test_range_0.00_1.00.json')
    json_data = json.load(open(json_path, 'r'))

    imgs = {}
    # for img in json_data['images']:
    #     imgs[img['id']] = img

    for tempData in json_data['images']:

        imgname = tempData['file_name']
        print(imgname)
        continue
        annotList = tempData['annotations']
        imgidx = tempData['image_id']

        for annot in annotList:
            if annot['keypoints'] is None:
                continue
            # keypoints processing
            keypoints = annot['keypoints']
            keypoints = np.reshape(keypoints, (19,3))
            keypoints[keypoints[:,2]>0,2] = 1
          # check if all major body joints are annotated


            #Change the following to select a subset of coco
            # if sum(keypoints[5:,2]>0) < 12:   #Original: cases that all body limbs are annotated
            #     continue
            # if sum(keypoints[5:,2]>0) >= 12:   #If all parts are valid. skip. we already have this
            #     continue
            if sum(keypoints[5:,2]>0) < 6:   #At least 6 joints should be there
                continue
                    
            # image name
            image_id = imgidx
            # annot_id = annot['id']
            img_name = imgname
            # img_name_full = join(dataset_path,'images', imgname)
            img_name_full = join('images', imgname)
            # keypoints
            part = np.zeros([24,3])
            part[joints_idx] = keypoints
            # scale and center
            bbox = annot['bbox']
            center = [bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]
            scale = scaleFactor*max(bbox[2], bbox[3])/200

            if False: #vis
                import cv2
                import viewer2D
                image = cv2.imread(img_name_full)
                image = viewer2D.Vis_Skeleton_2D_general(part[:,:2],image = image)
                viewer2D.ImShow(image)
                
            # read openpose detections
            # json_file = os.path.join(openpose_path, 'coco',
            #     img_name.replace('.jpg', '_keypoints.json'))
            # openpose = read_openpose(json_file, part, 'coco')
            openpose = np.zeros([25,3])       #blank
            
            # store data
            imgnames_.append(img_name_full)
            # annot_ids_.append(annot_id)
            centers_.append(center)
            scales_.append(scale)
            parts_.append(part)
            openposes_.append(openpose)

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'ochuman.npz')
    np.savez(out_file, imgname=imgnames_,
                    center=centers_,
                    scale=scales_,
                    part=parts_,
                    openpose=openposes_)


if __name__ == '__main__':
  
    coco_extract('/run/media/hjoo/disk/data/OCHuman', None, 'ochuman')

   

