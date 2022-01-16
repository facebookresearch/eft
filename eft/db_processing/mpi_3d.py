import os
from os.path import join
import sys
import json
import numpy as np
import h5py

from renderer import viewer2D
import cv2

# from .read_openpose import read_openpose

from read_openpose import read_openpose

def coco_extract(dataset_path, openpose_path, out_path):

    # convert joints to global order
    joints_idx = [19, 20, 21, 22, 23, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0]

    # bbox expansion factor
    scaleFactor = 1.2

    # structs we need
    imgnames_, scales_, centers_, parts_, openposes_ = [], [], [], [], []

    # json annotation file
    json_path = os.path.join(dataset_path, 
                             'annotations', 
                             'person_keypoints_train2014.json')
    json_data = json.load(open(json_path, 'r'))

    imgs = {}
    for img in json_data['images']:
        imgs[img['id']] = img

    for annot in json_data['annotations']:
        # keypoints processing
        keypoints = annot['keypoints']
        keypoints = np.reshape(keypoints, (17,3))
        keypoints[keypoints[:,2]>0,2] = 1
        # check if all major body joints are annotated
        if sum(keypoints[5:,2]>0) < 12:
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

        if False:
            # read openpose detections
            json_file = os.path.join(openpose_path, 'coco',
                img_name.replace('.jpg', '_keypoints.json'))
            openpose = read_openpose(json_file, part, 'coco')
        
        # store data
        imgnames_.append(img_name_full)
        centers_.append(center)
        scales_.append(scale)
        parts_.append(part)
        openposes_.append(openpose)



    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'coco_2014_train.npz')
    # np.savez(out_file, imgname=imgnames_,
    #                    center=centers_,
    #                    scale=scales_,
    #                    part=parts_,
    #                    openpose=openposes_)


    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=parts_,
                       openpose=openposes_,
                       pose=fits_3d['pose'],
                        shape=fits_3d['shape'],
                        has_smpl=fits_3d['has_smpl'],
                        S=Ss_)


def exportOursToSpin(cocoPose3DAll, out_path):

    scaleFactor = 1.2


    # annotation files
    annot_file = os.path.join('/home/hjoo/codes/SPIN/data', 'train.h5')
    # read annotations
    f = h5py.File(annot_file, 'r')
    centers, imgnames, parts, scales = \
        f['center'], f['imgname'], f['part'], f['scale']


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
        fileName_saved = os.path.join('images',imgSample[0]['imgId'])
        # fileName = os.path.basename(imgPathFull)
        # fileName_saved = os.path.join(os.path.basename(os.path.dirname(imgPathFull)), fileName) #start from train2014


        for sample in imgSample:

            validJointNum = np.sum(sample['pose2D_validity'][::2])

            if validJointNum<4:
                continue

            if np.isnan(sample['pose3DParam']['camScale']):     
                continue

            #visualize
            # if True:        #Save BBox
            #     inputImg_2dvis = inputImg.copy()
            #     inputImg_2dvis = viewer2D.Vis_Bbox(inputImg_2dvis,sample['bbr'])
            #     # viewer2D.ImShow(inputImg_2dvis)
            #     imgFilePath = '{0}/{1}.jpg'.format(bboxFolder, fileName)
            #     cv2.imwrite(imgFilePath, inputImg_2dvis)
            
            if 'bbr' in sample.keys():
                bbox = sample['bbr']
            else:

                keypoints = np.reshape(sample['pose2D_gt'], (-1,2))     #26,2
                valid = sample['pose2D_validity'][::2]      #(26,)
                valid_keypoints = keypoints[valid]
                
                min_pt = np.min(valid_keypoints, axis=0)
                max_pt = np.max(valid_keypoints, axis=0)
                bbox= [ min_pt[0], min_pt[1], max_pt[0] - min_pt[0], max_pt[1] - min_pt[1]]

            #Debug Visualization
            if False:
                imgPathFull = os.path.join( '/run/media/hjoo/disk/data/mpii_human_pose_v1', fileName_saved)
                inputImg_2dvis = cv2.imread(imgPathFull)
                inputImg_2dvis = viewer2D.Vis_Skeleton_2D_smplCOCO(sample['pose2D_gt'], pt2d_visibility= sample['pose2D_validity'] ,image=inputImg_2dvis)
                inputImg_2dvis = viewer2D.Vis_Bbox(inputImg_2dvis,bbox)
                viewer2D.ImShow(inputImg_2dvis,waitTime=0)
                # continue

            center = [bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]
            scale = scaleFactor*max(bbox[2], bbox[3])/200


            #Save data
            imgnames_.append(fileName_saved)
            openposes_.append(np.zeros([25,3]))       #blank
            centers_.append(center)
            scales_.append(scale)
            has_smpl_.append(1)
            poses_.append(sample['pose3DParam']['pose'])        #(72,)
            shapes_.append(sample['pose3DParam']['shape'])       #(10,)


            #2D keypoints (total26 -> SPIN24)
            poseidx_spin24 = [0,1,2,  3,4,5, 6,7,8,  9,10,11,  19,20,21,22,23,    13]   #13 head top
            poseidx_total26 =  [0,1,2,  3,4,5,  6,7,8,  9,10,11,  14, 15, 16, 17, 18,    19  ]       #19 head top
            part = np.zeros([24,3])
            gt_skel = np.reshape(sample['pose2D_gt'],(26,-1))
            gt_validity = np.reshape(sample['pose2D_validity'],(26,-1))
            part[poseidx_spin24,:2] = gt_skel[poseidx_total26] #(52,)  totalGT26 type
            part[poseidx_spin24,2] = 1*gt_validity[poseidx_total26,0]   
            parts_.append(part)

            #3D joint
            S = np.zeros([24,4])
            S[poseidx_spin24,:3] = sample['pose3D_pred'][poseidx_total26,:]
            S[poseidx_spin24,3] = 1
            

            skel3D_.append(S)


            # keypoints


    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'mpi_train_wShape_1537_ep200_exemplar.npz')

    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=parts_,
                       openpose=openposes_,
                       pose=poses_,

                        shape=shapes_,
                        has_smpl=has_smpl_,
                        S=skel3D_)


      
import pickle
if __name__ == '__main__':

    # #source
    # g_pose3D_pklFileName = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_deployed/10-26-wShape/10-26-COCO_train_wShape_1537_ep200_exemplar/pose3DAnnot.pkl'
    # with open(g_pose3D_pklFileName,'rb') as f:
    #     cocoPose3DAll_source = pickle.load(f)
    #     f.close()

    g_pose3D_pklFileName = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_deployed/10-26-wShape/10-26-MPII_train_wShape_1537_ep200_exemplar/pose3DAnnot.pkl'      #No Shape
    with open(g_pose3D_pklFileName,'rb') as f:
        cocoPose3DAll = pickle.load(f)
        f.close()

    # #Bbox copy
    # assert len(cocoPose3DAll_source) ==len(cocoPose3DAll)
    # for key_imgId, imgSample in sorted(cocoPose3DAll_source.items()):

    #     imgSample2 = cocoPose3DAll[key_imgId]

    #     for (sample1, sample2) in zip(imgSample, imgSample2):
    #         assert (sample1['pose2D_gt'] ==sample2['pose2D_gt']).all

    #         sample2['bbr'] = sample1['bbr']

 
    exportOursToSpin(cocoPose3DAll, '0_ours_dbgeneration')
    # coco_extract('/run/media/hjoo/disk/data/coco', None, 'coco_3d_train')

   

