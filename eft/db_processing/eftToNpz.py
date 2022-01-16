"""
Export EFT output (indepdent pkl files) to npz file
"""

import os
from os.path import join
import sys
import json
import numpy as np
# from .read_openpose import read_openpose

import cv2
from renderer import viewer2D
from renderer import glViewer

import torchgeometry as tgm
from fairmocap.utils.geometry import batch_rodrigues
import torch

from tqdm import tqdm
# from read_openpose import read_openpose

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

        # if False:
        #     # read openpose detections
        #     json_file = os.path.join(openpose_path, 'coco',
        #         img_name.replace('.jpg', '_keypoints.json'))
        #     openpose = read_openpose(json_file, part, 'coco')
        
        # store data
        imgnames_.append(img_name_full)
        centers_.append(center)
        scales_.append(scale)
        parts_.append(part)
        openposes_.append(openpose)

        #Fake 3D data
        data['pose']
        data['shape']
        data['has_smpl']
        data['S']

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'coco_3d_noshape_44257_exemplar.npz')
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


def exportOursToSpin(eftDir, out_path):

    # scaleFactor = 1.2

    # structs we need
    imgnames_, scales_, centers_, parts_, openposes_ = [], [], [], [], []

    #additional 3D
    poses_ , shapes_, skel3D_, has_smpl_  = [], [] ,[], []

    pose3DList = os.listdir(eftDir)


    # for imgSample in cocoPose3DAll:
    sampleNum = len(pose3DList)
    # totalSampleNum = [ len(cocoPose3DAll[imgSample]) for imgSample in cocoPose3DAll ]
    # totalSampleNum = sum(totalSampleNum)
    print("\n\n### SampleNum: {} ###".format(sampleNum))
    
    maxDiff =0
    for fname in tqdm(sorted(pose3DList)):

        fname_path = os.path.join(eftDir, fname)

        pose3d = pickle.load(open(fname_path,'rb'))

        #load image
        imgPathFull = pose3d['imageName'][0]
        fileName = os.path.basename(imgPathFull)
        fileName_saved = os.path.join(os.path.basename(os.path.dirname(imgPathFull)), fileName) #start from train2014
        center = pose3d['center'][0]
        scale = pose3d['scale'][0]

        smpl_shape = pose3d['pred_shape'].ravel()
        smpl_pose_mat = torch.from_numpy(pose3d['pred_pose_rotmat'][0])     #24,3,3
        pred_rotmat_hom = torch.cat( [ smpl_pose_mat.view(-1, 3, 3), torch.tensor([0,0,0], dtype=torch.float32,).view(1, 3, 1).expand(24, -1, -1)], dim=-1)
        smpl_pose = tgm.rotation_matrix_to_angle_axis(pred_rotmat_hom).contiguous().view(-1, 72)
        
        #verification
        if True:
            recon_mat = batch_rodrigues(smpl_pose.view(-1,3))       #24,3... axis -> rotmat
            diff = abs(recon_mat.numpy() - pose3d['pred_pose_rotmat'][0])       #2.1234155e-07
            # print(np.max(diff))
            maxDiff = max(maxDiff,np.max(diff))

        smpl_pose = smpl_pose.numpy().ravel()

        openpose2d= pose3d['keypoint2d'][0][:25]        #25,3
        spin2d_skel24= pose3d['keypoint2d'][0][25:]       #24,3


        #Save data
        imgnames_.append(fileName_saved)
        centers_.append(center)
        scales_.append(scale)
        has_smpl_.append(1)
        poses_.append(smpl_pose)        #(72,)
        shapes_.append(smpl_shape)       #(10,)


        openposes_.append(openpose2d)       #blank
        # print(openpose2d)/
        parts_.append(spin2d_skel24)

        #3D joint
        S = np.zeros([24,4])        #blank for 3d. TODO: may need to add valid data for this
        skel3D_.append(S)

        #Debug 2D Visualize
        if False:
            img = cv2.imread( os.path.join( '/run/media/hjoo/disk/data/coco',imgnames_[-1]) )
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
            data3D_h36m_vis *=100   #meter to cm

            # data3D_smpl24 = np.reshape(data3D_smpl24, (data3D_smpl24.shape[0],-1)).transpose()   #(Dim, F)
            # data3D_smpl24 *=0.1

            glViewer.setSkeleton( [ data3D_h36m_vis]  ,jointType='smplcoco')
            glViewer.show()


        # keypoints

    # print("Final Img Num: {}, Final Sample Num: {}".format( len(set(imgnames_) , len(imgnames_)) ) )
    print("Final Sample Num: {}".format( len(imgnames_)))
    print("maxDiff in rot conv.: {}".format(maxDiff))
    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, os.path.basename(eftDir) + '.npz')

    print(f"Save to {out_file}")
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

    # g_pose3D_pklFileName = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_deployed/10-20-coco-comparison/10-17-COCO_train-exemplar-44257/pose3DAnnot.pkl'      #No Shape
    # with open(g_pose3D_pklFileName,'rb') as f:
    #     cocoPose3DAll = pickle.load(f)
    #     f.close()

    # eftDir = '/run/media/hjoo/disk/data/cvpr2020_eft_researchoutput/0_SPIN/0_exemplarOutput/04-20_cocofoot_with8143_annotId'      #with Foot annotation
    eftDir = '/run/media/hjoo/disk/data/cvpr2020_eft_researchoutput/0_SPIN/0_exemplarOutput/04-14_coco_with8143_annotId'      #without foot annotation
    # eftDir = '/run/media/hjoo/disk/data/cvpr2020_eft_researchoutput/0_SPIN/0_exemplarOutput/04-14_cocoall_with8143_annotId'      #without foot annotation, 6kp or more

    exportOursToSpin(eftDir, '0_ours_dbgeneration')
    # coco_extract('/run/media/hjoo/disk/data/coco', None, 'coco_3d_train')

   

