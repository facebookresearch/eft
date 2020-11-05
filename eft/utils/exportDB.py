# Copyright (c) Facebook, Inc. and its affiliates.

"""
Export PKL fotmats to json
"""

import os
import pickle
import json
from tqdm import tqdm

import eft.cores.jointorders as jointorders

def pklToJson(pklDir, outputPath, metainfo):
    """
    Convert pkl to json.

    Args:
        pklDir: (str): write your description
        outputPath: (str): write your description
        metainfo: (todo): write your description
    """

    eft_fileList  = os.listdir(pklDir)       #Check all fitting files
    print(">> Found {} files in the fitting folder {}".format(len(eft_fileList), pklDir))
    totalCnt =0
    erroneousCnt =0

    essentialdata  = []
    for f in tqdm(sorted(eft_fileList)):
        
        #Load EFT data
        fileFullPath = os.path.join(pklDir, f)
        with open(fileFullPath,'rb') as f:
            eft_data = pickle.load(f)

        ########################
        if True:
            #Compute 2D reprojection error
            # if not (data['loss_keypoints_2d']<0.0001 or data['loss_keypoints_2d']>0.001 :
            #     continue
            maxBeta = abs(eft_data['pred_shape']).max()
            if eft_data['loss_keypoints_2d']>0.0005 or maxBeta>3:
                erroneousCnt +=1
                print(">>> Rejected: loss2d: {}, maxBeta: {}".format( eft_data['loss_keypoints_2d'],maxBeta) )
                continue

        """
        Useful data
        pose
        shape
        camera
        bbox

        imageName
        scale
        center
        annotId
        keypoint2d
        keypoint2d_cropped
        smpltype

        _sampleIdx
        """

        data ={}
        data['parm_pose'] = eft_data['pred_pose_rotmat'][0].tolist()      #(10,)
        data['parm_shape'] = eft_data['pred_shape'][0].tolist()      #(24,3,3)
        data['parm_cam'] = eft_data['pred_camera'][0].tolist()       #(3)
        data['bbox_scale'] = eft_data['scale'][0].tolist() 
        data['bbox_center'] = eft_data['center'][0].tolist() 

        # data['pred_keypoint_2d'] = 
        # data['pred_keypoint_validity'] = 
        data['gt_keypoint_2d'] = eft_data['keypoint2d'][0].tolist()          #GT keypoint 2d in SPIN format. In image space. 49,3
        spin24_joint_validity = eft_data['keypoint2d'][0][25:,2]
        data['joint_validity_openpose18'] = spin24_joint_validity[jointorders.JOINT_MAP_SPIN24_TO_OPENPOSE18].tolist()

        if 'smpltype' not in eft_data.keys():
            data['smpltype'] = 'smpl'#eft_data['smpltype']
        else:
            data['smpltype'] = eft_data['smpltype']


        if 'annotId' not in eft_data.keys():
            data['annotId'] = 0
        data['imageName'] = os.path.basename(eft_data['imageName'][0])      #Only save basename

        essentialdata.append(data)

        ##DEBUG TODO
        # if len(essentialdata)==50:
        #     break

    print(">>> Rejection Summary: {}/{}. Valid:{}".format( erroneousCnt, len(eft_fileList)  ,  len(essentialdata)) )


    with open(outputPath,'w') as f:
        json.dump({"ver":0.1, "data":essentialdata, "meta": metainfo},f)
        print(f"saved: {outputPath}")

if __name__ == '__main__':
    
    #Load PKL files
    # pklFileDir = '/run/media/hjoo/disk/data/cvpr2020_eft_researchoutput/0_SPIN/0_exemplarOutput/04-14_cocoall_with8143_annotId'
    # metainfo = {"dbname": "COCO2014-All", "rawname": '04-14_cocoall_with8143_annotId'}
    # outputPath = '04-14_cocoall_with8143_annotId.json'

    # pklFileDir = '/run/media/hjoo/disk/data/cvpr2020_eft_researchoutput/0_SPIN/0_exemplarOutput/04-14_coco_with8143_annotId'
    # metainfo = {"dbname": "COCO2014-Part", "rawname": '04-14_coco_with8143_annotId'}
    # outputPath = 'COCO2014-Part-04-14_coco_with8143_annotId.json'

    # pklFileDir = '/run/media/hjoo/disk/data/cvpr2020_eft_researchoutput/0_SPIN/0_exemplarOutput/11-08_lspet_with8143'
    # metainfo = {"dbname": "LSPet", "rawname": '11-08_lspet_with8143'}
    # outputPath = '11-08_lspet_with8143.json'

    pklFileDir = '/run/media/hjoo/disk/data/cvpr2020_eft_researchoutput/0_SPIN/0_exemplarOutput/11-08_mpii_with8143'
    metainfo = {"dbname": "MPII", "rawname": '11-08_mpii_with8143'}
    outputPath = '11-08_mpii_with8143.json'



    pklToJson(pklFileDir, outputPath, metainfo)



