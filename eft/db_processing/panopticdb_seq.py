import os
from os.path import join
import sys
import json
import numpy as np
# from .read_openpose import read_openpose

import cv2
from renderer import viewer2D
from renderer import glViewer

from tqdm import tqdm

# from read_openpose import read_openpose

def applyExtrinsic(joints, calib):
    x = np.dot(calib['R'], joints.T) + calib['t']       #3,19

    x = np.swapaxes(x, 0,1)

    return x

def project2D(joints, calib, imgwh=None, applyDistort=True):
    """
    Input:
    joints: N * 3 numpy array.
    calib: a dict containing 'R', 'K', 't', 'distCoef' (numpy array)
    Output:
    pt: 2 * N numpy array
    inside_img: (N, ) numpy array (bool)
    """
    x = np.dot(calib['R'], joints.T) + calib['t']
    xp = x[:2, :] / x[2, :]

    if applyDistort:
        X2 = xp[0, :] * xp[0, :]
        Y2 = xp[1, :] * xp[1, :]
        XY = X2 * Y2
        R2 = X2 + Y2
        R4 = R2 * R2
        R6 = R4 * R2

        dc = calib['distCoef']
        radial = 1.0 + dc[0] * R2 + dc[1] * R4 + dc[4] * R6
        tan_x = 2.0 * dc[2] * XY + dc[3] * (R2 + 2.0 * X2)
        tan_y = 2.0 * dc[3] * XY + dc[2] * (R2 + 2.0 * Y2)

        # xp = [radial;radial].*xp(1:2,:) + [tangential_x; tangential_y]
        xp[0, :] = radial * xp[0, :] + tan_x
        xp[1, :] = radial * xp[1, :] + tan_y

    # pt = bsxfun(@plus, cam.K(1:2,1:2)*xp, cam.K(1:2,3))';
    pt = np.dot(calib['K'][:2, :2], xp) + calib['K'][:2, 2].reshape((2, 1))

    if imgwh is not None:
        assert len(imgwh) == 2
        imw, imh = imgwh
        winside_img = np.logical_and(pt[0, :] > -0.5, pt[0, :] < imw-0.5) 
        hinside_img = np.logical_and(pt[1, :] > -0.5, pt[1, :] < imh-0.5) 
        inside_img = np.logical_and(winside_img, hinside_img) 
        inside_img = np.logical_and(inside_img, R2 < 1.0) 
        return pt, inside_img

    pt = np.swapaxes(pt,0,1) # 2,19 -> 19,2
    return pt


def exportOursToSpin(out_path):

    scaleFactor = 1.2

    hagglingDBdir ='/home/hjoo/data/pytorch_motionSynth/motionsynth_data/data/processed_panoptic'

    haggling_files = os.listdir(hagglingDBdir+'/panopticDB_pkl_hagglingProcessed')

    # structs we need
    imgnames_, scales_, centers_, parts_, openposes_ = [], [], [], [], []

    #additional 3D
    poses_ , shapes_, skel3D_, has_smpl_  = [], [] ,[], []

    rhand_2d_list, rhand_3d_list, lhand_2d_list, lhand_3d_list  = [], [] ,[], []
    subject_id_list =[]
    
    for mocapName in  haggling_files:
        seqname = mocapName[:18]#'170221_haggling_b1'

        if "170221_haggling_b" not in seqname and "170228_haggling_b" not in seqname:       #testing set
            continue

        print(f"processing{mocapName}")
        
        groupid = mocapName[24:-4]# 3

        print(f"{mocapName}, {seqname}, {groupid}")

        # groupname = f"group{groupid}"
        # mocapName =f'{seqname}_{groupname}.pkl'
        imgDir = f'/home/hjoo/data/panoptic-toolbox/{seqname}/hdImgs/00_00'
        calibdata = f'/home/hjoo/data/panoptic-toolbox/{seqname}/calibration_{seqname}.json'
        cam =json.load( open(calibdata, "r" ))['cameras']
        cam = cam[479:]     #479 is 00_00

        # with open('/run/media/hjoo/disk/data/panoptic_mtc/a4_release/annotation.pkl', 'rb') as f:
        #     data = pickle.load(f)

        # with open('/run/media/hjoo/disk/data/panoptic_mtc/a4_release/camera_data.pkl', 'rb') as f:
        #     cam = pickle.load(f)

        bodydatapath = f'{hagglingDBdir}/panopticDB_pkl_hagglingProcessed/' +mocapName
        bodydata = pickle.load( open( bodydatapath, "rb" ) , encoding='latin1')

        handdatapath = f'{hagglingDBdir}/panopticDB_hand_pkl_hagglingProcessed/' +mocapName
        handdata = pickle.load( open( handdatapath, "rb" ) , encoding='latin1')
        
        # if True:
        #     for subj in bodydata['subjects']:
        #         startFrame = subj['startFrame']

        #         for n in range(subj['joints19'].shape[1]):
        #             imgpath = os.path.join(imgDir,"00_00_%08d.jpg" % (startFrame+n))
        #             img = cv2.imread(imgpath)
        #             viewer2D.ImShow(img)

        groupStartFrame = bodydata['startFrame']
        subNum = len(bodydata['subjects'])
        for subid in range(subNum):#[1]:

            subj_body = bodydata['subjects'][subid]['joints19']

            if len(handdata['hand_right']) <=subid:
                continue

            if len(handdata['hand_left']) <=subid:
                continue

            subj_righthand = handdata['hand_right'][subid]['hand21']
            subj_lefthand = handdata['hand_left'][subid]['hand21']

            #Validity Score Right
            subj_righthand_validity = handdata['hand_right'][subid]['bValidFrame']
            # subj_righthand_score = handdata['hand_right'][subid]['scores']
            localStart = groupStartFrame - handdata['hand_right'][subid]['startFrame']          #This 
            subj_righthand_validity = subj_righthand_validity[localStart:]
            # subj_righthand_score = subj_righthand_score[: , localStart:]

            #Validity Score Left
            subj_lefthand_validity = handdata['hand_left'][subid]['bValidFrame']
            # subj_lefthand_score = handdata['hand_left'][subid]['scores']
            localStart = groupStartFrame - handdata['hand_left'][subid]['startFrame']          #This 
            subj_lefthand_validity = subj_lefthand_validity[localStart:]
            # subj_lefthand_score = subj_lefthand_score[:, localStart:]

            print("valid: {}/{}".format(  sum(handdata['hand_right'][subid]['bValidFrame']), len(handdata['hand_right'][subid]['bValidFrame']) ))
            print("valid: {}/{}".format(  sum(handdata['hand_left'][subid]['bValidFrame']), len(handdata['hand_left'][subid]['bValidFrame']) ))

            startFrame = bodydata['startFrame']

            # assert subj_body.shape[1] == subj_righthand.shape[1]

            frameLeng = subj_body.shape[1]

            for ii in range(frameLeng):
                
                bVis = False
                frameid = startFrame + ii
                subject_id ="{}_g{}_s{}_{:08d}".format(seqname,   groupid, subid, frameid)      #seqname, groupid. subid, frameid
                # print(subject_id)

                imgFullPath = os.path.join(imgDir,"00_00_%08d.jpg" % (frameid))
                body_landmark= np.array( subj_body[:,ii]).reshape(-1,3)      #19, 3          #SMC19 order
                # body_landmark = np.array(sample['body']['landmarks']).reshape(-1, 3)            #19, 3          #SMC19 order

                if subj_righthand.shape[1] <=ii:
                    rhand_landmark = np.zeros( (21,3))            #21, 3          #SMC19 order
                else:
                    rhand_landmark = np.array(subj_righthand[:,ii]).reshape(-1, 3)            #21, 3          #SMC19 order

                if subj_lefthand.shape[1] <=ii:
                    lhand_landmark = np.zeros( (21,3))            #21, 3          #SMC19 order
                else:
                    lhand_landmark = np.array(subj_lefthand[:,ii]).reshape(-1, 3)            #21, 3          #SMC19 order

                calib_data = cam[0]        #00
                for c in calib_data:
                    calib_data[c] = np.array(calib_data[c])

                skeleton_3d_camview = applyExtrinsic(body_landmark, calib_data)     #19,3
                rhand_3d_camview = applyExtrinsic(rhand_landmark, calib_data)     #21,3
                lhand_3d_camview = applyExtrinsic(lhand_landmark, calib_data)     #21,3

                skeleton_2d = project2D(body_landmark, calib_data)     #19,2
                rhand_2d = project2D(rhand_landmark, calib_data)     #19,2
                lhand_2d = project2D(lhand_landmark, calib_data)     #19,2

                imgName = "{}/hdImgs/00_00/00_00_{:08d}.jpg".format(seqname, frameid) 

                # print(imgName)
                # imgFullPath = os.path.join(imgDir, imgName)
                # if os.path.exists(imgFullPath) == False:
                #     continue
                # print(imgName)

                #Visulaize 3D 
                if False:
                    img = cv2.imread(imgFullPath)
                    img = viewer2D.Vis_Skeleton_2D_SMC19(skeleton_2d, image=img)
                    viewer2D.ImShow(img, waitTime=1)
                    
                    skeleton_3d_camview = skeleton_3d_camview.ravel()[:,np.newaxis]
                    rhand_3d_camview = rhand_3d_camview.ravel()[:,np.newaxis]
                    lhand_3d_camview = lhand_3d_camview.ravel()[:,np.newaxis]
                    glViewer.setSkeleton([skeleton_3d_camview, rhand_3d_camview, lhand_3d_camview])
                    # glViewer.setSkeleton([skeleton_3d_camview])

                    glViewer.setBackgroundTexture(img)
                    glViewer.SetOrthoCamera(True)
                    glViewer.show(0)
                    continue

                min_pt = np.min(skeleton_2d, axis=0)
                min_pt[0] = max(min_pt[0],0)
                min_pt[1] = max(min_pt[1],0)
                
                max_pt = np.max(skeleton_2d, axis=0)
                max_pt[0] = min(max_pt[0],1920)
                max_pt[1] = min(max_pt[1],1080)
                # bbox= [ min_pt[0], min_pt[1], max_pt[0], max_pt[1] ]
                bbox= [ min_pt[0], min_pt[1], max_pt[0] - min_pt[0], max_pt[1] - min_pt[1]]

                center = [bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]
                scale = scaleFactor*max(bbox[2], bbox[3])/200

                #Save data
                # imgnames_.append(os.path.join('train',fileName_saved)) 
                # imgnames_.append(os.path.join('train',fileName_saved)) 
                imgnames_.append(imgName) 
                openposes_.append(np.zeros([25,3]))       #blank
                centers_.append(center)
                scales_.append(scale)
                # has_smpl_.append(1)
                # poses_.append(sample['pose3DParam']['pose'])        #(72,)
                # shapes_.append(sample['pose3DParam']['shape'])       #(10,)

                #2D keypoints (total26 -> SPIN24)
                poseidx_spin24 = [0,1,2, 3,4,5, 6,7,8, 9,10,11, 14, 12, 19, 20, 22, 21, 23] 
                poseidx_smc19 =  [14,13,12, 6,7,8, 11,10,9, 3,4,5, 2, 0, 1, 15,16, 17,18]
                part = np.zeros([24,3])
                part[poseidx_spin24,:2] = skeleton_2d[poseidx_smc19] #(52,)  totalGT26 type
                part[poseidx_spin24,2] = 1
                parts_.append(part)

                #3D joint
                S = np.zeros([24,4])
                S[poseidx_spin24,:3] = skeleton_3d_camview[poseidx_smc19,:]  * 0.01     #Scaling skeleton 3D (currently cm) -> meter
                S[poseidx_spin24,3] = 1
                
                skel3D_.append(S)

                rhand_2d_list.append(rhand_2d)

                if len(subj_righthand_validity) <=ii:
                    rhand_validity = 0
                else:    
                    rhand_validity = subj_righthand_validity[ii]
                # rhand_score = subj_righthand_score[:,ii].mean()
                rhand_3d_camview = rhand_3d_camview* 0.01      
                if rhand_validity==1:
                    rhand_3d_camview = np.concatenate( [rhand_3d_camview, np.ones((21,1))], axis=1)
                else:
                    rhand_3d_camview = np.concatenate( [rhand_3d_camview, np.zeros((21,1))], axis=1)
                    bVis = True
                rhand_3d_list.append(rhand_3d_camview)

                lhand_2d_list.append(lhand_2d)

                if len(subj_lefthand_validity) <=ii:
                    lhand_validity = 0
                else:    
                    lhand_validity = subj_lefthand_validity[ii]

                # lhand_validity = subj_lefthand_validity[ii]
                lhand_3d_camview= lhand_3d_camview* 0.01
                if lhand_validity==1:
                    lhand_3d_camview = np.concatenate( [lhand_3d_camview,  np.ones((21,1))], axis=1)
                else:
                    lhand_3d_camview = np.concatenate( [lhand_3d_camview,  np.zeros((21,1))], axis=1)
                    bVis = True
                lhand_3d_list.append(lhand_3d_camview)

                subject_id_list.append(subject_id)

                #Add hand joints
                # print("right: {} : left: {}".format(rhand_score, lhand_score))
                # print("right: {} : left: {}".format(rhand_validity, lhand_validity))

                #Debug 2D Visualize
                if False:#bVis:
                    img = cv2.imread(imgFullPath)
                    # img = cv2.imread( os.path.join( '/run/media/hjoo/disk/data/mpii_human_pose_v1/images',imgnames_[-1]) )
                    img = viewer2D.Vis_Skeleton_2D_SMC19(skeleton_2d, image =img)
                    img = viewer2D.Vis_Skeleton_2D_Hand(rhand_2d, image =img)
                    img = viewer2D.Vis_Skeleton_2D_Hand(lhand_2d, image =img)

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

    # print("Final Img Num: {}, Final Sample Num: {}".format( len(set(imgnames_) , len(imgnames_)) ) )
    print("Final Sample Num: {}".format( len(imgnames_)))
    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    # out_file = os.path.join(out_path, '1031-mpii3D_train_44257_all.npz')
    out_file = os.path.join(out_path, f'panoptic_{mocapName[:-4]}')

    print(f"Save to {out_file}")

    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=parts_,
                       openpose=openposes_,
                        S=skel3D_,

                        rhand_3d=rhand_3d_list,
                        rhand_2d=rhand_2d_list,
                        lhand_3d=lhand_3d_list,
                        lhand_2d=lhand_2d_list,


                        subjectid=subject_id_list       #To handle sequence data and track the same person in output
                        )


import pickle
if __name__ == '__main__':

    exportOursToSpin('0_ours_dbgeneration')

   

