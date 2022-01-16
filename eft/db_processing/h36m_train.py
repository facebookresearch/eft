import os
os.environ["CDF_LIB"] = "/home/hjoo/codes_lib/cdf37_1-dist/lib"
from spacepy import pycdf


import os
import sys
import cv2
import glob
import h5py
import numpy as np
import argparse
from spacepy import pycdf

import glob
import pickle
import time

from renderer import glViewer
from renderer import viewer2D

h36mraw_dir = '/run/media/hjoo/disk/data/h36m-fetch/human36m_10fps_smplcocoTotalcap26_wShape_img'
scaleFactor = 1.2


def LoadAllH36mdata_wSMPL_perSeq(out_path):
    # data_dir = '/home/hjoo/data/h36m-fetch/human36m_50fps/'
    # data_dir = '/home/hjoo/data/h36m-fetch/human36m_10fps/'

    list_skel2Ds_h36m =[]
    list_skel3Ds_h36m =[]
    list_smplPose =[]
    list_smplShape =[]
    list_openpose = []
    
    list_imgNames =[]
    list_scale = []
    list_center = []


    # list_joint2d_spin24 = []
    # list_joint3d_spin24 = []


    TRAIN_SUBJECTS = [1,5,6,7,8]
    actionList = ["Directions","Discussion","Eating","Greeting",
            "Phoning","Photo","Posing","Purchases",
            "Sitting","SittingDown","Smoking","Waiting",
            "WalkDog","Walking","WalkTogether"]

    subjectList = TRAIN_SUBJECTS
    
    for subId in subjectList:
        for action in actionList:

            gtPathList = sorted(glob.glob('{}/S{}/{}_*/*/gt_poses_coco_smpl.pkl'.format(h36mraw_dir,subId,action)) )

            print("S{} - {}: {} files".format(subId, action,len(gtPathList)))
        
            for gtPath in gtPathList:
                with open(gtPath,'rb') as f:
                    gt_data = pickle.load(f, encoding='latin1')
                

                #Get Image List
                imgDir = os.path.dirname(gtPath)
                imgList_original = sorted(glob.glob( os.path.join(imgDir, '*.png')))
                folderLeg = len(h36mraw_dir) +1
                imgList = [ n[folderLeg:] for n in imgList_original ]
                data2D_h36m = np.array(gt_data['2d'])       #List -> (N,17,2)  
                data3D_h36m = np.array(gt_data['3d'])       #List -> (N,17,3)
                data3D_smplParams_pose = np.array(gt_data['smplParms']['poses_camCoord'])       #List -> (N,72)
                data3D_smplParams_shape = np.array(gt_data['smplParms']['betas'])       #(10,)

                N = data3D_smplParams_pose.shape[0]
                data3D_smplParams_shape = np.repeat(data3D_smplParams_shape[np.newaxis,:], N,axis =0)   #List -> (N,10)



                #Scaling skeleton 3D (currently mm) -> meter
                data3D_h36m*=0.001
                #optional (centering)
                data3D_h36m = data3D_h36m- data3D_h36m[:,0:1,:]


                scalelist =[]
                centerlist =[]
                bboxlist =[]
                #Generate BBox
                for i in range(len(data2D_h36m)):
                    min_pt = np.min(data2D_h36m[i], axis=0)
                    max_pt = np.max(data2D_h36m[i], axis=0)
                    bbox= [ min_pt[0], min_pt[1], max_pt[0], max_pt[1] ]
                    center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
                    scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200

                    bboxlist.append(bbox)
                    centerlist.append(center)
                    scalelist.append(scale)


                    bDraw =True
                    if bDraw:
                        rawImg = cv2.imread(imgFullPath)

                        # bbox_xyxy = conv_bboxinfo_centerscale_to_bboxXYXY(center, scale)
                        # rawImg = viewer2D.Vis_Bbox_minmaxPt(rawImg,bbox_xyxy[:2], bbox_xyxy[2:])
                        croppedImg, boxScale_o2n, bboxTopLeft = crop_bboxInfo(rawImg, center, scale, (constants.IMG_RES, constants.IMG_RES) )

                        #Visualize image
                        if False:
                            rawImg = viewer2D.Vis_Skeleton_2D_SPIN49(data['keypoint2d'][0][:,:2], pt2d_visibility= data['keypoint2d'][0][:,2], image=rawImg)
                            viewer2D.ImShow(rawImg, name='rawImg')
                            viewer2D.ImShow(croppedImg, name='croppedImg')
                            
                        b =0
                        ############### Visualize Mesh ############### 
                        camParam_scale = pred_camera_vis[b,0]
                        camParam_trans = pred_camera_vis[b,1:]
                        pred_vert_vis = ours_vertices[b].copy()
                        pred_vert_vis = convert_smpl_to_bbox(pred_vert_vis, camParam_scale, camParam_trans)

                        #From cropped space to original
                        pred_vert_vis = convert_bbox_to_oriIm(pred_vert_vis, boxScale_o2n, bboxTopLeft, rawImg.shape[1], rawImg.shape[0]) 

                        #Generate multi-level BBOx
                        bbox_list = multilvel_bbox_crop_gen(rawImg, pred_vert_vis, center, scale, bDebug =False)

                        if False:
                            pred_meshes = {'ver': pred_vert_vis, 'f': smpl.faces}
                            glViewer.setMeshData([pred_meshes], bComputeNormal= True)

                            # ################ Visualize Skeletons ############### 
                            #Vis pred-SMPL joint
                            pred_joints_vis = ours_joints_3d[b,:,:3].copy()     #(N,3)
                            pred_joints_vis = convert_smpl_to_bbox(pred_joints_vis, camParam_scale, camParam_trans)
                            pred_joints_vis = convert_bbox_to_oriIm(pred_joints_vis, boxScale_o2n, bboxTopLeft, rawImg.shape[1], rawImg.shape[0]) 
                    
                            glViewer.setBackgroundTexture(rawImg)
                            glViewer.setWindowSize(rawImg.shape[1], rawImg.shape[0])
                            glViewer.SetOrthoCamera(True)
                            glViewer.show(1)

                            

                assert len(imgList) == len(data2D_h36m)      
                assert len(imgList) == len(data3D_h36m)      
                assert len(imgList) == len(data3D_smplParams_pose)
                assert len(imgList) == len(data3D_smplParams_shape)    
                assert len(imgList) == len(scalelist)       
                assert len(imgList) == len(centerlist)      
                assert len(imgList) == len(bboxlist)       


                list_skel2Ds_h36m.append( data2D_h36m)
                list_skel3Ds_h36m.append( data3D_h36m)
                list_smplPose.append( data3D_smplParams_pose) 
                list_smplShape.append( data3D_smplParams_shape)

                list_imgNames += imgList
                list_scale += scalelist
                list_center += centerlist

                blankopenpose = np.zeros([N, 25,3])
                list_openpose.append( blankopenpose )
                
                #Debug 2D Visualize
                if True:
                    for idx in range(data2D_h36m.shape[0]):
                        img = cv2.imread( imgList_original[idx] )
                        img = viewer2D.Vis_Skeleton_2D_H36m(data2D_h36m[idx],image =img)
                        img = viewer2D.Vis_Bbox_minmaxPt(img, bboxlist[idx][:2], bboxlist[idx][2:])
                        viewer2D.ImShow(img)  

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

            #     break   #debug
            # break       #debug



    list_skel2Ds_h36m = np.vstack(list_skel2Ds_h36m)        #List of (N,17,2) ->  (NM, 17, 2)
    list_skel3Ds_h36m = np.vstack(list_skel3Ds_h36m)        #List of (N,17,3) ->  (NM, 17, 3)
    list_smplPose = np.vstack(list_smplPose)        #List of (N,72) ->  (NM, 72)
    list_smplShape = np.vstack(list_smplShape)        #List of (N,10) ->  (NM, 10/)
    list_openpose = np.vstack(list_openpose)        #List of (N,10) ->  (NM, 10/)

    assert len(list_imgNames) == list_skel2Ds_h36m.shape[0]
    assert len(list_imgNames) == list_skel3Ds_h36m.shape[0]
    assert len(list_imgNames) == list_smplPose.shape[0]
    assert len(list_imgNames) == list_smplShape.shape[0]
    assert len(list_imgNames) == list_openpose.shape[0]
    
    assert len(list_imgNames) == len(list_scale)       
    assert len(list_imgNames) == len(list_center)      


    #Convert H36M -> SPIN24
    # convert joints to global order
    # h36m_idx = [11, 6, 7, 8, 1, 2, 3, 12, 24, 14, 15, 17, 18, 19, 25, 26, 27]
    h36m_idx = [0, 4,5,6, 1,2,3, 7,8,9,10,  11,12,13, 14,15,16]
    global_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]


    sampleNum = len(list_imgNames) 
    joint2d_spin24 = np.zeros( (sampleNum, 24,3) )
    joint2d_spin24[:,global_idx,:2] = list_skel2Ds_h36m[:,h36m_idx,:]
    joint2d_spin24[:,global_idx,2] = 1
    
    
    joint3d_spin24 = np.zeros( (sampleNum, 24,4) )
    joint3d_spin24[:,global_idx,:3] = list_skel3Ds_h36m[:,h36m_idx,:]
    joint3d_spin24[:,global_idx,3] = 1


    list_has_smpl = np.ones( (sampleNum,), dtype =np.uint8)


    

            

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    out_file = os.path.join(out_path, 'h36m_training_fair_meter.npz')
    print("output: {}".format(out_file))
    np.savez(out_file, imgname=list_imgNames,
                        center=list_center,
                        scale=list_scale,
                        part=joint2d_spin24,
                        pose=list_smplPose,
                        shape=list_smplShape,
                        has_smpl=list_has_smpl,
                        S=joint3d_spin24,
                        openpose=list_openpose)


# Illustrative script for training data extraction
# No SMPL parameters will be included in the .npz file.
def h36m_train_extract(dataset_path, out_path, extract_img=False):

    # convert joints to global order
    h36m_idx = [11, 6, 7, 8, 1, 2, 3, 12, 24, 14, 15, 17, 18, 19, 25, 26, 27]
    global_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]

    # structs we use
    imgnames_, scales_, centers_, parts_, Ss_  = [], [], [], [], []


    #additional 3D
    poses_ , shapes_, skel3D_, has_smpl_  = [], [] ,[], []

    # users in validation set
    user_list = [1, 5, 6, 7, 8]

    # go over each user
    for user_i in user_list:
        user_name = 'S%d' % user_i
        
        

        # bbox_path = os.path.join(dataset_path, user_name, 'MySegmentsMat', 'ground_truth_bb')
        # path with GT 3D pose
        pose_path = os.path.join(dataset_path, user_name, 'Poses_D3_Positions_mono')
        # path with GT 2D pose
        pose2d_path = os.path.join(dataset_path, user_name, 'Poses_D2_Positions')
        # path with videos
        # vid_path = os.path.join(dataset_path, user_name, 'Videos')

        # go over all the sequences of each user
        seq_list = glob.glob(os.path.join(pose_path, '*.cdf'))
        seq_list.sort()
        for seq_i in seq_list:

            print('processing: {}'.format(seq_i))
            # sequence info
            seq_name = seq_i.split('/')[-1]
            action, camera, _ = seq_name.split('.')
            action = action.replace(' ', '_')
            # irrelevant sequences
            if action == '_ALL':
                continue

            # 3D pose file
            poses_3d = pycdf.CDF(seq_i)['Pose'][0]          #Nx96

            #load 2D pose file
            seq_i_2D_pose = os.path.join(pose2d_path, os.path.basename(seq_i))
            poses_2d = pycdf.CDF(seq_i_2D_pose)['Pose'][0]      #Nx64
            poses_2d = np.reshape(poses_2d, (-1,32,2))


            # # bbox file
            # bbox_file = os.path.join(bbox_path, seq_name.replace('cdf', 'mat'))
            # bbox_h5py = h5py.File(bbox_file)

            # video file
            if extract_img:
                vid_file = os.path.join(vid_path, seq_name.replace('cdf', 'mp4'))
                imgs_path = os.path.join(dataset_path, 'images')
                vidcap = cv2.VideoCapture(vid_file)
                success, image = vidcap.read()

            # go over each frame of the sequence
            for frame_i in range(poses_3d.shape[0]):
                # read video frame
                if extract_img:
                    success, image = vidcap.read()
                    if not success:
                        break

                # check if you can keep this frame
                if frame_i % 5 == 0:
                    # image name
                    imgname = '%s_%s.%s_%06d.jpg' % (user_name, action, camera, frame_i+1)
                    
                    #Read img
                    
                    # read GT 3D pose
                    Sall = np.reshape(poses_3d[frame_i,:], [-1,3])/1000.            #[32,3]
                    S17 = Sall[h36m_idx]
                    S17 -= S17[0] # root-centered
                    S24 = np.zeros([24,4])
                    S24[global_idx, :3] = S17
                    S24[global_idx, 3] = 1


                    # # read GT bounding box
                    # mask = bbox_h5py[bbox_h5py['Masks'][frame_i,0]].value.T
                    # ys, xs = np.where(mask==1)
                    # bbox = np.array([np.min(xs), np.min(ys), np.max(xs)+1, np.max(ys)+1])

                    curPose_2d = poses_2d[frame_i,:]
                    min_pt = np.min(curPose_2d, axis=0)
                    max_pt = np.max(curPose_2d, axis=0)
                    bbox= [ min_pt[0], min_pt[1], max_pt[0], max_pt[1] ]


                    #Skeleton
                    if False:
                        from renderer import glViewer
                        from renderer import viewer2D
                        image = viewer2D.Vis_Bbox_minmaxPt(image, min_pt, max_pt)
                        viewer2D.ImShow(image,waitTime=1)


                        S17s_ = np.array(S17) *100
                        skelVis = S17s_.ravel()[:,np.newaxis]
                        
                        glViewer.setSkeleton([skelVis])
                        glViewer.show()
                        continue

                    center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
                    scale = 0.9*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200.

                    # store data
                    imgnames_.append(os.path.join('images', imgname))
                    centers_.append(center)
                    scales_.append(scale)
                    Ss_.append(S24)

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'h36m_train.npz')
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=parts_,
                       S=Ss_)


if __name__ == '__main__':
  
    LoadAllH36mdata_wSMPL_perSeq('myh36m_out')
   



