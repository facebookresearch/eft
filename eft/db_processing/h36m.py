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

def h36m_extract(dataset_path, out_path, protocol=1, extract_img=True):

    # convert joints to global order
    h36m_idx = [11, 6, 7, 8, 1, 2, 3, 12, 24, 14, 15, 17, 18, 19, 25, 26, 27]
    global_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]

    # structs we use
    imgnames_, scales_, centers_, parts_, Ss_  = [], [], [], [], []

    # users in validation set
    user_list = [9, 11]

    # go over each user
    for user_i in user_list:
        user_name = 'S%d' % user_i
        # path with GT bounding boxes
        # bbox_path = os.path.join(dataset_path, user_name, 'MySegmentsMat', 'ground_truth_bb')
        # # path with GT 3D pose
        # pose_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures', 'D3_Positions_mono')
        # # path with videos
        vid_path = os.path.join(dataset_path, user_name, 'Videos')

        pose_path = os.path.join(dataset_path, user_name, 'Poses_D3_Positions_mono')
        # path with GT 2D pose
        pose2d_path = os.path.join(dataset_path, user_name, 'Poses_D2_Positions')

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
                if frame_i % 5 == 0 and (protocol == 1 or camera == '60457274'):
                    # image name
                    imgname = '%s_%s.%s_%06d.jpg' % (user_name, action, camera, frame_i+1)
                    
                    # save image
                    
                    if extract_img:
                        img_out = os.path.join(imgs_path, imgname)
                        cv2.imwrite(img_out, image)

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
    out_file = os.path.join(out_path, 
        'h36m_valid_protocol%d.npz' % protocol)
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       S=Ss_)


if __name__ == '__main__':
  
    # MPI-INF-3DHP dataset preprocessing (training set)
    h36m_extract('/run/media/hjoo/disk/data/h36m-fetch/extracted', '/run/media/hjoo/disk/data/h36m-fetch/human36m_10fps_smplcocoTotalcap26_wShape_img/S9_openpose', extract_img=True)
    # h36m_extract('/run/media/hjoo/disk/data/h36m-fetch/human36m_10fps_smplcocoTotalcap26_wShape_img', '/run/media/hjoo/disk/data/mpi_inf_3dhp/openpose', extract_img=False)

   

