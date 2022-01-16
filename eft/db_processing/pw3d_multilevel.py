import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm

from fairmocap.models import hmr, SMPL
from fairmocap.core import config 
from fairmocap.utils.imutils import crop, crop_bboxInfo, flip_img, flip_pose, flip_kp, transform, rot_aa, conv_bboxinfo_centerscale_to_bboxXYXY, conv_bboxinfo_bboxXYHW_to_centerscale
from renderer import viewer2D
smpl_male = SMPL('/home/hjoo/codes/fairMocap/smpl/SMPL_MALE.pkl',
                        batch_size=1,
                        create_transl=False)

smpl_female = SMPL('/home/hjoo/codes/fairMocap/smpl/SMPL_FEMALE.pkl',
                        batch_size=1,
                        create_transl=False)
                        
vibe_protocol = False
import pickle
with open('/home/hjoo/codes/fairMocap/extradata/smpl_label/smpl_label_face.pkl','rb') as f:
    g_smpl_facepart = pickle.load(f)


is_export_imgs =True
export_root = '/run/media/hjoo/disk/data/3dpw/imageFiles_crop'

if is_export_imgs:
    for crop_level in [1,2,4]:
        os.makedirs( os.path.join(export_root, f'croplev_{crop_level}'), exist_ok=True)


    
def pw3d_extract(dataset_path, out_path):

    pw3d_multicrop_info={}
    # pw3d_crop_info[ (image_name, pid)]
    # E.g., ('downtown_sitOnStairs_00/image_00000.jpg', 0)
    # return an array with 8 level of bbox (0 face, 7 whole body) 
    # bbox_list[0]
    # {'bbox_xyhw': [995.412413595738, 374.69671840965594, 98.54587305319353, 81.94162583240131], 'center': [1044.6853501223347, 415.6675313258566], 'ratio_bbox_over_face': 1.0, 'scale': 0.5912752383191612}

    # scale factor
    scaleFactor = 1.2

    # structs we use
    imgnames_, scales_, centers_, parts_ = [], [], [], []
    poses_, shapes_, genders_ = [], [], [] 
    multilevel_bboxinfo_ =[]
    # get a list of .pkl files in the directory
    dataset_path = os.path.join(dataset_path, 'sequenceFiles', 'test')
    files = [os.path.join(dataset_path, f) 
        for f in os.listdir(dataset_path) if f.endswith('.pkl')]
    # go through all the .pkl files
    for filename in files:
        with open(filename, 'rb') as f:
            print(f"processing: {filename}")
            data = pickle.load(f, encoding='latin1')
            smpl_pose = data['poses']
            smpl_betas = data['betas']
            poses2d = data['poses2d']       #(N, 3, 18)
            global_poses = data['cam_poses']
            genders = data['genders']
            valid = np.array(data['campose_valid']).astype(np.bool)

            # if False:        #Temporal. To export all 3DPW data
            #     for ii in range(len(valid)):
            #         valid[ii][:] =True

            num_people = len(smpl_pose)
            num_frames = len(smpl_pose[0])
            seq_name = str(data['sequence'])
            img_names = np.array(['imageFiles/' + seq_name + '/image_%s.jpg' % str(i).zfill(5) for i in range(num_frames)])
            smpl_trans = data['trans']

            # get through all the people in the sequence
            for p_id in range(num_people):
                
                valid_pose = smpl_pose[p_id][valid[p_id]]
                valid_betas = np.tile(smpl_betas[p_id][:10].reshape(1,-1), (num_frames, 1))
                valid_betas = valid_betas[valid[p_id]]
                valid_keypoints_2d = poses2d[p_id][valid[p_id]]
                valid_img_names = img_names[valid[p_id]]
                valid_global_poses = global_poses[valid[p_id]]
                valid_smpl_trans = smpl_trans[p_id][valid[p_id]]

                gender = genders[p_id]

                assert(gender=='m')

                # consider only valid frames
                for valid_i in tqdm(range(valid_pose.shape[0])):
                    part = valid_keypoints_2d[valid_i,:,:].T
                    cur_img_name = valid_img_names[valid_i]

                    #Disable lower bodies (openpose COCO18 index)
                    # part[ [9,10,12,13], 2] = 0      #Upper body only  by ignoring 
                    # bHeadOnly = False
                    # if bHeadOnly:
                    #     part[ [4,7, 3,6, 8, 11], 2] = 0         

                    target_joint = [10, 9, 8, 11, 12, 13, 4, 3, 2, 5, 6, 7, 0, 0]       #From VIBE
                    valid_joint_cnt = part[target_joint,2]>0.3
                    valid_joint_cnt[12:] =0

                    if vibe_protocol and sum(valid_joint_cnt)<=6:       #Following VIBE's prop
                        # reject_cnt+=1
                        continue
    
                    part = part[part[:,2]>0,:]
                    bbox = [min(part[:,0]), min(part[:,1]),         #Tight bbox from keypoint, minX, minY, maxX, maxY
                                    max(part[:,0]), max(part[:,1])]
                    # if bHeadOnly:       #To cover head top
                    #     bbox[1] -= abs(bbox[3] - bbox[1])
                    center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
                    scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200
                    
                    # transform global pose
                    pose = valid_pose[valid_i].copy()
                    extrinsics = valid_global_poses[valid_i][:3,:3]
                    pose[:3] = cv2.Rodrigues(np.dot(extrinsics, cv2.Rodrigues(pose[:3])[0]))[0].T[0]     

                    imgnames_.append(valid_img_names[valid_i])
                    centers_.append(center)
                    scales_.append(scale)
                    poses_.append(pose)
                    shapes_.append(valid_betas[valid_i])
                    genders_.append(gender)

                  
                    poseidx_spin24 = [0,1,2,  3,4,5, 6,7,8,  9,10,11, 19,20,21,22,23] 
                    poseidx_openpose18 =  [10,9,8, 11,12,13, 4,3,2, 5,6, 7, 0 ,15 ,14, 17 ,16  ]
                    part = np.zeros([24,3])
                    openpose_pt2d = valid_keypoints_2d[valid_i,:,:].T       #18,3
                    part[poseidx_spin24,:] = openpose_pt2d[poseidx_openpose18,:]
                    part[poseidx_spin24,2] = 1*(part[poseidx_spin24,2]>0.3)
                    parts_.append(part)

                    #2D keypoints (total26 -> SPIN24)
                    if False:
                        
                        imgName = os.path.join('/run/media/hjoo/disk/data/3dpw', valid_img_names[valid_i])
                        rawImg = cv2.imread(imgName)
                        # viewer2D.ImShow(rawImg)
                        # rawImg = viewer2D.Vis_Bbox_minmaxPt(rawImg, bbox[:2], bbox[2:])
                        # rawImg = viewer2D.Vis_Skeleton_2D_Openpose18(openpose_pt2d[:,:2].ravel(), image= rawImg, pt2d_visibility=openpose_pt2d[:,2]>0.2)
                        # rawImg = viewer2D.Vis_Skeleton_2D_Openpose18(openpose_pt2d[:,:2].ravel(), image= rawImg, pt2d_visibility=openpose_pt2d[:,2]>0.2)
                        # rawImg = viewer2D.Vis_Skeleton_2D_SPIN49(openpose_pt2d[:,:2], image= rawImg, pt2d_visibility=openpose_pt2d[:,2]>0.2)
                        # viewer2D.ImShow(rawImg)

                     #Draw Mesh SMPL
                    bDebugVis = False
                    if True:
                        imgName = os.path.join('/run/media/hjoo/disk/data/3dpw', valid_img_names[valid_i])
                        rawImg = cv2.imread(imgName)
                        
                        bbox_list =[]
                        # cam_ext = data['cam_poses'][valid_i]    #4x4
                        cam_int = data['cam_intrinsics']        #3x3

                        import torch
                        import glViewer
                        valid_betas_vis = torch.from_numpy(shapes_[-1][np.newaxis,:]).float()
                        valid_pose_vis = torch.from_numpy(valid_pose[valid_i].copy()[np.newaxis,:]).float()
                        smpl_out = smpl_male(betas=valid_betas_vis, body_pose=valid_pose_vis[:,3:], global_orient=valid_pose_vis[:,:3])

                        ours_vertices = smpl_out.vertices.detach().cpu().numpy()[0]
                        ours_vertices += valid_smpl_trans[valid_i]

                        #Projection
                        ver_3d_camview = np.matmul(valid_global_poses[valid_i,:3,:3], ours_vertices.transpose()).transpose() + valid_global_poses[valid_i,:3,3]
                        ver_2d = np.matmul(cam_int, ver_3d_camview.transpose()).transpose()
                        ver_2d[:,0] =ver_2d[:,0]/ver_2d[:,2]
                        ver_2d[:,1] =ver_2d[:,1]/ver_2d[:,2]
                        ver_2d= ver_2d[:,:2] 

                        #Find face bbox, tight human bbox
                        bbox_xyxy_full = np.array([ min(ver_2d[:,0]), min(ver_2d[:,1]), max(ver_2d[:,0]), max(ver_2d[:,1])])

                        # Get face bbox (min size)
                        headVerIdx = g_smpl_facepart['head']
                        headVert = ver_2d[headVerIdx]
                        minPt = [ min(headVert[:,0]), min(headVert[:,1]) ] 
                        maxPt = [ max(headVert[:,0]), max(headVert[:,1]) ]  
                        bbox_xyxy_small = np.array([minPt[0],minPt[1], maxPt[0], maxPt[1]])

                        # rawImg= viewer2D.Vis_Pt2ds(ver_2d,rawImg)
                        # rawImg = viewer2D.Vis_Bbox_minmaxPt(rawImg, bbox_xyxy_full[:2], bbox_xyxy_full[2:], color=(255,255,0))
                        # rawImg = viewer2D.Vis_Bbox_minmaxPt(rawImg,  bbox_xyxy_small[:2], bbox_xyxy_small[2:] ,color=(255,255,0))

                        #Interpolation
                        minPt_d =  bbox_xyxy_full[:2] - bbox_xyxy_small[:2]
                        maxPt_d =  bbox_xyxy_full[2:] - bbox_xyxy_small[2:]

                        for i in range(8):
                            crop_level = i
                        # if True:
                            # i = crop_level
                            cur_minPt = bbox_xyxy_small[:2] + minPt_d * i/7.0
                            cur_maxPt = bbox_xyxy_small[2:] + maxPt_d * i/7.0
                        
                            bbox_xyhw = [cur_minPt[0],cur_minPt[1], cur_maxPt[0]-cur_minPt[0], cur_maxPt[1]-cur_minPt[1] ]
                            cur_center, cur_scale = conv_bboxinfo_bboxXYHW_to_centerscale(bbox_xyhw)
                            cur_scale *= 1.2        #Scaling factor
                            cur_new_bboxXYXY = conv_bboxinfo_centerscale_to_bboxXYXY(cur_center, cur_scale)

                            if is_export_imgs and crop_level in [7]:#[1,2,4]:
                                #export cropped image into files
                                """Process rgb image and do augmentation."""        
                                cropped_img = crop(rawImg, cur_center, cur_scale, 
                                            [224,224], rot=0)
                                # viewer2D.ImShow(cropped_img,waitTime=0,name="cropped")
                                export_img_name = seq_name + '_' + os.path.basename(imgName)[:-4] + f'_pid{p_id}.jpg'
                                export_img_path = os.path.join( export_root, f'croplev_{crop_level}' ,export_img_name)

                                cv2.imwrite(export_img_path,cropped_img)
                            #Compute face to cur bbox ratio   cur_scale / face_scale
                            if i==0:
                                ratio_bbox_over_face = 1.0
                            else:
                                ratio_bbox_over_face = cur_scale/ bbox_list[0]['scale']

                            bbox_list.append({"scale":cur_scale, "center": cur_center, "ratio_bbox_over_face": ratio_bbox_over_face, "bbox_xyhw": bbox_xyhw})

                            if bDebugVis:  #Draw full size bbox
                                print(f"{i}: {cur_scale}, {center}, {ratio_bbox_over_face}")
                                # tempImg = viewer2D.Vis_Bbox_minmaxPt(rawImg,  cur_minPt, cur_maxPt ,color=(255,255,255))
                                if i in [1,2,4]:
                                    tempImg = viewer2D.Vis_Bbox_minmaxPt(rawImg,  cur_new_bboxXYXY[:2], cur_new_bboxXYXY[2:] ,color=(0,255,255))
                                else:
                                    tempImg = viewer2D.Vis_Bbox_minmaxPt(rawImg,  cur_new_bboxXYXY[:2], cur_new_bboxXYXY[2:] ,color=(255,0,0))
                                viewer2D.ImShow(tempImg,name="bboxGen", waitTime =0)
                        
                        # viewer2D.ImShow(rawImg)
                        multilevel_bboxinfo_.append(bbox_list)
                        key_name = (cur_img_name[11:], p_id)
                        assert key_name not in pw3d_multicrop_info.keys()
                        pw3d_multicrop_info[key_name] = [ dt['bbox_xyhw']  for dt in bbox_list]
                        # if valid_i==5:
                        #     break

    # store data
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    multicrop_out_path = os.path.join( out_path,'pw3d_multicrop_info_sample{}_py2.pkl'.format(len(imgnames_)))
    with open(multicrop_out_path, 'wb') as f:
        pickle.dump(pw3d_multicrop_info,f, protocol=2)
        # pickle.dump(your_object, your_file, protocol=2)

    for level in range(8):
        scales_ = []
        centers_ = []
        for i in range(len(multilevel_bboxinfo_)):
            scales_.append(multilevel_bboxinfo_[i][level]['scale'])
            centers_.append(multilevel_bboxinfo_[i][level]['center'])

        out_file = os.path.join(out_path,
            f'3dpw_test_multilevel_{level}.npz')
        
        np.savez(out_file, imgname=imgnames_,
                        center=centers_,
                        scale=scales_,
                        pose=poses_,
                        shape=shapes_,
                        gender=genders_,
                        part=parts_)

if __name__ == '__main__':

    # for level in range(8):      #range 0-7: 0 headonly, 7 whole body
    pw3d_extract('/run/media/hjoo/disk/data/3dpw/', '/home/hjoo/codes/fairMocap/0_ours_dbgeneration')