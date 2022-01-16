import os
import cv2
import numpy as np
import pickle
import json

g_traintest ='test'

MIN_KP = 6

def export_bbox(args, image_folder, tracking_results):

    bbox_out_path_root = os.path.join(args.output_folder,"bbox")
    os.makedirs(bbox_out_path_root, exist_ok=True)
    print(f"Saving to {bbox_out_path_root}")

    #Load all frames

    img_files = sorted(os.listdir(image_folder))

    #Add blank elements (to handle frame without any bbox)
    bboxex_per_image ={}
    for img_name in img_files:
        bboxex_per_image[img_name] =[] 

    #{"image_path": "xxx.jpg", "hand_bbox_list":[{"left_hand":[x,y,w,h], "right_hand":[x,y,w,h]}], "body_bbox_list":[[x,y,w,h]]}
    #Aggregate bbox for the same frame
    for person_id in list(tracking_results.keys()):
        # bbox_dir = os.path.join( bbox_out_path_root, f'{person_id}')
        # os.makedirs(bbox_dir,exist_ok=True)
        for fidx,bbox  in zip(tracking_results[person_id]['frames'],tracking_results[person_id]['bbox']):

            img_name = img_files[fidx]
            bbox_xywh = [ int(0.5 + bbox[0] -  0.5 *bbox[2]) ,int(0.5 + bbox[1]- 0.5*bbox[3]), bbox[2],bbox[3] ]
            bboxex_per_image[img_name].append({'bbox':bbox_xywh ,'id': person_id })

    for img_name in bboxex_per_image:

        image_path =  os.path.join( image_folder,img_name )      #Images: 1 based, 'frames': 0 based
        assert os.path.exists(image_path)

        bbox_list =[]
        id_list =[]
        for elm  in bboxex_per_image[img_name]:
            bbox_list.append(elm['bbox'])
            id_list.append(elm['id'])

        bbox_element = {"image_path":image_path, "body_bbox_list":bbox_list, "id_list":id_list}
        json_out_path = os.path.join(bbox_out_path_root, img_name[:-4] + '.json')
        print(f"Saved to {json_out_path}")
        with open(json_out_path,'w') as f:
            json.dump(bbox_element,f)



def pw3d_extract(dataset_path, out_path):

    # scale factor
    scaleFactor = 1.2

    # structs we use
    imgnames_, scales_, centers_, parts_ = [], [], [], []
    poses_, shapes_, genders_ = [], [], [] 
    subjectIds_ =[]

    # get a list of .pkl files in the directory
    dataset_path = os.path.join(dataset_path, 'sequenceFiles', g_traintest)
    files = [os.path.join(dataset_path, f) 
        for f in os.listdir(dataset_path) if f.endswith('.pkl')]

    reject_cnt = 0
    # go through all the .pkl files
    for filename in files:
        with open(filename, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            smpl_pose = data['poses']
            smpl_betas = data['betas']
            poses2d = data['poses2d']       #(N, 3, 18)
            global_poses = data['cam_poses']
            genders = data['genders']
            valid = np.array(data['campose_valid']).astype(np.bool)
            num_people = len(smpl_pose)
            num_frames = len(smpl_pose[0])
            seq_name = str(data['sequence'])
            img_names = np.array(['imageFiles/' + seq_name + '/image_%s.jpg' % str(i).zfill(5) for i in range(num_frames)])
            # get through all the people in the sequence
            for i in range(num_people):
                p_id = i
                
                valid_pose = smpl_pose[i][valid[i]]
                valid_betas = np.tile(smpl_betas[i][:10].reshape(1,-1), (num_frames, 1))
                valid_betas = valid_betas[valid[i]]
                valid_keypoints_2d = poses2d[i][valid[i]]
                valid_img_names = img_names[valid[i]]
                valid_global_poses = global_poses[valid[i]]
                gender = genders[i]
                # consider only valid frames
                for valid_i in range(valid_pose.shape[0]):
                    part = valid_keypoints_2d[valid_i,:,:].T
                                    
                    target_joint = [10, 9, 8, 11, 12, 13, 4, 3, 2, 5, 6, 7, 0, 0]       #From VIBE
                    valid_joint_cnt = part[target_joint,2]>0.3
                    valid_joint_cnt[12:] =0
                    if sum(valid_joint_cnt)<=6:       #Following VIBE's prop
                        reject_cnt+=1
                        continue

                    part = part[part[:,2]>0,:]
                    bbox = [min(part[:,0]), min(part[:,1]),
                        max(part[:,0]), max(part[:,1])]


                    ##Temporary. Export bbox
                    if True:
                        bbox_out_path_root ='/run/media/hjoo/disk/data/3dpw/bbox'
                        img_name = valid_img_names[valid_i]
                        image_full_path = os.path.join( '/run/media/hjoo/disk/data/3dpw', img_name)
                        bbox_xywh = [ bbox[0], bbox[1], bbox[2]- bbox[0], bbox[3] - bbox[1]]
                        bbox_element = {"image_path":image_full_path, "body_bbox_list":[bbox_xywh], "id_list":[p_id]}
                        json_out_path = os.path.join(bbox_out_path_root, img_name[11:].replace('/','_')[:-4] + f'_pid{p_id}'+  '.json')
                        print(f"Saved to {json_out_path}")
                        with open(json_out_path,'w') as f:
                            json.dump(bbox_element,f)

                    center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
                    scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200
                    
                    # transform global pose
                    pose = valid_pose[valid_i]  #(72,)
                    extrinsics = valid_global_poses[valid_i][:3,:3]
                    pose[:3] = cv2.Rodrigues(np.dot(extrinsics, cv2.Rodrigues(pose[:3])[0]))[0].T[0]     

                    imgnames_.append(valid_img_names[valid_i])
                    centers_.append(center)     #BBox center
                    scales_.append(scale)        #bbox scale (from tight bbox w.r.t 200)
                    poses_.append(pose)         #,72
                    shapes_.append(valid_betas[valid_i])
                    genders_.append(gender)
                    subjectIds_.append(p_id)

                  
                    poseidx_spin24 = [0,1,2,  3,4,5, 6,7,8,  9,10,11, 19,20,21,22,23] 
                    poseidx_openpose18 =  [10,9,8, 11,12,13, 4,3,2, 5,6, 7, 0 ,15 ,14, 17 ,16  ]
                    part = np.zeros([24,3])
                    openpose_pt2d = valid_keypoints_2d[valid_i,:,:].T       #18,3
                    part[poseidx_spin24,:] = openpose_pt2d[poseidx_openpose18,:]
                    part[poseidx_spin24,2] = 1*(part[poseidx_spin24,2]>0.3)
                    parts_.append(part)

                    #2D keypoints (total26 -> SPIN24)
                    if False:
                        from renderer import viewer2D
                        imgName = os.path.join('/run/media/hjoo/disk/data/3dpw', valid_img_names[valid_i])
                        rawImg = cv2.imread(imgName)
                        # viewer2D.ImShow(rawImg)

                        rawImg = viewer2D.Vis_Skeleton_2D_Openpose18(openpose_pt2d[:,:2].ravel(), image= rawImg, pt2d_visibility=openpose_pt2d[:,2]>0.2)
                        # rawImg = viewer2D.Vis_Skeleton_2D_SPIN49(openpose_pt2d[:,:2], image= rawImg, pt2d_visibility=openpose_pt2d[:,2]>0.2)
                        viewer2D.ImShow(rawImg)



    print("reject_cnt: {}, valid_cnt:{}".format(reject_cnt,len(imgnames_) ))
    sampleNum = len(imgnames_)
    # store data
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path,
        f'3dpw_{g_traintest}_{sampleNum}_subjId.npz')
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       pose=poses_,
                       shape=shapes_,
                       gender=genders_,
                       subjectIds= subjectIds_,
                       part=parts_)

if __name__ == '__main__':
    pw3d_extract('/run/media/hjoo/disk/data/3dpw/', '/home/hjoo/codes/fairMocap/0_ours_dbgeneration')