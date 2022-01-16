import numpy as np
import json
import os

def read_openpose(json_file, gt_part=None, dataset=None):

    if gt_part is None:
        gt_part = np.zeros([24,3])
    if dataset is None:
        dataset ='coco'
        
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
        scale = max(max(gt_part[:,0])-min(gt_part[:,0]),max(gt_part[:,1])-min(gt_part[:,1]))
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



def read_openpose_wHand(json_file, gt_part=None, dataset=None):

    if gt_part is None:
        gt_part = np.zeros([24,3])
    if dataset is None:
        dataset ='coco'
    OP_output ={}

    # get only the arms/legs joints
    op_to_12 = [11, 10, 9, 12, 13, 14, 4, 3, 2, 5, 6, 7]
    # read the openpose detection
    if os.path.exists(json_file)==False:
        return None, None
    json_data = json.load(open(json_file, 'r'))
    people = json_data['people']
    if len(people) == 0:
        return None, None
        pass
        # no openpose detection
        # keyp25 = np.zeros([25,3])
    else:
        # size of person in pixels
        scale = max(max(gt_part[:,0])-min(gt_part[:,0]),max(gt_part[:,1])-min(gt_part[:,1]))
        # go through all people and find a match
        dist_conf = np.inf*np.ones(len(people))
        for i, person in enumerate(people):
            # openpose keypoints
            op_keyp25 = np.reshape(person['pose_keypoints_2d'], [25,3])
            op_keyp12 = op_keyp25[op_to_12, :2]
            op_conf12 = op_keyp25[op_to_12, 2:3] > 0.2
            # all the relevant joints should be detected
            # if min(op_conf12) > 0:
                # weighted distance of keypoints
            dist_conf[i] = 12- sum(op_conf12)# np.mean(np.sqrt(np.sum(op_conf12*(op_keyp12 - gt_part[:12, :2])**2, axis=1)))
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
        if False:#min(dist_conf)/scale > 0.1 and min(dist_conf) < thresh:
            # keyp25 = np.zeros([25,3])
            pass
        else:
            
            for k in people[p_sel]:
                if len(people[p_sel][k])==0:
                    continue
                if k=="person_id":
                    OP_output[k] = people[p_sel][k]
                else:
                    OP_output[k] = np.reshape(people[p_sel][k], [-1,3])


    
    return OP_output,  people[p_sel]
