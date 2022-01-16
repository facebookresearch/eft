# Copyright (c) Facebook, Inc. and its affiliates.

#Modified from https://github.com/nkolot/SPIN/blob/master/LICENSE

"""
This file contains functions that are used to perform data augmentation.
"""
import torch
import numpy as np
import scipy.misc
import cv2

from eft.cores import constants
from torchvision.transforms import Normalize

def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale     #h becomes the original bbox max(height, min). 
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h         #This becomes a scaling factor to rescale original bbox -> res size (default: 224x224)
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t

def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0]-1, pt[1]-1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int)+1

def crop(img, center, scale, res, rot=0):
    """Crop image according to the supplied bounding box."""
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1))-1
    # Bottom right point
    br = np.array(transform([res[0]+1,
                             res[1]+1], center, scale, res, invert=1))-1

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if new_shape[0]>15000 or new_shape[1]>15000:
        print("Image Size Too Big!  scale{}, new_shape{} br{}, ul{}".format(scale, new_shape, br, ul))
        return  None


    if len(img.shape) > 2:
        new_shape += [img.shape[2]]


    new_img = np.zeros(new_shape, dtype=np.uint8)

    # #Compute bbox for Han's format
    # bboxScale_o2n = 224/new_img.shape[0]
    # bboxTopLeft = ul *bboxScale_o2n


    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    # print("{} vs {}  || {} vs {}".format(new_y[1] - new_y[0] , old_y[1] - old_y[0], new_x[1] - new_x[0], old_x[1] -old_x[0] )   )
    if new_y[1] - new_y[0] != old_y[1] - old_y[0] or new_x[1] - new_x[0] != old_x[1] -old_x[0] or  new_y[1] - new_y[0] <0 or  new_x[1] - new_x[0] <0:
        print("Warning: maybe person is out of image boundary!")
        return None
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1],
                                                        old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    new_img = cv2.resize(new_img, tuple(res))
    # new_img = scipy.misc.imresize(new_img, res)     #Need this to get the same number with the old model (trained with this resize)

    return new_img#, bboxScale_o2n, bboxTopLeft

#renamed from crop_bboxInfo
# def crop_bboxInfo(img, center, scale, res, rot=0):
#Return with bbox info
# def crop_bboxInfo(img, center, scale, res =(224,224), rot=0):
def crop_bboxInfo(img, center, scale, res =(224,224)):
    """Crop image according to the supplied bounding box."""
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1))-1
    # Bottom right point
    br = np.array(transform([res[0]+1,
                             res[1]+1], center, scale, res, invert=1))-1


    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    # new_img = np.zeros(new_shape)
    if new_shape[0] <1  or new_shape[1] <1:
        return None, None, None
    new_img = np.zeros(new_shape, dtype=np.uint8)

    if new_img.shape[0] ==0:
        return None, None, None

    #Compute bbox for Han's format
    bboxScale_o2n = res[0]/new_img.shape[0]             #224/ 531

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])

    if new_y[0] <0 or new_y[1]<0 or new_x[0] <0 or new_x[1]<0 :
        return None, None, None

    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1],
                                                        old_x[0]:old_x[1]]

    # bboxTopLeft_inOriginal = (old_x[0], old_y[0] )
    bboxTopLeft_inOriginal = (ul[0], ul[1] )

    if new_img.shape[0] <20 or new_img.shape[1]<20:
        return None, None, None
    # print(bboxTopLeft_inOriginal)
    # from renderer import viewer2D
    # viewer2D.ImShow(new_img.astype(np.uint8),name='cropped')

    new_img = cv2.resize(new_img, res)

    # viewer2D.ImShow(new_img.astype(np.uint8),name='original')


    return new_img, bboxScale_o2n, np.array(bboxTopLeft_inOriginal)


# def crop_frombboxXYXY(image, bbox_xyxy):
#     """
#     Naive cropping
#     """

#     bbox = bbox_xyxy.astype(np.int32)
#     croppedImg = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
#     return croppedImg



# def crop_bboxInfo_frombboxXYXY(img, bboxXYXY, res=224):

#     """Crop image according to the supplied bounding box."""
#     if bbox_file is None and openpose_file is None:
#         # Assume that the person is centerered in the image
#         height = img.shape[0]
#         width = img.shape[1]
#         center = np.array([width // 2, height // 2])
#         scale = max(height, width) / 200
#     else:
#         if bbox_file is not None:
#             center, scale = bbox_from_json(bbox_file)
#         elif openpose_file is not None:
#             center, scale = bbox_from_openpose(openpose_file)

#             if center is None:
#                 return None, None, None



#     # Upper left point
#     ul = np.array(transform([1, 1], center, scale, res, invert=1))-1
#     # Bottom right point
#     br = np.array(transform([res[0]+1,
#                              res[1]+1], center, scale, res, invert=1))-1


#     # Padding so that when rotated proper amount of context is included
#     pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)

#     new_shape = [br[1] - ul[1], br[0] - ul[0]]
#     if len(img.shape) > 2:
#         new_shape += [img.shape[2]]
#     # new_img = np.zeros(new_shape)
#     if new_shape[0] <1  or new_shape[1] <1:
#         return None, None, None
#     new_img = np.zeros(new_shape, dtype=np.uint8)

#     if new_img.shape[0] ==0:
#         return None, None, None

#     #Compute bbox for Han's format
#     bboxScale_o2n = res[0]/new_img.shape[0]             #224/ 531

#     # Range to fill new array
#     new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
#     new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
#     # Range to sample from original image
#     old_x = max(0, ul[0]), min(len(img[0]), br[0])
#     old_y = max(0, ul[1]), min(len(img), br[1])
#     new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1],
#                                                         old_x[0]:old_x[1]]

#     # bboxTopLeft_inOriginal = (old_x[0], old_y[0] )
#     bboxTopLeft_inOriginal = (ul[0], ul[1] )


#     if new_img.shape[0] <20 or new_img.shape[1]<20:
#         return None, None, None
#     # print(bboxTopLeft_inOriginal)
#     # from renderer import viewer2D
#     # viewer2D.ImShow(new_img.astype(np.uint8),name='cropped')

#     new_img = cv2.resize(new_img, res)


#     return new_img, np.array(bboxScale_o2n), np.array(bboxTopLeft_inOriginal)




def conv_bboxinfo_center2topleft(scale, center):
    """
    from (scale, center) -> (o2n, topleft)
    """

    hmr_res = (224,224)
    
    """Crop image according to the supplied bounding box."""
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, hmr_res, invert=1))-1
    # Bottom right point
    br = np.array(transform([hmr_res[0]+1,
                             hmr_res[1]+1], center, scale, hmr_res, invert=1))-1

    new_shape = [br[1] - ul[1], br[0] - ul[0]]

    #Compute bbox for Han's format
    bboxScale_o2n = hmr_res[0]/new_shape[0]             #224/ 531
    # bboxTopLeft_inOriginal = (old_x[0], old_y[0] )
    bboxTopLeft_inOriginal = (ul[0], ul[1] )

    return np.array(bboxScale_o2n), np.array(bboxTopLeft_inOriginal)


#Aliasing. Deprecated
def conv_bboxinfo_bboxXYXY(scale, center):
    """
    from (scale, center) -> (topleft, bottom right)
    """
    return conv_bboxinfo_centerscale_to_bboxXYXY(center,scale)

def conv_bboxinfo_centerscale_to_bboxXYXY(center, scale):
    """
    from (scale, center) -> (topleft, bottom right)
    """

    hmr_res = (224,224)
    
    """Crop image according to the supplied bounding box."""
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, hmr_res, invert=1))-1
    # Bottom right point
    br = np.array(transform([hmr_res[0]+1,
                             hmr_res[1]+1], center, scale, hmr_res, invert=1))-1

    return np.concatenate( (ul, br))


def conv_bboxinfo_bboxXYHW_to_centerscale(bbox_xyhw, bLooseBox = False):
    """
    from (bbox_xyhw) -> (center, scale)
    Args:
        bbox_xyhw: [minX,minY,W,H]
        bLooseBox: if true, draw less tight box with sufficient margin (SPIN's default setting)
    Output:
        center: bbox center
        scale: to make max size to 1.0
    """

    center = [bbox_xyhw[0] + bbox_xyhw[2]/2, bbox_xyhw[1] + bbox_xyhw[3]/2]

    if bLooseBox:
        scaleFactor =1.2
        scale = scaleFactor*max(bbox_xyhw[2], bbox_xyhw[3])/200       #This is the one used in SPIN's pre-processing. See preprocessdb/coco.py
    else:
        scale = max(bbox_xyhw[2], bbox_xyhw[3])/224   

    return center, scale
    # bbr = [bbox_xyxy[0],bbox_xyxy[1], bbox_xyxy[2]-bbox_xyxy[0]  , bbox_xyxy[3]-bbox_xyxy[2]]
    # center = bbr[:2] + 0.5 * bbr[2:]
    # bbox_size = max(bbr[2:])
    # hand_boxScale_o2n = 224.0/bbox_size         #original to 224
    # return hand_boxScale_o2n
    
    # return center, scale 
    

def uncrop(img, center, scale, orig_shape, rot=0, is_rgb=True):
    """'Undo' the image cropping/resizing.
    This function is used when evaluating mask/part segmentation.
    """
    res = img.shape[:2]
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1))-1
    # Bottom right point
    br = np.array(transform([res[0]+1,res[1]+1], center, scale, res, invert=1))-1
    # size of cropped image
    crop_shape = [br[1] - ul[1], br[0] - ul[0]]

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(orig_shape, dtype=np.uint8)
    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], orig_shape[1]) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], orig_shape[0]) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(orig_shape[1], br[0])
    old_y = max(0, ul[1]), min(orig_shape[0], br[1])
    img = cv2.resize(img, crop_shape, interp='nearest')
    new_img[old_y[0]:old_y[1], old_x[0]:old_x[1]] = img[new_y[0]:new_y[1], new_x[0]:new_x[1]]
    return new_img

def rot_aa(aa, rot):
    """Rotate axis angle parameters."""
    # pose parameters
    R = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                  [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                  [0, 0, 1]])
    # find the rotation of the body in camera frame
    per_rdg, _ = cv2.Rodrigues(aa)
    # apply the global rotation to the global orientation
    resrot, _ = cv2.Rodrigues(np.dot(R,per_rdg))
    aa = (resrot.T)[0]
    return aa

def flip_img(img):
    """Flip rgb images or masks.
    channels come last, e.g. (256,256,3).
    """
    img = np.fliplr(img)
    return img

def flip_kp(kp):
    """Flip keypoints."""
    if len(kp) == 24:
        flipped_parts = constants.J24_FLIP_PERM
    elif len(kp) == 49:
        flipped_parts = constants.J49_FLIP_PERM
    kp = kp[flipped_parts]
    kp[:,0] = - kp[:,0]
    return kp

def flip_pose(pose):
    """Flip pose.
    The flipping is based on SMPL parameters.
    """
    flipped_parts = constants.SMPL_POSE_FLIP_PERM
    pose = pose[flipped_parts]
    # we also negate the second and the third dimension of the axis-angle
    pose[1::3] = -pose[1::3]
    pose[2::3] = -pose[2::3]
    return pose


## For converting coordinate between SMPL 3D coord <-> 2D bbox <-> original 2D image 
#data3D: (N,3), where N is number of 3D points in "smpl"'s 3D coordinate (vertex or skeleton)

# Note: camScal is based on normalized 2D keypoint (-1~1). 112 = 0.5 =224 (to come back to the original resolution)
# camScale and camTrans is for normalized coord.
# (camScale*(vert) + camTras )  ==> normalized coordinate  (-1 ~ 1)
# 112* ((camScale*(vert) + camTras )  + 1) == 112*camScale*vert +  112*camTrans + 112
def convert_smpl_to_bbox(data3D, scale, trans, bAppTransFirst=False):
    hmrIntputSize_half = 224 *0.5

    if bAppTransFirst:      #Hand model
        data3D[:,0:2] += trans
        data3D *= scale           #apply scaling
    else:
        data3D *= scale           #apply scaling
        data3D[:,0:2] += trans
    
    data3D*= hmrIntputSize_half         #112 is originated from hrm's input size (224,24)

    return data3D

def convert_bbox_to_oriIm(data3D, boxScale_o2n, bboxTopLeft, imgSizeW, imgSizeH):
    hmrIntputSize_half = 224 *0.5

    # if type(imgSize) is tuple:
    #     imgSize = np.array(imgSize)
    imgSize = np.array([imgSizeW,imgSizeH])

    # pred_vert_vis = convert_bbox_to_oriIm(pred_vert_vis, boxScale_o2n, bboxTopLeft, rawImg.shape)
    data3D/=boxScale_o2n
    data3D[:,:2] += bboxTopLeft - imgSize*0.5 + hmrIntputSize_half/boxScale_o2n
    # data3D[:,1] += bboxTopLeft[1] - rawImg.shape[0]*0.5 + 112/boxScale_o2n
    return data3D



def convert_smpl_to_bbox_perspective(data3D, scale_ori, trans_ori, focalLeng, scaleFactor=1.0):
    hmrIntputSize_half = 224 *0.5

    scale = scale_ori* hmrIntputSize_half
    trans = trans_ori *hmrIntputSize_half

    if False:   #Weak perspective
        data3D *= scale           #apply scaling
        data3D[:,0:2] += trans
    # data3D*= hmrIntputSize_half         #112 is originated from hrm's input size (224,24)
    else:
        # delta = (trans - imageShape*0.5)/scale            #Current projection already consider camera center during the rendering. Thus no need to consider principle axis
        delta = (trans )/scale
        data3D[:,0:2] +=delta

        newZ = focalLeng/scale
        deltaZ =  newZ - np.mean(data3D[:,2])
        data3D[:,2] +=deltaZ
        # data3D[:,2] +=16.471718554146534        #debug


    if False:   #Scaling to be a certain dist from camera
        texture_plan_depth = 500
        ratio = texture_plan_depth /np.mean(data3D[:,2])
        data3D *=ratio  
    else:
        # height  = np.max(data3D[:,1]) - np.min(data3D[:,1])
        # # print(f"height: {height}")
        # # targetHeight = 380
        # ratio = targetHeight /height
        # data3D *=ratio  

        data3D *=scaleFactor

    return data3D


def convert_bbox_to_oriIm_perspective(data3D, boxScale_o2n, bboxTopLeft, imgSizeW, imgSizeH, focalLeng):
    hmrIntputSize_half = 224 *0.5

    # if type(imgSize) is tuple:
    #     imgSize = np.array(imgSize)
    imgSize = np.array([imgSizeW,imgSizeH])

    # pred_vert_vis = convert_bbox_to_oriIm(pred_vert_vis, boxScale_o2n, bboxTopLeft, rawImg.shape)
    if False:
        data3D/=boxScale_o2n
        data3D[:,:2] += bboxTopLeft - imgSize*0.5 + hmrIntputSize_half/boxScale_o2n
    else:
        scale = 1.0/boxScale_o2n
        # print(f"Scale: {scale}")
        # deltaZ =  focalLeng/scale - np.mean(data3D[:,2]) 
        deltaZ =  np.mean(data3D[:,2])/scale - np.mean(data3D[:,2]) 
        data3D[:,2] +=deltaZ
        # data3D[:,2] += 400

        trans = bboxTopLeft - imgSize*0.5 + hmrIntputSize_half/boxScale_o2n
        delta = np.mean(data3D[:,2]) /focalLeng *trans
        # delta = (trans )*boxScale_o2n
        data3D[:,:2] += delta

        # newZ = focalLeng/scale
        # deltaZ =  newZ - np.mean(data3D[:,2])
        # data3D[:,2] +=deltaZ


    # data3D[:,1] += bboxTopLeft[1] - rawImg.shape[0]*0.5 + 112/boxScale_o2n
    return data3D




## Conversion for Antrho


def anthro_crop_fromRaw(rawimage, bbox_XYXY):
    bbox_w = bbox_XYXY[2] - bbox_XYXY[0]
    bbox_h = bbox_XYXY[3] - bbox_XYXY[1]
    bbox_size = max(bbox_w, bbox_h)     #take the max
    bbox_center = (bbox_XYXY[:2] + bbox_XYXY[2:])*0.5
    pt_ul = (bbox_center - bbox_size*0.5).astype(np.int32)
    pt_br = (bbox_center + bbox_size*0.5).astype(np.int32)
    croppedImg = rawimage[pt_ul[1]:pt_br[1], pt_ul[0]:pt_br[0]]
    croppedImg = np.ascontiguousarray(croppedImg)
    return rawimage, pt_ul, pt_br

def anthro_convert_smpl_to_bbox(data3D, scale, trans, bbox_max_size):
    hmrIntputSize_half = bbox_max_size *0.5

    data3D *= scale           #apply scaling
    # data3D[:,0] += data3D[b,1]        #apply translation x
    # data3D[:,1] += data3D[b,2]        #apply translation y
    data3D[:,0:2] += trans
    data3D*= hmrIntputSize_half         #112 is originated from hrm's input size (224,24)

    return data3D

# def anthro_convert_bbox_to_oriIm(data3D, boxScale_o2n, bboxTopLeft, imgSizeW, imgSizeH):
def anthro_convert_bbox_to_oriIm(pred_vert_vis, rawImg_w, rawImg_h, bbox_pt_ul, bbox_max_size):
    pred_vert_vis[:,:2] +=  bbox_pt_ul - np.array((rawImg_w, rawImg_h))*0.5 +(bbox_max_size*0.5)  # + hmrIntputSize_half#+ hmrIntputSize_half 
    return pred_vert_vis



""" Extract bbox information """
def bbox_from_openpose(openpose_file, rescale=1.2, detection_thresh=0.2):
    """Get center and scale for bounding box from openpose detections."""
    with open(openpose_file, 'r') as f:
        data = json.load(f)
        if 'people' not in data or len(data['people'])==0:
            return None, None
        # keypoints = json.load(f)['people'][0]['pose_keypoints_2d']
        keypoints = data['people'][0]['pose_keypoints_2d']
    keypoints = np.reshape(np.array(keypoints), (-1,3))
    valid = keypoints[:,-1] > detection_thresh

    valid_keypoints = keypoints[valid][:,:-1]           #(25,2)

    # min_pt = np.min(valid_keypoints, axis=0)
    # max_pt = np.max(valid_keypoints, axis=0)
    # bbox= [ min_pt[0], min_pt[1], max_pt[0] - min_pt[0], max_pt[1] - min_pt[1]]

    center = valid_keypoints.mean(axis=0)
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)).max()
    # adjust bounding box tightness
    scale = bbox_size / 200.0
    scale *= rescale
    return center, scale#, bbox


# keypoints: (Nx3)
def bbox_from_keypoint2d(keypoints, rescale=1.2, detection_thresh=0.2):
    """
        output:
            center: bbox center
            scale: scale_n2o: 224x224 -> original bbox size (max length if not a square bbox)
    """
    # """Get center and scale for bounding box from openpose detections."""

    if len(keypoints.shape)==2 and keypoints.shape[1]==2:       #(X,2)
        valid_keypoints = keypoints
    else:
        keypoints = np.reshape(np.array(keypoints), (-1,3))
        valid = keypoints[:,-1] > detection_thresh

        valid_keypoints = keypoints[valid][:,:-1]           #(25,2)

    # min_pt = np.min(valid_keypoints, axis=0)
    # max_pt = np.max(valid_keypoints, axis=0)
    # bbox= [ min_pt[0], min_pt[1], max_pt[0] - min_pt[0], max_pt[1] - min_pt[1]]

    center = valid_keypoints.mean(axis=0)
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)).max()


    # adjust bounding box tightness
    scale = bbox_size / 200.0
    scale *= rescale
    return center, scale#, bbox




# def bbox_from_bboxXYXY(bboxXYXY, rescale=1.2):
#     """
#     bboxXYXY
#     """
    # pass

def bbox_from_keypoints(keypoints, rescale=1.2, detection_thresh=0.2, imageHeight= None):
    """Get center and scale for bounding box from openpose detections."""
    # with open(openpose_file, 'r') as f:
    #     data = json.load(f)
    #     if 'people' not in data or len(data['people'])==0:
    #         return None, None
    #     # keypoints = json.load(f)['people'][0]['pose_keypoints_2d']
    #     keypoints = data['people'][0]['pose_keypoints_2d']
    keypoints = np.reshape(np.array(keypoints), (-1,3))
    valid = keypoints[:,-1] > detection_thresh

    # if g_debugUpperBodyOnly:    #Intentionally remove lower bodies
    #     valid[ [ 9,10,11,12,13,14, 22,23,24, 19,20,21] ] = False

    valid_keypoints = keypoints[valid][:,:-1]           #(25,2)

    if len(valid_keypoints)<2:
        return None, None, None


    if False:            #Should have all limbs and nose
        if np.sum(valid[ [ 2,3,4, 5,6,7, 9,10, 12,13,1,0] ]) <12:
            return None, None, None

    min_pt = np.min(valid_keypoints, axis=0)
    max_pt = np.max(valid_keypoints, axis=0)

    
    bbox= [ min_pt[0], min_pt[1], max_pt[0] - min_pt[0], max_pt[1] - min_pt[1]]



    # print(valid_keypoints)
    # print(valid)
    print(bbox)

    if imageHeight is not None:

        if valid[10]==False and valid[13] == False:  # No knees ub ioeb
            max_pt[1] = min(max_pt[1] + (max_pt[1]- min_pt[1]), imageHeight )
            bbox= [ min_pt[0], min_pt[1], max_pt[0] - min_pt[0], max_pt[1] - min_pt[1]]
            valid_keypoints = np.vstack( (valid_keypoints, np.array(max_pt)) )


        elif valid[11]==False and valid[14] == False: #No foot
            max_pt[1] = min(max_pt[1] + (max_pt[1]- min_pt[1])*0.2, imageHeight )
            bbox= [ min_pt[0], min_pt[1], max_pt[0] - min_pt[0], max_pt[1] - min_pt[1]]

            valid_keypoints = np.vstack( (valid_keypoints, np.array(max_pt)) )


    center = valid_keypoints.mean(axis=0)
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)).max()
    # adjust bounding box tightness
    scale = bbox_size / 200.0
    scale *= rescale
    return center, scale, bbox


def process_image_keypoints(img, keypoints, input_res=224):
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    img = img[:,:,::-1].copy() # PyTorch does not support negative stride at the moment

    center, scale, bbox = bbox_from_keypoints(keypoints, imageHeight = img.shape[0])
    if center is None:
        return None, None, None, None, None

    img, boxScale_o2n, bboxTopLeft = crop_bboxInfo(img, center, scale, (input_res, input_res))

    # viewer2D.ImShow(img, name='cropped', waitTime=1)        #224,224,3


    if img is None:
        return None, None, None, None, None


    # unCropped = uncrop(img, center, scale, (input_res, input_res))

    # if True:
    #     viewer2D.ImShow(img)
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2,0,1)
    norm_img = normalize_img(img.clone())[None]
    # return img, norm_img, img_original, boxScale_o2n, bboxTopLeft, bbox
    bboxInfo ={"center": center, "scale": scale, "bboxXYWH":bbox}
    return img, norm_img, boxScale_o2n, bboxTopLeft, bboxInfo


#bbr: (minX, minY, width, height)
def bbox_from_bbr(bbox_XYWH, rescale=1.2, detection_thresh=0.2, imageHeight= None):
    """Get center and scale for bounding box from openpose detections."""

    # bbox= bbr
    # if imageHeight is not None:

    #     if valid[10]==False and valid[13] == False:  # No knees ub ioeb
    #         max_pt[1] = min(max_pt[1] + (max_pt[1]- min_pt[1]), imageHeight )
    #         bbox= [ min_pt[0], min_pt[1], max_pt[0] - min_pt[0], max_pt[1] - min_pt[1]]
    #         valid_keypoints = np.vstack( (valid_keypoints, np.array(max_pt)) )


    #     elif valid[11]==False and valid[14] == False: #No foot
    #         max_pt[1] = min(max_pt[1] + (max_pt[1]- min_pt[1])*0.2, imageHeight )
    #         bbox= [ min_pt[0], min_pt[1], max_pt[0] - min_pt[0], max_pt[1] - min_pt[1]]

    #         valid_keypoints = np.vstack( (valid_keypoints, np.array(max_pt)) )


    center = bbox_XYWH[:2] + 0.5 * bbox_XYWH[2:]
    bbox_size = max(bbox_XYWH[2:])
    # adjust bounding box tightness
    scale = bbox_size / 200.0
    scale *= rescale
    return center, scale#, bbox_XYWH

def bbox_from_json(bbox_file):
    """Get center and scale of bounding box from bounding box annotations.
    The expected format is [top_left(x), top_left(y), width, height].
    """
    with open(bbox_file, 'r') as f:
        bbox = np.array(json.load(f)['bbox']).astype(np.float32)
    ul_corner = bbox[:2]
    center = ul_corner + 0.5 * bbox[2:]
    width = max(bbox[2], bbox[3])
    scale = width / 200.0
    # make sure the bounding box is rectangular
    return center, scale



def process_image_bbox(img_original, bbox_XYWH, input_res=224):
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    img_original = img_original[:,:,::-1].copy() # PyTorch does not support negative stride at the moment
    img = img_original.copy()

    center, scale = bbox_from_bbr(bbox_XYWH, imageHeight = img.shape[0])
    if center is None:
        return None, None,  None, None, None

    img, boxScale_o2n, bboxTopLeft = crop_bboxInfo(img, center, scale, (input_res, input_res))

    # viewer2D.ImShow(img, name='cropped', waitTime=1)        #224,224,3


    if img is None:
        return None, None,  None, None, None


    # unCropped = uncrop(img, center, scale, (input_res, input_res))

    # if True:
    #     viewer2D.ImShow(img)
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2,0,1)
    norm_img = normalize_img(img.clone())[None]

    bboxInfo ={"center": center, "scale": scale, "bboxXYWH":bbox_XYWH}
    return img, norm_img, boxScale_o2n, bboxTopLeft, bboxInfo



g_de_normalize_img = Normalize(mean=[ -constants.IMG_NORM_MEAN[0]/constants.IMG_NORM_STD[0]    , -constants.IMG_NORM_MEAN[1]/constants.IMG_NORM_STD[1], -constants.IMG_NORM_MEAN[2]/constants.IMG_NORM_STD[2]], std=[1/constants.IMG_NORM_STD[0], 1/constants.IMG_NORM_STD[1], 1/constants.IMG_NORM_STD[2]])
def deNormalizeBatchImg(normTensorImg):
    """
    Normalized Batch Img to original image
    Input: 
        normImg: normTensorImg in cpu
    Input: 
        deNormImg: numpy form
    """
    deNormImg = g_de_normalize_img(normTensorImg).numpy()
    deNormImg = np.transpose(deNormImg , (1,2,0) )*255.0
    deNormImg = deNormImg[:,:,[2,1,0]] 
    deNormImg = np.ascontiguousarray(deNormImg, dtype=np.uint8)

    return deNormImg




