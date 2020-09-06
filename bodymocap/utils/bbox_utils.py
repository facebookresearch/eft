import numpy as np
def bbox_conv_bboxXYHW_to_bbox_centerscale(bbox_XYWH, rescale=1.2, imageHeight= None):
    """
    Bbox conversion from bbox_xywh to (bbox_center, bbox_scale)
        # bbox_center: center of the bbox in original cooridnate
        # bbox_scale: scaling factor applied to the original image, before applying 224x224 cropping 

        # In the code: 200, and rescale ==1.2 are some magic numbers used for dataset generation in SPIN code
    """

    center = bbox_XYWH[:2] + 0.5 * bbox_XYWH[2:]
    bbox_size = max(bbox_XYWH[2:])
    # adjust bounding box tightness
    scale = bbox_size / 200.0           #2
    scale *= rescale
    return center, scale#, bbox_XYWH

def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
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


def bbox_conv_centerscale_scaleo2n_topleft(center,scale, hmr_res = (224,224)):
    """
    from (center, scale) -> (scale_o2n, bbox_topleft)
    """
    
    
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

#Aliasing    
def conv_bboxinfo_bboxXYXY(scale, center):
    """
    from (scale, center) -> (topleft, bottom right)
    """
    return conv_bboxinfo_centerscale_to_bboxXYXY(center, scale)

def conv_bboxinfo_centerscale_to_bboxXYXY(center, scale):
    """
    from (center, scale) ->  (topleft, bottom right)  or (minX,minY,maxX,maxY)
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
        scale: scaling images before cropping. reference size is 200 pix (why??). >1.0 means size up, <1.0 means size down. See get_transform()
                h = 200 * scale
                t = np.zeros((3, 3))
                t[0, 0] = float(res[1]) / h
                t[1, 1] = float(res[0]) / h
                t[0, 2] = res[1] * (-float(center[0]) / h + .5)
                t[1, 2] = res[0] * (-float(center[1]) / h + .5)
                t[2, 2] = 1
    """

    center = [bbox_xyhw[0] + bbox_xyhw[2]/2, bbox_xyhw[1] + bbox_xyhw[3]/2]

    if bLooseBox:
        scaleFactor =1.2
        scale = scaleFactor*max(bbox_xyhw[2], bbox_xyhw[3])/200       #This is the one used in SPIN's pre-processing. See preprocessdb/coco.py
    else:
        scale = max(bbox_xyhw[2], bbox_xyhw[3])/200   

    return center, scale