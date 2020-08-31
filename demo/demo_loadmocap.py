# Copyright (c) Facebook, Inc. and its affiliates.

import os
import sys
import torch
import numpy as np
import cv2
import argparse
import json
import pickle
from bodymocap.models import SMPL

############# input parameters  #############
from bodymocap.core import config 
from renderer import viewer2D#, glViewer
from renderer.visualizer import Visualizer

from bodymocap.utils.timer import Timer
g_timer = Timer()

parser = argparse.ArgumentParser()
parser.add_argument('--mocapdir', type=str, default=None, help='Folder of output images.')

def get_video_path(args):
    if args.webcam:
        video_path = 0
    elif args.url:
        if args.download:
            os.makedirs("./webvideos",exist_ok=True)
            downloadPath ="./webvideos/{0}.mp4".format(os.path.basename(args.url))
            cmd_download = "youtube-dl -f best {0} -o {1}".format(args.url,downloadPath)
            print(">> Downloading: {}".format(args.url))
            print(">> {}".format(cmd_download))
            #download via youtube-dl
            os.system(cmd_download)
            video_path = downloadPath
        else:
            try:
                import pafy
                url = args.url #'https://www.youtube.com/watch?v=c5nhWy7Zoxg'
                vPafy = pafy.new(url)
                play = vPafy.getbest(preftype="webm")
                video_path = play.url
                video_path = url
            except:
                video_path = args.url
    elif args.vPath:
        video_path = args.vPath
    else:
        assert False
    return video_path


def RunMonomocap(args, smpl, mocapDir, visualizer):
    fileNames = sorted(os.listdir(mocapDir))

    print(f"Loading mocap data from: {mocapDir}")
    mocapDataAll = []
    for fname in fileNames:

        filePath = os.path.join(mocapDir, fname)

        with open(filePath,'rb') as f:
            mocapData_frame = pickle.load(f)

        mocapDataAll.append(mocapData_frame)

    frameNum = len(mocapDataAll)
    frameIter =0
    assert len(mocapDataAll)>0
    while(True):#for mocapData_frame in mocapDataAll:

        print(f"frameIdx:{frameIter} / {frameNum} ")
        mocapData_frame = mocapDataAll[frameIter]

        frameIter+=1
        if frameIter >=frameNum:
            frameIter =0    #Looping

        # personNum = len(mocapData_frame)
        for mocapData in mocapData_frame:

            pred_betas = torch.from_numpy( mocapData['pred_betas'][np.newaxis,:])
            pred_rotmat = torch.from_numpy( mocapData['pred_rotmat'][np.newaxis,:])
            pred_vertices_imgspace =mocapData['pred_vertices_imgspace']
            pred_joints_imgspace =mocapData['pred_joints_imgspace']

            # pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            # pred_vertices = pred_output.vertices
            # pred_joints_3d = pred_output.joints
            # pred_vertices = pred_vertices[0].cpu().numpy()
            # pred_camera = pred_camera[0].numpy().ravel()
            # camScale = pred_camera[0]#
            # camTrans = pred_camera[1:]
            # #Convert mesh to original image space (X,Y are aligned to image)
            # pred_vertices_bbox = convert_smpl_to_bbox(pred_vertices, camScale, camTrans)  #SMPL -> 2D bbox
            # pred_vertices_img = convert_bbox_to_oriIm(pred_vertices_bbox, boxScale_o2n, bboxTopLeft, img_original.shape[1], img_original.shape[0])       #2D bbox -> original 2D image

            tempMesh = {'ver': pred_vertices_imgspace, 'f':  smpl.faces}
            meshList=[]
            skelList=[]
            bboxXYWH_list=[]
            meshList.append(tempMesh)
            skelList.append(pred_joints_imgspace.ravel()[:,np.newaxis])  #(49x3, 1)

            # visualizer.visualize_screenless_naive(meshList, skelList, bboxXYWH_list, img_original_bgr)
            visualizer.visualize_gui_naive(meshList, skelList, bboxXYWH_list)
        # g_timer.toc(average =True, bPrint=True,title="Detect+Regress+Vis")

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    # checkpoint = args.checkpoint
    # video_path = get_video_path(args)
    mocapDir = args.mocapdir

    visualizer = Visualizer('gui')

    smplModelPath = './extradata/smpl//basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
    smpl = SMPL(smplModelPath, batch_size=1, create_transl=False).to('cpu')

    RunMonomocap(args, smpl, mocapDir, visualizer)