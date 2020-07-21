# Copyright (c) Facebook, Inc. and its affiliates.

#Spread Sheet: https://docs.google.com/spreadsheets/d/1_1dLdaX-sbMkCKr_JzJW_RZCpwBwd7rcKkWT_VgAQ_0/edit#gid=0

#From SPIN paper: https://github.com/nkolot/SPIN/blob/master/LICENSE
SPIN49_JOINT_NAMES = [
'OP_Nose', 'OP_Neck', 'OP_RShoulder',           #0,1,2
'OP_RElbow', 'OP_RWrist', 'OP_LShoulder',       #3,4,5
'OP_LElbow', 'OP_LWrist', 'OP_MidHip',          #6, 7,8
'OP_RHip', 'OP_RKnee', 'OP_RAnkle',             #9,10,11
'OP_LHip', 'OP_LKnee', 'OP_LAnkle',             #12,13,14
'OP_REye', 'OP_LEye', 'OP_REar',                #15,16,17
'OP_LEar', 'OP_LBigToe', 'OP_LSmallToe',        #18,19,20
'OP_LHeel', 'OP_RBigToe', 'OP_RSmallToe', 'OP_RHeel',  #21, 22, 23, 24  ##Total 25 joints  for openpose
'Right Ankle', 'Right Knee', 'Right Hip',               #0,1,2
'Left Hip', 'Left Knee', 'Left Ankle',                  #3, 4, 5
'Right Wrist', 'Right Elbow', 'Right Shoulder',     #6
'Left Shoulder', 'Left Elbow', 'Left Wrist',            #9
'Neck (LSP)', 'Top of Head (LSP)',                      #12, 13
'Pelvis (MPII)', 'Thorax (MPII)',                       #14, 15
'Spine (H36M)', 'Jaw (H36M)',                           #16, 17
'Head (H36M)', 'Nose', 'Left Eye',                      #18, 19, 20
'Right Eye', 'Left Ear', 'Right Ear'                    #21,22,23 (Total 24 joints)
]
SPIN49_JOINT_IDS = {SPIN49_JOINT_NAMES[i]: i for i in range(len(SPIN49_JOINT_NAMES))}

OPENPOSE25_JOINT_NAMES = [
'OP_Nose', 'OP_Neck', 'OP_RShoulder',           #0,1,2
'OP_RElbow', 'OP_RWrist', 'OP_LShoulder',       #3,4,5
'OP_LElbow', 'OP_LWrist', 'OP_MidHip',          #6, 7,8
'OP_RHip', 'OP_RKnee', 'OP_RAnkle',             #9,10,11
'OP_LHip', 'OP_LKnee', 'OP_LAnkle',             #12,13,14
'OP_REye', 'OP_LEye', 'OP_REar',                #15,16,17
'OP_LEar', 'OP_LBigToe', 'OP_LSmallToe',        #18,19,20
'OP_LHeel', 'OP_RBigToe', 'OP_RSmallToe', 'OP_RHeel',  #21, 22, 23, 24  ##Total 25 joints  for openpose
]
OPENPOSE25_JOINT_IDS = {OPENPOSE25_JOINT_NAMES[i]: i for i in range(len(OPENPOSE25_JOINT_NAMES))}

OPENPOSE18_JOINT_NAMES = [
'OP_Nose', 'OP_Neck', 'OP_RShoulder',           #0,1,2
'OP_RElbow', 'OP_RWrist', 'OP_LShoulder',       #3,4,5
'OP_LElbow', 'OP_LWrist',           #6, 7
'OP_RHip', 'OP_RKnee', 'OP_RAnkle',             #8, 9,10,
'OP_LHip', 'OP_LKnee', 'OP_LAnkle',             #11, 12,13,
'OP_REye', 'OP_LEye', 'OP_REar',                #14, 15,16,
'OP_LEar',      #17
]
OPENPOSE18_JOINT_IDS = {OPENPOSE18_JOINT_NAMES[i]: i for i in range(len(OPENPOSE18_JOINT_NAMES))}

SMPL45_JOINT_NAMES = [
'OP_MidHip',      #0    #TODO: Not exactly the same as OP
'OP_LHip', 'OP_RHip',    #1, #2 #TODO: Not exactly the same as OP
'SMPL_SPIN1', #3
'OP_LKnee', #4
'OP_RKnee', #5
'SMPL_SPIN2', #6
'OP_LAnkle', #7
'OP_RAnkle', #8
'SMPL_SPIN3', #9
'SMPL_Left_Foot', #10
'SMPL_Right_Foot', #11
'OP_Neck', #12
'SMPL_Left_Collar', #13
'SMPL_Right_Collar', #14
'SMPL_Head', #15
'OP_LShoulder', #16     #TODO: Not exactly the same as OP
'OP_RShoulder', #17     #TODO: Not exactly the same as OP
'OP_LElbow', #18
'OP_RElbow', #19
'OP_LWrist', # 20
'OP_RWrist', # 21
'SMPL_Blank22', # 22
'SMPL_Blank23', # 23
'OP_Nose', # 24
'OP_REye', # 25
'OP_LEye', # 26
'OP_REar', # 27
'OP_LEar', # 28
]
SMPL45_JOINT_IDS = {SMPL45_JOINT_NAMES[i]: i for i in range(len(SMPL45_JOINT_NAMES))}


#### Joint Maps ####

#SMPL45_Joint[JOINT_MAP_SMPL45_TO_OPENPOSE18] == OPENPOSE18_Joint
JOINT_MAP_SMPL45_TO_OPENPOSE18 = [SMPL45_JOINT_IDS[name] for name in OPENPOSE18_JOINT_IDS]

#SPIN49_Joint[JOINT_MAP_SPIN49_TO_OPENPOSE18] == OPENPOSE18_Joint
JOINT_MAP_SPIN49_TO_OPENPOSE18 = [SPIN49_JOINT_IDS[name] for name in OPENPOSE18_JOINT_IDS]



SPIN24_JOINT_NAMES = [
'OP_RAnkle', 'OP_RKnee', 'OP_RHip',                 #0, 1, 2
'OP_LHip', 'OP_LKnee', 'OP_LAnkle',                  #3, 4, 5
'OP_RWrist', 'OP_RElbow', 'OP_RShoulder',     #6
'OP_LShoulder', 'OP_LElbow', 'OP_LWrist',            #9
'OP_Neck', 'Top of Head (LSP)',                      #12, 13
'Pelvis (MPII)', 'Thorax (MPII)',                       #14, 15
'Spine (H36M)', 'Jaw (H36M)',                           #16, 17
'Head (H36M)', 'OP_Nose', 'OP_LEye',                      #18, 19, 20
'OP_REye', 'OP_LEar', 'OP_REar'                    #21,22,23 (Total 24 joints)
]
SPIN24_JOINT_IDS = {SPIN24_JOINT_NAMES[i]: i for i in range(len(SPIN24_JOINT_NAMES))}

JOINT_MAP_SPIN24_TO_OPENPOSE18 = [SPIN24_JOINT_IDS[name] for name in OPENPOSE18_JOINT_IDS]
