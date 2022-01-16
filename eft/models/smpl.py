# Copyright (c) Facebook, Inc. and its affiliates.

#Modified from https://github.com/nkolot/SPIN/blob/master/LICENSE

import torch
import numpy as np
from smplx import SMPL as _SMPL
# from smplx.body_models import ModelOutput       #old version of smplx: 0.1.13
from smplx.body_models import SMPLOutput       #old version of smplx: 0.1.13
from smplx.lbs import vertices2joints

import eft.cores.config as config
import eft.cores.constants as constants
import eft.cores.jointorders as jointorders

from collections import namedtuple
# ModelOutput = namedtuple('ModelOutput',
#                          ['vertices', 'joints', 'full_pose', 'betas',
#                           'global_orient',
#                           'body_pose', 'expression',
#                           'left_hand_pose', 'right_hand_pose',
#                           'right_hand_joints', 'left_hand_joints',
#                           'jaw_pose'])

# class SMPL(_SMPL):   

#     def __init__(self, *args, **kwargs):
#         super(SMPL, self).__init__(*args, **kwargs)
#         self.joint_map_smpl45_to_openpose19 = torch.tensor(jointorders.JOINT_MAP_SMPL45_TO_OPENPOSE18, dtype=torch.long)

#     def forward(self, *args, **kwargs):
#         kwargs['get_skin'] = True
#         smpl_output = super(SMPL, self).forward(*args, **kwargs)
#         reordered_joints = smpl_output.joints[:, self.joint_map_smpl45_to_openpose19, :]       #Reordering

#         new_output = SMPLOutput(vertices=smpl_output.vertices,
#                              global_orient=smpl_output.global_orient,
#                              body_pose=smpl_output.body_pose,
#                              joints=reordered_joints,
#                              betas=smpl_output.betas,
#                              full_pose=smpl_output.full_pose)

#         return new_output

class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        joints = [constants.JOINT_MAP[i] for i in constants.JOINT_NAMES]
        J_regressor_extra = np.load(config.JOINT_REGRESSOR_TRAIN_EXTRA)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
        self.joint_map = torch.tensor(joints, dtype=torch.long)

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)        #Additional 9 joints #Check doc/J_regressor_extra.png
        joints = torch.cat([smpl_output.joints, extra_joints], dim=1)               #[N, 24 + 21, 3]  + [N, 9, 3]
        joints = joints[:, self.joint_map, :]
        output = SMPLOutput(vertices=smpl_output.vertices,
                             global_orient=smpl_output.global_orient,
                             body_pose=smpl_output.body_pose,
                             joints=joints,
                             betas=smpl_output.betas,
                             full_pose=smpl_output.full_pose)
        return output
