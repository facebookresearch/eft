# Copyright (c) Facebook, Inc. and its affiliates.

#Modified from https://github.com/nkolot/SPIN/blob/master/LICENSE

import torch
import numpy as np
from smplx import SMPL as _SMPL
from smplx.body_models import ModelOutput
from smplx.lbs import vertices2joints

# import eft.cores.config as config
# import eft.cores.constants as constants
import eft.cores.jointorders as jointorders

class SMPL(_SMPL):   

    def __init__(self, *args, **kwargs):
        """
        Initialize the map.

        Args:
            self: (todo): write your description
        """
        super(SMPL, self).__init__(*args, **kwargs)
        self.joint_map_smpl45_to_openpose19 = torch.tensor(jointorders.JOINT_MAP_SMPL45_TO_OPENPOSE18, dtype=torch.long)

    def forward(self, *args, **kwargs):
        """
        Parameters ---------- forward.

        Args:
            self: (todo): write your description
        """
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        reordered_joints = smpl_output.joints[:, self.joint_map_smpl45_to_openpose19, :]       #Reordering

        new_output = ModelOutput(vertices=smpl_output.vertices,
                             global_orient=smpl_output.global_orient,
                             body_pose=smpl_output.body_pose,
                             joints=reordered_joints,
                             betas=smpl_output.betas,
                             full_pose=smpl_output.full_pose)

        return new_output
