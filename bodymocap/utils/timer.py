# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Timing related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time


class Timer(object):
    """A simple timer."""

    def __init__(self):
        """
        Reset the internal state.

        Args:
            self: (todo): write your description
        """
        self.reset()

    def tic(self):
        """
        Start the timer.

        Args:
            self: (todo): write your description
        """
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True, bPrint=False,title="Time"):
        """
        Prints the results

        Args:
            self: (todo): write your description
            average: (bool): write your description
            bPrint: (todo): write your description
            title: (str): write your description
        """
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            if bPrint:
                # print("Avg Time: {}".format(self.average_time))
                print("{}: {:0.2} sec/frame, FPS {:0.2}".format(title, self.diff, 1.0/self.diff))

            return self.average_time
        else:
            if bPrint:
                print("{}: {}, FPS {}".format(title, self.diff , 1.0/self.diff))
            return self.diff

    def reset(self):
        """
        Reset the progress bar.

        Args:
            self: (todo): write your description
        """
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
