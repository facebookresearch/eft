from __future__ import division

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
import numpy as np
import cv2
from os.path import join

from eft.cores import config
from eft.cores import constants
from eft.utils.imutils import crop, crop_bboxInfo, flip_img, flip_pose, flip_kp, transform, rot_aa

import json

g_debugMode = False      #DEBUG mode No augmentation, load only selected Idx

class BaseDataset(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in utils/config.py.
    """

    #bEnforceUpperOnly makes it always to do upper only cropping
    def __init__(self, options, dataset, ignore_3d=False, use_augmentation=True, is_train=True, bMiniTest=False, bEnforceUpperOnly= False):
        super(BaseDataset, self).__init__()
        self.datasetName = dataset
        self.is_train = is_train
        self.options = options
        self.img_dir = config.DATASET_FOLDERS[dataset]
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)

        #Upper body only test
        self.bEnforceUpperOnly = False
        if bEnforceUpperOnly or (self.options is not None and  self.options.extCrop_bEnforceUpperBody):
            print(">>  Enforcing Upper body Crop always!!!!!!!!!!!!!!!!")
            self.bEnforceUpperOnly = True

        if options is not None:

            if dataset=='coco3d' and options.db_coco3d_name is not None :
                config.SetDBName(dataset, options.db_coco3d_name)
            if dataset=='cocoall3d' and options.db_cocoall3d_name is not None :
                config.SetDBName(dataset, options.db_cocoall3d_name)
            if dataset=='mpii3d' and options.db_mpii3d_name is not None:
                config.SetDBName(dataset, options.db_mpii3d_name)
            if dataset=='lspet3d'and options.db_lspet3d_name is not None:
                config.SetDBName(dataset, options.db_lspet3d_name)
            if dataset=='posetrack3d' and options.db_posetrack3d_name is not None:
                config.SetDBName(dataset, options.db_posetrack3d_name)
            # if dataset=='cocoplus3d' and options.db_cocoplus3d_name is not None:
            #     config.SetDBName(dataset, options.db_cocoplus3d_name)

        # self.data = np.load(config.DATASET_FILES[is_train][dataset])
        self.data = np.load(config.DATASET_FILES[is_train][dataset],allow_pickle=True)      #For multibbox

        """ 
        The folling should have the same length

        self.imgname
        self.annotIds
        self.scale
        self.center
        self.pose
        self.keypoints
        self.gender
        self.has_smpl
        self.maskname
        self.partname
        self.trackIds

        This should reset: self.length

        """
        self.imgname = self.data['imgname']

        #COCO and Pose Track only
        self.annotIds = None
        if 'annotIds' in self.data.files:
            self.annotIds = self.data['annotIds']

            assert len(self.imgname)==len(self.annotIds)

        #Sequence data only
        self.subjectIds = None
        if 'subjectid' in self.data.files:
            # assert False    #Old code... to know when this was used
        #     self.subjectIds = self.data['subjectid']
            self.subjectIds = self.data['subjectid']

        if 'subjectIds' in self.data.files:
            self.subjectIds = self.data['subjectIds']

            assert len(self.imgname)==len(self.subjectIds)


        print(">>> BaseDataset:: Loading DB {}: {} samples: from {}".format(self.datasetName, len(self.imgname), config.DATASET_FILES[is_train][dataset]))
        
        # Get paths to gt masks, if available
        try:
            self.maskname = self.data['maskname']
        except KeyError:
            pass
        try:
            self.partname = self.data['partname']
        except KeyError:
            pass

        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data['scale']
        self.center = self.data['center']

        try:
            self.cam_param = self.data['cam_param']         #Newrly added to check reprojection error
        except KeyError:
            pass
        
        # If False, do not do augmentation
        self.use_augmentation = use_augmentation
        
        # Get gt SMPL parameters, if available
        try:
            self.pose = self.data['pose'].astype(np.float)
            self.betas = self.data['shape'].astype(np.float)
            if 'has_smpl' in self.data:
                self.has_smpl = self.data['has_smpl']
            else:
                self.has_smpl = np.ones(len(self.imgname))
        except KeyError:
            self.has_smpl = np.zeros(len(self.imgname))
        if ignore_3d:
            self.has_smpl = np.zeros(len(self.imgname))
        
        # Get gt 3D pose, if available
        try:
            self.pose_3d = self.data['S']
            self.has_pose_3d = 1
        except KeyError:
            self.has_pose_3d = 0
        if ignore_3d:
            self.has_pose_3d = 0
        
        # Get 2D keypoints
        try:
            keypoints_gt = self.data['part']
        except KeyError:
            keypoints_gt = np.zeros((len(self.imgname), 24, 3))
        try:
            keypoints_openpose = self.data['openpose']
        except KeyError:
            keypoints_openpose = np.zeros((len(self.imgname), 25, 3))
        self.keypoints = np.concatenate([keypoints_openpose, keypoints_gt], axis=1)


        # Get 2D hand keypoint (openpose order)
        try:
            self.kp_leftHand_gt = self.data['kp_leftHand']
        except KeyError:
            self.kp_leftHand_gt = None
        try:
            self.kp_rightHand_gt = self.data['kp_rightHand']
        except KeyError:
            self.kp_rightHand_gt = None


        # Get 2D face keypoint (openpose order)




        # Get hand data, if available... This if from 3D hand (e.g., Panoptic DB)
        if self.options is not None and self.options.bUseHand3D:
            try:
                #Out dated
                # self.rhand_2d = self.data['rhand_2d']
                # self.lhand_2d = self.data['lhand_2d']
                
                self.rhand_3d = self.data['rhand_3d']
                self.lhand_3d = self.data['lhand_3d']

            except KeyError:
                pass

        # Get gender data, if available
        try:
            gender = self.data['gender']
            self.gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)
        except KeyError:
            self.gender = -1*np.ones(len(self.imgname)).astype(np.int32)
        
        self.length = self.scale.shape[0]

        self.bLoadImage = True


        #Multi-level bbox
        # try:
        #     self.multilevel_cropinfo = self.data['multibbox']
        # except KeyError:
        #     self.multilevel_cropinfo = None
        if 'multibbox' in self.data.keys():
            self.multilevel_cropinfo = self.data['multibbox']
            print(f"Loading multilevel bbox crop info: {dataset}")
        else:
            self.multilevel_cropinfo = None
            print(f"No multilevel bbox crop info: {dataset}")

        # if self.is_train and (self.options.multilevel_crop>=0 or self.options.multilevel_crop_rand):
        #     assert self.multilevel_cropinfo  is not None

        # if self.options.multilevel_crop >=0:
        #     self.bMultiLevelCrop = True
        #     self.bMultiLevelCrop_level = self.options.multilevel_crop

        if self.bEnforceUpperOnly or (self.options is not None and self.options.bUpperBodyTest):
            scaleFactor = 1.2
            self.scale_upperbody = np.zeros(self.scale.shape)
            self.center_upperbody = np.zeros(self.center.shape)

            for i in range(self.keypoints.shape[0]):

                validity = self.keypoints[i,25:,2]==1       #(N, 24)
                # validity[ [0,1,2,3,4,5]] = False      #Disable lower bodies
                validity[ [0,1,4,5]] = False      #Disable lower bodies
 
                if np.sum(validity)>=3:
                    min_pt = np.min(self.keypoints[i,25:,:2][validity], axis=0)
                    max_pt = np.max(self.keypoints[i,25:,:2][validity], axis=0)
                    # bbox= [ min_pt[0], min_pt[1], max_pt[0], max_pt[1] ]
                    bbox= [ min_pt[0], min_pt[1], max_pt[0] - min_pt[0], max_pt[1] - min_pt[1]]
                    center = [bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]
                    scale = scaleFactor*max(bbox[2], bbox[3])/200

                    self.center_upperbody[i]= center
                    self.scale_upperbody[i]= scale
                else:
                    # self.center_upperbody[i]= [0,0]
                    self.scale_upperbody[i]= -1     #Garbage should be ignored

            
            self.center_facePart = np.zeros(self.center.shape)
            self.scale_facePart = np.zeros(self.scale.shape)        #More extreme case

            for i in range(self.keypoints.shape[0]):

                validity = self.keypoints[i,25:,2]==1       #(N, 24)
                validity[ [0,1,2,3,4,5,1,14,   6,11 ]] = False      #Disable, even hand, arms. Only consider shoulders

                if np.sum(validity)>=3:
                    min_pt = np.min(self.keypoints[i,25:,:2][validity], axis=0)
                    max_pt = np.max(self.keypoints[i,25:,:2][validity], axis=0)
                    # bbox= [ min_pt[0], min_pt[1], max_pt[0], max_pt[1] ]
                    bbox= [ min_pt[0], min_pt[1], max_pt[0] - min_pt[0], max_pt[1] - min_pt[1]]
                    center = [bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]
                    scale = scaleFactor*max(bbox[2], bbox[3])/200

                    self.center_facePart[i]= center
                    self.scale_facePart[i]= scale
                else:
                    # self.center_upperbody[i]= [0,0]
                    self.scale_facePart[i]= -1     #Garbage should be ignored

        if bMiniTest: #Sampling to 1/10 size
                # validIdx = range(0,len(self.imgname),10)
                if self.options.bExemplar_badsample_finder:
                    validIdx = range(0,len(self.imgname),100)
                else:
                    validIdx = range(0,len(self.imgname),10)
                print(">>> Debug: sampling originalSize {} -> miniSize {}".format(len(self.imgname), len(validIdx)))

                self.imgname = self.imgname[validIdx]
                try:
                    self.pose = self.pose[validIdx]
                except:
                    pass

                try:            
                    self.betas = self.betas[validIdx]
                except:
                    pass

                try:
                    self.pose_3d = self.pose_3d[validIdx]
                except:
                    pass
                
                try:
                    self.keypoints = self.keypoints [validIdx]
                except:
                    pass
                
                try:
                    self.keypoints = self.keypoints [validIdx]
                except:
                    pass
                try:
                    self.scale_upperbody = self.scale_upperbody[validIdx]
                except:
                    pass
                
                try:
                    self.center_upperbody = self.center_upperbody[validIdx]
                except:
                    pass

                # try:
                #     self.scale_facePart = self.scale_facePart[validIdx]
                # except:
                #     pass
                
                # try:
                #     self.center_facePart = self.center_facePart[validIdx]
                # except:
                #     pass
                

                self.has_smpl = self.has_smpl[validIdx]
                # self.has_pose_3d = self.has_pose_3d[validIdx]
                self.scale  =self.scale[validIdx]
                self.center= self.center[validIdx]
                self.gender= self.gender[validIdx]

                if self.annotIds is not None:
                    self.annotIds = self.annotIds[validIdx]

        test_3dpw = False
        if test_3dpw:
                #3DPW test
                # validIdx = range(0,len(self.imgname),10)
                validIdx =[]
                for i, n in enumerate(self.imgname):
                    if 'downtown_upstairs_00' in n:
                        validIdx.append(i)
                print(">>> Debug: sampling originalSize {} -> miniSize {}".format(len(self.imgname), len(validIdx)))

                self.imgname = self.imgname[validIdx]
                try:
                    self.pose = self.pose[validIdx]
                except:
                    pass

                try:            
                    self.betas = self.betas[validIdx]
                except:
                    pass

                try:
                    self.pose_3d = self.pose_3d[validIdx]
                except:
                    pass
                
                try:
                    self.keypoints = self.keypoints [validIdx]
                except:
                    pass
                
                try:
                    self.keypoints = self.keypoints [validIdx]
                except:
                    pass
                try:
                    self.scale_upperbody = self.scale_upperbody[validIdx]
                except:
                    pass
                
                try:
                    self.center_upperbody = self.center_upperbody[validIdx]
                except:
                    pass

                self.has_smpl = self.has_smpl[validIdx]
                # self.has_pose_3d = self.has_pose_3d[validIdx]
                self.scale  =self.scale[validIdx]
                self.center= self.center[validIdx]
                self.gender= self.gender[validIdx]

                if self.annotIds is not None:
                    self.annotIds = self.annotIds[validIdx]

        #Debug: choose specific range to run paralle (training code only)
        if self.options is not None and\
                     is_train == True and\
                          self.options.bExemplar_dataLoaderStart >=0:# and self.options.bExemplar_dataLoaderEnd >=0:

            startIdx = self.options.bExemplar_dataLoaderStart
            print(">> {}".format( len(self.imgname)))

            if self.options.bExemplar_dataLoaderEnd>=0:
                endIdx = min( self.options.bExemplar_dataLoaderEnd, len(self.imgname))
            else:
                endIdx = len(self.imgname)

            #Run only from bExemplar_dataLoaderStart ~ bExemplar_dataLoaderEnd
            validIdx = range(startIdx,endIdx)
            print(validIdx)
            
            self.imgname = self.imgname[validIdx]
            try:
                self.pose = self.pose[validIdx]
            except:
                pass
            try:
                self.betas = self.betas[validIdx]
            except:
                pass

            try:
                self.pose_3d = self.pose_3d[validIdx]
            except:
                pass

            try:
                self.multilevel_cropinfo = self.multilevel_cropinfo[validIdx]
            except:
                pass

            

            #Hand part (outdated)
            # try:
            #     self.rhand_2d = self.rhand_2d[validIdx]
            # except:
            #     pass
            # try:
            #     self.lhand_2d = self.lhand_2d[validIdx]
            # except:
            #     pass
            try:
                self.rhand_3d = self.rhand_3d[validIdx]
            except:
                pass
            
            try:
                self.lhand_3d = self.lhand_3d[validIdx]
            except:
                pass

            # self.has_pose_3d = self.has_pose_3d[validIdx]

            self.keypoints = self.keypoints [validIdx]
            self.has_smpl = self.has_smpl[validIdx]
            self.scale  =self.scale[validIdx]
            self.center= self.center[validIdx]
            self.gender= self.gender[validIdx]

            #Cam_param for EFT dataset only
            try:
                self.cam_param = self.cam_param[validIdx]
            except:
                pass



            #COCO Whole Keypoint DB
            try:
                self.kp_leftHand_gt = self.kp_leftHand_gt[validIdx]
            except:
                pass

            try:
                self.kp_rightHand_gt = self.kp_rightHand_gt[validIdx]
            except:
                pass

            if self.annotIds is not None:       #bug fixed
                self.annotIds = self.annotIds[validIdx]
            
            if self.subjectIds is not None:       #bug fixed
                self.subjectIds = self.subjectIds[validIdx]

            # self.dataset= self.dataset[validIdx]

            # item['maskname'] = self.maskname[index]
            # item['partname'] = ''
            print("Selected IDx: {} - {}. Total number {}".format(startIdx, endIdx, self.__len__()))


        #Debug: choose specific sequence
        if False:
            # seqName ='office_phoneCall_00'
            seqName ='downtown'

            validIdx =[]
            for i in range(len(self.imgname)):
                if seqName in self.imgname[i]:
                    validIdx.append(i)
            
            self.imgname = self.imgname[validIdx]
            self.pose = self.pose[validIdx]
            self.betas = self.betas[validIdx]
            # self.pose_3d = self.pose_3d[validIdx]
            self.keypoints = self.keypoints [validIdx]
            self.has_smpl = self.has_smpl[validIdx]
            # self.has_pose_3d = self.has_pose_3d[validIdx]
            self.scale  =self.scale[validIdx]
            self.center= self.center[validIdx]
            self.gender= self.gender[validIdx]
            # self.dataset= self.dataset[validIdx]

            # item['maskname'] = self.maskname[index]
            # item['partname'] = ''


        #Check
        dbLeng = len(self.imgname)
        self.length = dbLeng

        assert(dbLeng==len(self.scale)==len(self.center)==len(self.keypoints)==len(self.gender)==len(self.has_smpl))
        
        if self.annotIds is not None:   #Bug fixed
            dbLeng = len(self.annotIds)

        if hasattr(self,'subjectIds'):      #Bug fixed
            if self.subjectIds is not None:
                assert(dbLeng ==len(self.subjectIds))

        if hasattr(self,'cam_param'):
            assert(dbLeng ==len(self.cam_param))

        if hasattr(self,'pose'):
            assert(dbLeng ==len(self.pose))
        if hasattr(self,'betas'):
            assert(dbLeng ==len(self.betas))
        if hasattr(self,'pose_3d'):
            assert(dbLeng ==len(self.pose_3d))
        if hasattr(self,'maskname'):
            assert(dbLeng ==len(self.maskname))
        if hasattr(self,'partname'):
            assert(dbLeng ==len(self.partname))

        if self.multilevel_cropinfo is not None:
            assert(dbLeng == len(self.multilevel_cropinfo))

        #outdated
        # if hasattr(self,'rhand_2d'):
        #     assert(dbLeng ==len(self.rhand_2d))
        # if hasattr(self,'lhand_2d'):
        #     assert(dbLeng ==len(self.lhand_2d))
        if self.kp_leftHand_gt is not None:
            assert(dbLeng == len(self.kp_leftHand_gt))
        if self.kp_rightHand_gt is not None:
            assert(dbLeng == len(self.kp_rightHand_gt))


    def augm_params(self):
        """ augmentation parameters."""
        flip = 0            # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0            # rotation
        sc = 1            # scaling
        if self.is_train:        #Debug
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1
            
            # Each channel is multiplied with a number 
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1-self.options.noise_factor, 1+self.options.noise_factor, 3)
            
            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(2*self.options.rot_factor,
                    max(-2*self.options.rot_factor, np.random.randn()*self.options.rot_factor))
            
            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1+self.options.scale_factor,
                    max(1-self.options.scale_factor, np.random.randn()*self.options.scale_factor+1))
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0
        
        return flip, pn, rot, sc

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn):
        
        bDebug = False
        
        if bDebug:        #Debug visualize
            from renderer import viewer2D
            viewer2D.ImShow(rgb_img.astype(np.uint8), waitTime=0)

        """Process rgb image and do augmentation."""        
        rgb_img = crop(rgb_img, center, scale, 
                      [constants.IMG_RES, constants.IMG_RES], rot=rot)
        # rgb_img, boxScale_o2n, bboxTopLeft  = crop_bboxInfo(rgb_img, center, scale, 
        #               [constants.IMG_RES, constants.IMG_RES], rot=rot)

        # img, boxScale_o2n, bboxTopLeft = crop_bboxInfo(img, center, scale, (input_res, input_res))


        if rgb_img is None:
            return None

        # flip the image 
        if flip:
            rgb_img = flip_img(rgb_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))


        if bDebug:        #Debug visualize
            print("center{}, scale{}, rot{}, flip{}, pn{}".format(center, scale, rot, flip, pn))
            from renderer import viewer2D
            viewer2D.ImShow(rgb_img, waitTime=0,name='cropped')

        # (3,224,224),float,[0,1]
        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        return rgb_img

    def j2d_processing(self, kp, center, scale, r=0, f=False):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = transform(kp[i,0:2]+1, center, scale, 
                                  [constants.IMG_RES, constants.IMG_RES], rot=r)
        # convert to normalized coordinates
        kp[:,:-1] = 2.*kp[:,:-1]/constants.IMG_RES - 1.         #-1 to 1
        # flip the x coordinates
        if f:
             kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

    def j3d_processing(self, S, r, f):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn,cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0,:2] = [cs, -sn]
            rot_mat[1,:2] = [sn, cs]
        S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1]) 
        # flip the x coordinates
        if f:
            # S[:,:-1] = flip_kp(S[:, :-1])
            S = flip_kp(S)      #https://github.com/nkolot/SPIN/commit/3c5d02bd4549d4d108e0e13bbc5f62e41ae67295
        S = S.astype('float32')
        return S

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def __getitem__(self, index):

        # bLoadImage = self.bLoadImage
        # print("Load: {}".format(index) )

        if g_debugMode: ###############  DEBUG MODE ###############  
            import os
            if os.path.exists("exemplarTestTarget.txt"):        #Read target index from this file 
                with open("exemplarTestTarget.txt","r") as f:
                    targetIdx =  f.readlines()[0]
                    print("targetImgFrame: {}".format(targetIdx))
                    self.options.exemplar_targetIdx = targetIdx #Save this. If a new one model should be reinitiated in base_trainer.py
            else: 
                targetIdx = '001'
            targetIdx = '004906'
            targetIdx = '468792'
            for i in range(len(self.imgname)):
                if targetIdx in self.imgname[i]:
                    index =i
                    break
            print("DEBUG: selected index {}".format(index))
                
        # index=347 #TODO
        # index = 8940 #TODO
        # index = 665 #TODO

        # for i in range(len(self.imgname)):
        #     # if 'COCO_train2014_000000000322' in self.imgname[i]:
        #     if '00_15_00030840' in self.imgname[i]:
        #         index =i
        #         break
        # print("DEBUG: selected index {}".format(index))
        #Good panoptic sample: 171204_pose6/00003110/00_00_00003110.jpg

        item = {}

        #Get scale/center from multilevel crop information using the chosing crop level [0,7]
        if self.is_train and \
                 self.multilevel_cropinfo is not None and  \
                    self.options is not None and   \
                    self.options.multilevel_crop_rand:

            # print(f"{self.datasetName}: yes multilevel")
            randomNum  =np.random.uniform()
            if randomNum <= self.options.multilevel_crop_rand_prob:

                cropLevel = np.random.randint(1,high=7)     #random 1-6 .. ignore 7 which is whole body
                # print(cropLevel)
                assert cropLevel>=1 and cropLevel<=7
                # print(f"Apply multi-crop: level{cropLevel}")
                scale = np.array(self.multilevel_cropinfo[index][cropLevel]['scale'])
                center = np.array(self.multilevel_cropinfo[index][cropLevel]['center'])
            else:
                scale = self.scale[index].copy()
                center = self.center[index].copy()
        elif self.is_train and self.options is not None and self.options.multilevel_crop>=0:
            cropLevel = self.options.multilevel_crop
            # print(f"Apply multi-crop: level{cropLevel}")
            scale = np.array(self.multilevel_cropinfo[index][cropLevel]['scale'])
            center = np.array(self.multilevel_cropinfo[index][cropLevel]['center'])


        elif self.bEnforceUpperOnly:      #Always upper only 
            if self.scale_upperbody[index]>0:
                scale = self.scale_upperbody[index].copy()
                center = self.center_upperbody[index].copy()
            else:   #except no valid cropping
                scale = self.scale[index].copy()
                center = self.center[index].copy()

        elif self.is_train and (self.options is not None and self.options.bUpperBodyTest):

            randomNum  =np.random.uniform()
            # print(randomNum)
            if self.scale_upperbody[index]>0 and randomNum <= self.options.upperBodyTest_prob:
                
                if self.options.bFacePartTest:     
                    randomNumAgain  =np.random.uniform()
                    if randomNumAgain<0.5 and self.scale_facePart[index]>0:
                        # print("facepart")
                        scale = self.scale_facePart[index].copy()
                        center = self.center_facePart[index].copy()
                    else:
                        # print("upperbody")
                        scale = self.scale_upperbody[index].copy()
                        center = self.center_upperbody[index].copy()
                        
                else:
                    scale = self.scale_upperbody[index].copy()
                    center = self.center_upperbody[index].copy()

            else:
                scale = self.scale[index].copy()
                center = self.center[index].copy()
        else:
            # print(f"{self.datasetName}: no multilevel")
            scale = self.scale[index].copy()
            center = self.center[index].copy()

        # Get augmentation parameters
        if (self.options is not None and self.options.bExemplarMode) or g_debugMode:    #Debug: No augmentation#g_debugMode:     ###############  DEBUG MODE ###############  
            flip = 0            # flipping
            pn = np.ones(3)  # per channel pixel-noise
            rot = 0            # rotation
            sc = 1   
        else:
            flip, pn, rot, sc = self.augm_params()
        
        # Load image
        imgname = join(self.img_dir, self.imgname[index])
        # print("dbName: {} | imgname: {}".format(self.datasetName, imgname))

        if self.bLoadImage:
            try:
                img = cv2.imread(imgname)[:,:,::-1].copy().astype(np.float32)           ##Note: BGR to RGB. We always use RGB
            except TypeError:
                print("Error: cannnot find image from: {}".format(imgname) )
            orig_shape = np.array(img.shape)[:2]
        else:
            orig_shape =0

        # Get SMPL parameters, if available
        if self.has_smpl[index]:
            pose = self.pose[index].copy()
            betas = self.betas[index].copy()
        else:
            pose = np.zeros(72)
            betas = np.zeros(10)

        # Process image
        if self.bLoadImage:
            try:
                img = self.rgb_processing(img, center, sc*scale, rot, flip, pn)
            except:
                # print("Warning: image processing has been failed: {}".format(imgname))
                print("Warning: image processing has been failed: {}: sc{}, rot{}, flipt{},pn{}".format(imgname,sc,rot,flip,pn))
                img = np.zeros((3,224,224), dtype=np.float32)       #generate blank image

            if img is None:
                # print("Warning: image processing has been failed: {}".format(imgname))
                print("Warning: image processing has been failed: {}: sc{}, rot{}, flipt{},pn{}".format(imgname,sc,rot,flip,pn))
                img = np.zeros((3,224,224), dtype=np.float32)       #generate blank image

            img = torch.from_numpy(img).float()
            item['img'] = self.normalize_img(img)
        else:
            item['img'] =''


        # Store image before normalization to use it in visualization
        item['pose'] = torch.from_numpy(self.pose_processing(pose, rot, flip)).float()
        item['betas'] = torch.from_numpy(betas).float()
        item['imgname'] = imgname
        if self.annotIds is not None:
            item['annotId'] = self.annotIds[index]
        else:
            item['annotId'] = -1        #Garbage

        # print("Debug: {}".format(item['annotId']))

        # Get 3D pose, if available
        if self.has_pose_3d:
            S = self.pose_3d[index].copy()
            item['pose_3d'] = torch.from_numpy(self.j3d_processing(S, rot, flip)).float()
        else:
            item['pose_3d'] = torch.zeros(24,4, dtype=torch.float32)

        if self.subjectIds is not None:
            item['subjectId'] = self.subjectIds[index]
        else:
            item['subjectId'] = ""        #Garbage
        # print(item['subjectId'])

        #Processing 3D Hands
        if self.options != None and self.options.bUseHand3D:
            # S = np.ones( (self.rhand_3d[index].shape[0], self.rhand_3d[index].shape[1]+1)   )
            # S[:,:-1] = self.rhand_3d[index].copy()     #
            S = self.rhand_3d[index].copy()     #
            item['rhand_3d'] = torch.from_numpy(self.j3d_processing(S, rot, flip)).float()

            # S = np.ones( (self.lhand_3d[index].shape[0], self.lhand_3d[index].shape[1]+1)   )
            # S[:,:-1] = self.lhand_3d[index].copy()
            S = self.lhand_3d[index].copy()
            item['lhand_3d'] = torch.from_numpy(self.j3d_processing(S, rot, flip)).float()            
        # else:
        #     item['pose_3d'] = torch.zeros(24,4, dtype=torch.float32)

        # Get 2D keypoints and apply augmentation transforms
        keypoints = self.keypoints[index].copy()            #(49,3)
        #Disable specifically foot, if too close to boundarys
        imgHeight = orig_shape[0]
        if abs(keypoints[25+0,1] - imgHeight)<10 and keypoints[10,2]<0.1:        #Right Foot. within 10 pix from the boundary
            keypoints[25+0,2] = 0 #Disable
        if abs(keypoints[25+5,1] - imgHeight)<10 and keypoints[13,2]<0.1:       #Left Foot. 
            keypoints[25+5,2] =0 #Disable
        item['keypoints_original'] = self.keypoints[index].copy()       #In original space
        item['keypoints'] = torch.from_numpy(self.j2d_processing(keypoints, center, sc*scale, rot, flip)).float()       #Processing to make in bbox space

        # 2D Hand keypoints
        if self.kp_leftHand_gt is not None:
            lhand_keypoints = self.kp_leftHand_gt[index].copy()            #(49,3)
            item['kp_leftHand_gt'] = torch.from_numpy(self.j2d_processing(lhand_keypoints, center, sc*scale, rot, flip)).float()       #Processing to make in bbox space

        if self.kp_rightHand_gt is not None:
            rhand_keypoints = self.kp_rightHand_gt[index].copy()            #(49,3)
            item['kp_rightHand_gt'] = torch.from_numpy(self.j2d_processing(rhand_keypoints, center, sc*scale, rot, flip)).float()       #Processing to make in bbox space

        item['has_smpl'] = self.has_smpl[index]
        item['has_pose_3d'] = self.has_pose_3d
        item['scale'] = float(sc * scale)
        item['center'] = center.astype(np.float32)
        item['orig_shape'] = orig_shape
        item['is_flipped'] = flip
        item['rot_angle'] = np.float32(rot)
        item['gender'] = self.gender[index]
        item['sample_index'] = index
        item['dataset_name'] = self.datasetName

        if self.is_train == False:
            try:
                item['cam_param'] = self.cam_param[index]
            except AttributeError:
                item['cam_param'] = ''



        #for Panoptic DB.... to check existence...bad code.
        # parentIdx = 100* (int(index /100.0) + 1)
        # item['pkl_save_name'] = join(self.img_dir,self.imgname[parentIdx])

        try:
            item['maskname'] = self.maskname[index]
        except AttributeError:
            item['maskname'] = ''
        try:
            item['partname'] = self.partname[index]
        except AttributeError:
            item['partname'] = ''

        return item

    def __len__(self):
        return len(self.imgname)
    