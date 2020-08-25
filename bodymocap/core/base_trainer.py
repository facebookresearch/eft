# Original code from SPIN: https://github.com/nkolot/SPIN

from __future__ import division
import sys
import time

import torch
from tqdm import tqdm
tqdm.monitor_interval = 0
from torch.utils.tensorboard import SummaryWriter

# from .utils import CheckpointDataLoader, CheckpointSaver
from bodymocap.utils import CheckpointDataLoader, CheckpointSaver

# from datasets import BaseDataset
# from datasets import BaseDataset


# g_exemplarOutputPath = '/run/media/hjoo/disk/data/cocoPose3D_amt/0_SPIN/0_exemplarOutput/'

import pickle
import copy
from torchgeometry import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis

from bodymocap.utils.timer import Timer
g_timer = Timer()

class BaseTrainer(object):
    """Base class for Trainer objects.
    Takes care of checkpointing/logging/resuming training.
    """
    def __init__(self, options):
        self.options = options
        self.endtime = time.time() + self.options.time_to_run
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # override this function to define your model, optimizers etc.
        self.init_fn()
        self.saver = CheckpointSaver(save_dir=options.checkpoint_dir)
        self.summary_writer = SummaryWriter(self.options.summary_dir)

        self.checkpoint = None
        if self.options.resume and self.saver.exists_checkpoint():
            self.checkpoint = self.saver.load_checkpoint(self.models_dict, self.optimizers_dict, checkpoint_file=self.options.checkpoint)

        if self.checkpoint is None:
            self.epoch_count = 0
            self.step_count = 0
        else:
            self.epoch_count = self.checkpoint['epoch']
            self.step_count = self.checkpoint['total_step_count']

    def load_pretrained(self, checkpoint_file=None):
        """Load a pretrained checkpoint.
        This is different from resuming training using --resume.
        """
        if checkpoint_file is not None:
            checkpoint = torch.load(checkpoint_file)
            for model in self.models_dict:
                if model in checkpoint:
                    self.models_dict[model].load_state_dict(checkpoint[model], strict=False)
                    print('Checkpoint loaded')

    def train(self, test_dataset_3dpw, test_dataset_3dpw_crop, test_dataset_h36m):
        """Training process."""
       
        # test_dataset_3dpw = BaseDataset(None, '3dpw', is_train=False,bMiniTest =True)

        exemplar_prevTarget =-1
        #Start with test  #########
        if self.options.noEval ==False:       #Debug
            best_error_3dpw = self.test(test_dataset_3dpw, '3dpw')
            self.summary_writer.add_scalar('3dpw_test_err_mm', best_error_3dpw, self.step_count)

            cur_error_3dpw_crop = self.test(test_dataset_3dpw_crop, '3dpw-crop')
            self.summary_writer.add_scalar('3dpw_crop_test_err_mm', cur_error_3dpw_crop, self.step_count)

            best_error_h36m = self.test(test_dataset_h36m, 'h36m-p1')
            self.summary_writer.add_scalar('h36m-p1_test_err_mm', best_error_h36m, self.step_count)

        epoch = self.epoch_count
        for epoch in tqdm(range(self.epoch_count, self.options.num_epochs), total=self.options.num_epochs, initial=self.epoch_count):

            print(">>> Epoch!")

            # Run training for num_epochs epochs
            # Create new DataLoader every epoch and (possibly) resume from an arbitrary step inside an epoch
            train_data_loader = CheckpointDataLoader(self.train_ds,checkpoint=self.checkpoint,
                                                    batch_size=self.options.batch_size,
                                                    num_workers=self.options.num_workers,
                                                    pin_memory=self.options.pin_memory,
                                                    shuffle=self.options.shuffle_train)


            # Iterate over all batches in an epoch
            for step, batch in enumerate(tqdm(train_data_loader, desc='Epoch '+str(epoch),
                                              total=len(self.train_ds) // self.options.batch_size,
                                              initial=train_data_loader.checkpoint_batch_idx),
                                         train_data_loader.checkpoint_batch_idx):

                g_bExemplarMode = True
                if g_bExemplarMode:
                    if exemplar_prevTarget != self.options.exemplar_targetIdx:    #This looks a newtarget. Reload 
                        print(">> New sample detected: prevID: {} -> curId:{}".format(exemplar_prevTarget, self.options.exemplar_targetIdx))
                        self.reloadModel()
                        exemplar_prevTarget = self.options.exemplar_targetIdx

                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
                out = self.train_step(batch)
                self.step_count += 1

                # # Tensorboard logging every summary_steps steps
                if self.step_count % self.options.summary_steps == 0:
                    self.train_summaries(batch, *out)


            if (epoch+1) % self.options.test_epoch_inter == 0:
                cur_error_3dpw = self.test(test_dataset_3dpw, '3dpw')
                self.summary_writer.add_scalar('3dpw_test_err_mm', cur_error_3dpw, self.step_count)

                cur_error_3dpw_crop = self.test(test_dataset_3dpw_crop, '3dpw-crop')
                self.summary_writer.add_scalar('3dpw_crop_test_err_mm', cur_error_3dpw_crop, self.step_count)

                cur_error_h36m = self.test(test_dataset_h36m, 'h36m-p1')
                self.summary_writer.add_scalar('h36m-p1_test_err_mm', cur_error_h36m, self.step_count)

                if best_error_3dpw>cur_error_3dpw: #or >cur_error_h36m:
                    print(">>> Great! Found a new best model. Save this!")
                    self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch+1, 0, self.options.batch_size, None, self.step_count, suffix='best-{}'.format(cur_error_3dpw))
                    best_error_3dpw = cur_error_3dpw #New best



            # load a checkpoint only on startup, for the next epochs
            # just iterate over the dataset as usual
            self.checkpoint=None
            # save checkpoint after each epoch
            if (epoch+1) % self.options.save_epoch_inter == 0:
                # self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch+1, 0, self.step_count)
                self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch+1, 0, self.options.batch_size, None, self.step_count)

        ########### Finalize #############
        tqdm.write('Done')  
        self.finalize()
        self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch+1, step, self.options.batch_size, train_data_loader.sampler.dataset_perm, self.step_count) 
        tqdm.write('Checkpoint saved')
        sys.exit(0)
        return
    

    def finalize(self):
        pass

    # The following methods (with the possible exception of test) have to be implemented in the derived classes
    def init_fn(self):
        raise NotImplementedError('You need to provide an _init_fn method')

    def train_step(self, input_batch):
        raise NotImplementedError('You need to provide a _train_step method')

    def train_summaries(self, input_batch):
        raise NotImplementedError('You need to provide a _train_summaries method')

    def test(self, dataset_test, datasetName):
        return -1

    #    #Code for exemplar tuning
    # def backupModel(self):
    #     self.model_backup = self.model.state_dict()
    #     self.optimizer_backup = self.optimizer.state_dict()

    # def reloadModel(self):
    #     self.model.load_state_dict(self.model_backup)
    #     self.optimizer.load_state_dict(self.optimizer_backup)

      #Code for exemplar tuning
    def backupModel(self):
        
        print(">>> Model status saved!")
        self.model_backup = copy.deepcopy(self.model.state_dict())
        self.optimizer_backup = copy.deepcopy(self.optimizer.state_dict())

    def reloadModel(self):
        print(">>> Model status has been reloaded to initial!")
        # print(sum(sum(self.model.state_dict()['layer3.5.conv3.weight'])))
        # print("checking: {}".format(sum(sum(self.model_backup['layer3.5.conv3.weight']))))
        self.model.load_state_dict(self.model_backup)
        self.optimizer.load_state_dict(self.optimizer_backup)


    def exemplerTrainingMode():
        assert False

    # def test(self):
    #     pass
    
