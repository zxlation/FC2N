# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 16:06:56 2018

@author: zxlation
"""


import numpy as np
import threading
import os
import imageio
import time
import options
from queue import Queue
from pathlib import Path

batch_size = options.params.batch_size
patch_size = options.params.patch_size
channel    = options.params.channel
scale      = options.params.scale

class data_loader:
    """
    A basic data loader for loading data for model training and testing. Note that 
    some hyper-parameters are defined in option file.
    """
    def __init__(self, n_threads = 1, q_capacity = 16, **kwargs):
        self.__name__     = 'loader'
        self.__dtype__    = np.uint8
        self.n_threads    = n_threads
        self.q_capacity   = q_capacity
        self.threads      = []
        self.work_queue   = Queue(self.q_capacity)
        self.augmentation = True
        self.batch_size   = batch_size
        self.patch_size   = patch_size
        self.channel      = channel
        self.scale        = scale
        self.data_path    = './datasets/'
        self.data_mean    = [114.444 , 111.4605, 103.02]
        self.data_norm    = 255.0
        self.data_range   = 255.0
        self.num_train_images = 800
    
    
    def enqueue_loop(self):
        print("%s: %s starting..." % (self.__name__, threading.current_thread().name))
        while True:
            batch = self.one_shot_batch()
            self.work_queue.put(batch)
    
    
    def load_batch(self):
        for i in range(self.n_threads):
            t = threading.Thread(target = self.enqueue_loop)
            t.daemon = True
            t.start()
            self.threads.append(t)
            time.sleep(0.001)
        
        return self.work_queue
        
        
    def load_train_dataset(self):
        """
        We load all training data into memory for quick data preparation. For machines
        with more than 6GB memory, this should not be problematic. We use the DIV2K for 
        model training, you can specify you own training dataset by changing the lDir 
        and hDir paths below.
        """
        scale = self.scale
        
        if scale == 8:
            lDir  = os.path.join(self.data_path, 'DIV2K/DIV2K_train_LR_bicubic/X8/LR/')
            hDir  = os.path.join(self.data_path, 'DIV2K/DIV2K_train_LR_bicubic/X8/HR/')
        else:
            hDir  = os.path.join(self.data_path, 'DIV2K/DIV2K_train_HR')
            lDir  = os.path.join(self.data_path, 'DIV2K/DIV2K_train_LR_bicubic/X%d' % scale)
        
        lr_names = sorted(os.listdir(lDir))
        hr_names = sorted(os.listdir(hDir))
        num_images = len(hr_names)
        
        self.HR_images = []
        self.LR_images = []
        for im_idx in range(num_images):
            imLR = imageio.imread(os.path.join(lDir, lr_names[im_idx]))
            imHR = imageio.imread(os.path.join(hDir, hr_names[im_idx]))
            self.LR_images.append(imLR)
            self.HR_images.append(imHR)
            if (im_idx + 1) % 100 == 0:
                print('%d/%d loaded.' % (im_idx + 1, num_images))
        
        self.num_train_images = num_images
        self.scale = scale
    
    
    def load_valid_dataset(self):
        """ Load valid dataset for model validation. Normally we use Set5 for validation.
        You can specify you own valid dataset by set different lDir and hDir below.
        """
        scale = self.scale
        self.valid_dataset = []
        lDir = os.path.join(self.data_path, 'Set5/image_SRF_%d' % (scale), 'LR')
        hDir = os.path.join(self.data_path, 'Set5/image_SRF_%d' % (scale), 'HR')
        lr_names = sorted(os.listdir(lDir))
        hr_names = sorted(os.listdir(hDir))
        num_image = len(lr_names)
        assert(num_image == len(hr_names))
            
        for i in range(num_image):
            lim = imageio.imread(os.path.join(lDir, lr_names[i]))
            him = imageio.imread(os.path.join(hDir, hr_names[i]))
            self.valid_dataset.append((lim, him, hr_names[i]))
                
        self.num_valid_images = num_image
    
    
    def load_test_datasets(self, datasets = ['Set5'], scales = [4]):
        """ This data loader is also designed for model testing. In this case, all test
        datasets will be loaded into memory by specifying their names as a list.
        """
        self.test_datasets   = {}
        self.num_test_images = {}
        for dataset in datasets:
            print('Loading testing dataset %s...' % dataset)
            self.test_datasets[dataset] = {}
            for idx_scale, scale in enumerate(scales):
                print('\tscale X%d...' % scale)
                scaleKey = 'X%d' % scale
                self.test_datasets[dataset][scaleKey] = []
                
                lDir = os.path.join(self.data_path, '%s/image_SRF_%d' % (dataset, scale), 'LR')
                hDir = os.path.join(self.data_path, '%s/image_SRF_%d' % (dataset, scale), 'HR')
                
                lr_names = sorted(os.listdir(lDir))
                hr_names = sorted(os.listdir(hDir))
                num_image = len(lr_names)
                assert(num_image == len(hr_names))
            
                for i in range(num_image):
                    lim = imageio.imread(os.path.join(str(Path(lDir)), lr_names[i]))
                    him = imageio.imread(os.path.join(str(Path(hDir)), hr_names[i]))
                    
                    self.test_datasets[dataset][scaleKey].append((lim, him, hr_names[i]))
                
            self.num_test_images[dataset] = num_image
    
    
    def one_shot_batch(self):
        """ Generate a LR/HR patch pair for batching, according to
            specific configuration in the initializer.
        """        
        bSize = self.batch_size
        pSize = self.patch_size
        nChan = self.channel
        scale = self.scale
        
        lSize, hSize = pSize, pSize*scale
        
        # select a group of indexes for a batch image patches.
        im_idxs = np.random.randint(low = 0, high = self.num_train_images, size = bSize)
        
        batchL = np.zeros([bSize, lSize, lSize, nChan], np.float32)
        batchH = np.zeros([bSize, hSize, hSize, nChan], np.float32)
        for i in range(bSize):
            imL = self.LR_images[im_idxs[i]]
            imH = self.HR_images[im_idxs[i]]
            
            [lH, lW, lC] = imL.shape
            lh = np.random.randint(low = 0, high = lH - pSize + 1)
            lw = np.random.randint(low = 0, high = lW - pSize + 1)
            hh, hw = lh*scale, lw*scale
            
            patchL = imL[lh:lh + lSize, lw:lw + lSize, :]
            patchH = imH[hh:hh + hSize, hw:hw + hSize, :]
            
            if self.augmentation:
                patchL, patchH = self.data_augment(patchL, patchH)
            
            # normalization
            batchL[i, ...] = patchL
            batchH[i, ...] = patchH
        
        return (batchL, batchH, scale)
    
    
    def data_augment(self, patchL, patchH):
        """ data augmentaton: including left/right and up/down fliping, and transpose.
        Args:
            patchL: A numpy array representing a LR image patch.
            patchH: A numpy array representing the corresponding HR image patch.
        Returns:
            The transformed image patches.
        """
        # data range complementation
        rng = np.random.randint(low = 0, high = 2)
        if rng == 1:
            patchL = self.data_range - patchL
            patchH = self.data_range - patchH
        
        # geometric self-ensemble
        rot = np.random.randint(low = 2, high = 9)
        if rot == 2:
            patchL = patchL[::-1, :, :]
            patchH = patchH[::-1, :, :]
        elif rot == 3:
            patchL = patchL[:, ::-1, :]
            patchH = patchH[:, ::-1, :]
        elif rot == 4:
            patchL = patchL[::-1, ::-1, :]
            patchH = patchH[::-1, ::-1, :]
        elif rot == 5:
            patchL = np.transpose(patchL, [1, 0, 2])
            patchH = np.transpose(patchH, [1, 0, 2])
        elif rot == 6:
            patchL = np.transpose(patchL, [1, 0, 2])[::-1, :, :]
            patchH = np.transpose(patchH, [1, 0, 2])[::-1, :, :]
        elif rot == 7:
            patchL = np.transpose(patchL, [1, 0, 2])[:, ::-1, :]
            patchH = np.transpose(patchH, [1, 0, 2])[:, ::-1, :]
        elif rot == 8:
            patchL = np.transpose(patchL, [1, 0, 2])[::-1, ::-1, :]
            patchH = np.transpose(patchH, [1, 0, 2])[::-1, ::-1, :]
        
        return patchL, patchH



"""============================================================================
                                UNIT TEST
============================================================================"""
def unit_train_loader():
    import matplotlib.pyplot as plt
    
    loader = data_loader(n_threads = 4, q_capacity = 16)
    loader.load_train_dataset()
    
    workQ = loader.load_batch()
    for i in range(10):
        [batchL, batchH, scale] = workQ.get()
        print(batchH.dtype, np.max(batchH), np.min(batchH))
        plt.imshow(batchL[0, ...].astype(np.uint8)), plt.title('LR patch'), plt.show()
        plt.imshow(batchH[0, ...].astype(np.uint8)), plt.title('HR patch'), plt.show()
    

def unit_test_loader():
    loader = data_loader('test')
    loader.load_test_datasets(['Set5'], scales = [2, 3, 4])
    
    num_test_sets = len(loader.test_datasets)
    num_scls_set5 = len(loader.test_datasets['Set5'])
    num_imgs_set5 = len(loader.test_datasets['Set5']['X2'])
    print('%d testing datasets are loaded.' % num_test_sets)
    print('%d scales of Set5 has be loaded.' % num_scls_set5)
    print('Set5 x2 has %d images.' % num_imgs_set5)
    

def unit_valid_loader():
    import matplotlib.pyplot as plt
    
    loader = data_loader('valid')
    loader.load_valid_dataset()
    print(loader.num_valid_images)
    
    idx = 0
    lIm, hIm, im_name = loader.valid_dataset[idx]
    lIm = lIm.astype(np.uint8)
    hIm = hIm.astype(np.uint8)
    
    plt.imshow(lIm)
    plt.title('LR -- ' + im_name), plt.show()
    
    plt.imshow(hIm)
    plt.title('HR -- ' + im_name), plt.show()


if __name__ == '__main__':
    unit_valid_loader()