from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import numpy as np
import cv2
from scipy import misc
import os
import scipy.io as sio
import h5py
import copy

PI = math.pi

class NYU_input():
    def __init__(self, dataset, n_epochs, batch_size, n_pairs, r_range=[13,19], depth_thresh=1, do_shuffle=True):
        '''
        Arguments:
            dataset: a string (can be changed) specifies root directory of dataset
            n_epochs: number of epochs
            batch_size: size of a minibatch
            n_pairs: number of random sampled point pairs per image 
            r_range: a parameter of sample_point_pairs function
            depth_thresh: a parameter of sample_point_pairs function
            do_shuffle: a boolean to determine whether dataset shuffle performed as an epoch ends
        '''
        self._n_epochs = n_epochs
        self._batch_size = batch_size
        self._dataset = dataset
        self._n_pairs = n_pairs
        self._r_range = r_range
        self._depth_thresh = depth_thresh
        # total number of samples in given dataset
        self._n_samples = []
        # the index of first image in current batch
        self._read_head = 0
        # extra tape for indexing, used for shuffled index, 
        # access an image --> all_images_in_dataset[self._index[self._read_head]] 
        self._index = []
        self._image_size = []
        self._do_shuffle = do_shuffle
        self.data=self._parse_dataset()
        
        #self.data_len = 0

    def shuffle(self):
        '''
        FUNC: random shuffle entire dataset, only can be used after entire epoch completed
        '''
        if self._read_head==0:
            self._index = np.random.permutation(self._n_samples)
        else:
            raise ValueError('Have not complete an entire epoch yet, cannot shuffle dataset')

    def next_batch(self):
        '''
        Returns:
            image_batch: RGB images, ndarray, shape=[batch_size,img_H,img_W,3]
            pair_batch: sampled point pairs, ndarray, shape=[batch_size,n_pairs,5],
                        each image has n_pairs(may be different for different 
                        images), 5 is (y1,x1,y2,x2,relations)   
        '''
        data = self.data
        print("parse dataset succeed")
        image_data = data['image'].copy()
        depth_data = data['depth'].copy()
        
        if self._read_head ==0:
            self.shuffle()
            image_batch = image_data[self._index[self._read_head:self._read_head+self.batch_size]].copy()
            depth_batch = depth_data[self._index[self._read_head:self._read_head+self.batch_size]].copy()
            pair_batch = copy.deepcopy(self._get_pair_batch(depth_batch))
            #self._read_head = self._read_head + self.batch_size
        else:
            image_batch = image_data[self._index[self._read_head:self._read_head+self.batch_size]].copy()
            depth_batch = depth_data[self._index[self._read_head:self._read_head+self.batch_size]].copy()
            pair_batch = copy.deepcopy(self._get_pair_batch(depth_batch))
        
        #print((depth_batch.dtype))
        #print((pair_batch.dtype))
        image_batch = np.transpose(image_batch,(0,3,2,1))
        image_batch = image_batch.astype(np.float32)/127.5 -1
        pair_batch = pair_batch.astype(np.int32)
        #print((pair_batch.dtype))
        return image_batch, pair_batch
        '''
        # obtain current image batch
        '''
    def _get_pair_batch(self, current_image_batch):
        pair_batch = []
        for idx, image in enumerate(current_image_batch):
            pairs = sample_point_pairs(image, self._n_pairs, self._r_range, self._depth_thresh)
            pair_batch.append(pairs)
        pair_batch = np.array(pair_batch)
    
        # update read head
        self._read_head = self._read_head + self._batch_size
        # precompute the next read head, to check if an epoch completes(the last possible batch) after current batch ends
        # now, self._read_head is the read head of next batch, and we check the read head of the batch after next batch
        if self._read_head+self._batch_size > self._n_samples:
            # an epoch completed, reset read head to 0
            self._read_head = 0
            if self._do_shuffle:
                self.shuffle()
        
        return pair_batch

    def _parse_dataset(self):
        '''
        FUNC: parse datapath of given dataset root directory --> do you need this?
        '''
        mat = h5py.File(self._dataset)
        depth_data = copy.copy(np.array(mat['depths']))
        image_data = copy.copy(np.array(mat['images']))
        assert depth_data.shape[0]==image_data.shape[0]
        print("Dataset loaded")
        self._image_size = [image_data.shape[2],image_data.shape[3]]
        
        self._n_samples = depth_data.shape[0]
        self._index = np.arange(self._n_samples)
        print("Data prepared!!")
        return {'depth':depth_data, 'image':image_data, 'image_length':image_data.shape[0], 'depth_length': depth_data.shape[0]}

        # operation depending on self._dataset
        #self._parsed_dataset = ??? # may be a list of string with each string as a full path of one sample(image)
        
        #raise NotImplementedError

    # may not need edit-property function --> ??

    # class property, protected
    @property
    def batch_size(self):
        return self._batch_size
    @property
    def n_pairs(self):
        return self._n_pairs
    @property
    def read_head(self):
        return self._read_head
    @property
    def r_range(self):
        return self._r_range
    @property
    def num_samples(self):
        return self._num_samples
    def __str__(self):
        return 'Hi, I am an instance of NYU_input <3'

# WARNING!!!
# have not been able to handle shattered depth map
def sample_point_pairs(depth_img, n_pairs=None, r_range=[13,19], depth_thresh=1):
    '''
    FUNC: random sample point pairs over given depth image
    Arguments:
        depth_img: a numpy array with shape (*,*)
        n_pairs: number of point pairs to be sampled
        r_range: a 2-element list specifying the range of distance between
                 2 points in a pair, where the 1st element is lower bound
                 and the 2nd element is upper bound
        depth_thresh: if absolute depth difference between 2 sampled points 
                      in a pair, relation is 0(hard to tell)
    Returns:
        pairs: n_pairs rows, for each row [y1, x1, y2, x2, relation], relation
               can be +1(point1>point2), -1(point1<point2), or 0(hard to tell)
    '''
    try:
        img_h, img_w = depth_img.shape
    except:
        raise ValueError('input image of function sample_point_pairs must have shape (*,*)')
    if not n_pairs:
        n_pairs = math.ceil(img_h*img_w / 50**2)
    # sample 1st points
    y1 = np.random.randint(r_range[1], img_h-r_range[1], (n_pairs,1))
    x1 = np.random.randint(r_range[1], img_w-r_range[1], (n_pairs,1))
    pt1 = np.concatenate([y1,x1], axis=1)
    # sample 2nd points
    r = np.random.randint(r_range[0], r_range[1], (n_pairs,1))
    ang = np.random.uniform(-PI, PI, (n_pairs,1))
    y2 = y1 + r*np.sin(ang)
    x2 = x1 + r*np.cos(ang)
    pt2 = np.concatenate([y2,x2], axis=1)
    pt2 = np.around(pt2).astype(np.int)
    # obtain ordinal relations
    pt1_val = copy.deepcopy(depth_img[pt1[:,0],pt1[:,1]])
    pt2_val = copy.deepcopy(depth_img[pt2[:,0],pt2[:,1]])
    larger = pt1_val > pt2_val # +1
    relations = np.zeros((n_pairs,), dtype=np.int) + larger - np.invert(larger)
    margin = np.absolute(pt1_val-pt2_val) < depth_thresh # 0
    relations = relations * np.invert(margin) # can be +2, 0, or -1
    relations = np.reshape(relations, (n_pairs,1))
    # form pairs
    pairs = np.concatenate([pt1,pt2,relations], axis=1)

    return pairs

def visualize_point_pairs(img, pairs):
    '''
    FUNC: draw point pairs on the image, with each pair connected by a line,
          to test the validatiy of sample_point_pairs function
    '''
    line_color_pos = (255,0,0) # blue
    line_color_neg = (0,0,255) # red
    line_color_0 = (0,255,0) # green
    line_thickness = 1
    # convert image to 3 channel to draw line with different colors
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # draw line
    for i in xrange(pairs.shape[0]):
        y1, x1 = pairs[i,0:2]
        y2, x2 = pairs[i,2:4]
        relation = pairs[i,4]
        if relation == 1:
            color = line_color_pos
        elif relation == -1:
            color = line_color_neg
        else:
            color = line_color_0
        cv2.line(img,(x1,y1),(x2,y2),color,line_thickness)  
    # show image
    cv2.imshow('visualize point pairs', img)
    cv2.waitKey(0)
    cv2.destroyWindow('visualize point pairs')

    return img
