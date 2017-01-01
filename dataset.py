import cv2
import numpy as np
from glob import glob


class Dataset:
    def __init__(self, path, img_height, img_width, batch_size, n_pairs, do_shuffle=True):
        if path[-1] == '/':
            path = path[:-1]
        self._root_path = path
        self._pano_path = '%s/pano_resize'%(self._root_path)
        self._depth_path = '%s/depth_origin'%(self._root_path)
        self._height = img_height
        self._width = img_width

        #self._n_epochs = n_epochs
        self._batch_index = 0
        self._batch_size = batch_size
        self._n_pairs = n_pairs
        self._do_shuffle = do_shuffle

        self._pano_lst = self.GetPanoLst()
        self._depth_lst = self.GetDepthLst()
        self._CheckLength()
        #self._CheckLst()
        self._total_data = len(self._depth_lst)
        self._data_index = np.arange(self._total_data)
    
    def GetNextBatch(self):
        index = self._batch_index
        batch_size = self._batch_size
        start = index * batch_size
        end = (index+1) * batch_size
        self._batch_index += 1
        if start > self._total_data - 1:
            self._batch_index = 0
            self.Shuffle()
            return self._GetFirstBatch()
        elif end > self._total_data:
            count = (self._total_data - start)
            B = np.random.choice(self._data_index, batch_size-count, replace=False)
            A = np.arange(start, start+count)
            indice = np.concatenate([A, B])
            return self._pano_lst[indice], self._depth_lst[indice]
        else:
            return self._pano_lst[start:end], self._depth_lst[start:end]

    def GetPanoLst(self):
        lst = glob('%s/*.jpg'%self._pano_path)
        return np.array(sorted(lst))

    def GetDepthLst(self):
        lst = glob('%s/*.npy'%self._depth_path)
        return np.array(sorted(lst))
    
    def Shuffle(self):
        np.random.shuffle(self._data_index)
        self._pano_lst = self._pano_lst[self._data_index]
        self._depth_lst = self._depth_lst[self._data_index]
        #print 'Shuffle !!!!!!!!!!!!!!!!!!'
    
    def _ReadImageData(self, lst):
        batch_size = self._batch_size
        height = self._height
        width = self._width
        buf = np.zeros([batch_size, height, width, 3], dtype=np.float32)
        for index, path in enumerate(lst):
            img = np.float32(cv2.imread(path)) / 255
            buf[index, :, :, :] = img
        return buf

    def _GetRelationPairs(self, lst):
        batch_size = self._batch_size
        height = self._height
        width = self._width
        n_pairs = self._n_pairs
        buf = np.zeros([batch_size, n_pairs, 5], dtype=np.int32)
        for index, path in enumerate(lst):
            depth_map = np.load(path)
            #tmp = np.zeros([n_pairs, 5], dtype=np.int32)
            row1_indice = np.random.choice(range(height), n_pairs)
            col1_indice = np.random.choice(range(width), n_pairs)

            row2_indice = np.random.choice(range(height), n_pairs)
            col2_indice = np.random.choice(range(width), n_pairs)
            
            depth1 = depth_map[row1_indice, col1_indice]
            depth2 = depth_map[row2_indice, col2_indice]
            relation = np.zeros([n_pairs, 1], dtype=np.int32)
            relation[depth1 > depth2] = 1
            relation[depth1 < depth2] = -1
            buf[index, :, :] = np.concatenate([row1_indice.T, col1_indice.T, row2_indice.T, 
                                               col2_indice.T, relation], axis=1)

            

    def _GetFirstBatch(self):
        index = self._batch_index
        batch_size = self._batch_size
        start = index * batch_size
        end = (index+1) * batch_size
        #print start,end
        self._batch_index += 1
        #print index
        #print self._pano_lst[0:10]
        #print self._depth_lst[0:10]
        return self._pano_lst[start:end], self._depth_lst[start:end]

    def _CheckLst(self):
        if not (len(self._depth_lst) == len(self._pano_lst)):
            print 'Dataset error'
            exit()

        for index in range(len(self._depth_lst)):
            pano_id = self._pano_lst[index].split('/')[-1][5:-4]
            depth_id = self._depth_lst[index].split('/')[-1][5:-4]
            if not pano_id == depth_id:
                print 'Dataset error',pano_id, depth_id
                exit()
            #exit()

    def _CheckLength(self):
        if not (len(self._depth_lst) == len(self._pano_lst)):
            print 'Dataset error'
            exit()

if __name__ == '__main__':
    test = Dataset('../data', 2000, 0, 0)
    for i in range(50):
        if i==0:
            test._GetFirstBatch()
        a = test.GetNextBatch()
        #print len(a[0]), len(a[1])
