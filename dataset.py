import cv2
import numpy as np
from glob import glob


class Dataset:
    def __init__(self, path, img_height, img_width, batch_size, n_pairs, do_shuffle=False):
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
        self.Shuffle()
    
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
            return self._ReadImageData(self._pano_lst[indice]), self._GetRelationPairs(self._depth_lst[indice])
        else:
            return self._ReadImageData(self._pano_lst[start:end]), self._GetRelationPairs(self._depth_lst[start:end])

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
        print 'Shuffle !!!!!!!!!!!!!!!!!!'
    
    def _ReadImageData(self, lst):
        batch_size = self._batch_size
        height = self._height
        width = self._width
        buf = np.zeros([batch_size, height, width, 3], dtype=np.float32)
        for index, path in enumerate(lst):
            img = np.interp(np.float32(cv2.imread(path)), [0, 255], [-1, 1])
            buf[index, :, :, :] = img
        return buf

    def _GetRelationPairs(self, lst, depth_thresh = 2):
        batch_size = self._batch_size
        height = self._height
        width = self._width
        indice_buf = np.arange(height * width, dtype=np.int32)
        n_pairs = self._n_pairs
        buf = np.zeros([batch_size, n_pairs, 5], dtype=np.int32)
        for index, path in enumerate(lst):
            depth_map = np.load(path)
            non_inf_indice = np.logical_not(np.isinf(depth_map.reshape(height * width)))
            indice_non_inf = indice_buf[non_inf_indice]
            
            indice1 = np.random.choice(indice_non_inf, n_pairs)
            row1_indice = np.int32(indice1 / width)
            col1_indice = indice1 - row1_indice * width 
            
            indice2 = np.random.choice(indice_non_inf, n_pairs)
            row2_indice = np.int32(indice2 / width)
            col2_indice = indice2 - row2_indice * width
                        
            depth1 = depth_map[row1_indice, col1_indice]
            depth2 = depth_map[row2_indice, col2_indice]
            relation = np.zeros([n_pairs], dtype=np.int32)
            hard_to_tell_indice = np.abs(depth1 - depth2) <= depth_thresh
            relation[depth1 > depth2] = 1
            relation[depth1 < depth2] = -1
            relation[hard_to_tell_indice] = 0
            #print row2_indice
            #print col2_indice
            #print relation
            #print np.concatenate([[row1_indice], [col1_indice], [row2_indice,col2_indice], [relation]]).T
            buf[index, :, :] = np.concatenate([[row1_indice], [col1_indice], [row2_indice],
                                               [col2_indice], [relation]]).T
        return buf 

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
        return self._ReadImageData(self._pano_lst[start:end]), self._GetRelationPairs(self._depth_lst[start:end])

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
    test = Dataset('../data', 256, 512 , 100, 3000, 0)
    a = test._GetFirstBatch()
    print a[1][0, :, :]
    #print a[0][0, 0, 0, :]
    exit()
    test._batch_index = 295
    for i in range(50):
        print i
        #if i==0:
        #    test._GetFirstBatch()
        a = test.GetNextBatch()
        #print len(a[0]), len(a[1])
