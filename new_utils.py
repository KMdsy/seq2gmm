import os, pickle, pwlf
import numpy as np
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import KMeans


class SplitData:
    def __init__(self):
        self.layer = 0
        self.split_num = None # number of segments
        self.segment_idx = 0 
        self.prev_split_num = None
        self.prev_segment_idx = 0
        self.normal_cluster = None
        self.flag = ''

    def _k_means(self, data, K):
        if K == 1:
            score = -np.inf
        else:
            max_len = max([len(item) for item in data])
            sample_num = len(data) // K
            tmp_data = np.zeros([sample_num, max_len])
            for sample_idx in range(sample_num):
                for p_idx in range(len(data[sample_idx])):
                    tmp_data[sample_idx][p_idx] = data[sample_idx][p_idx]
            model = KMeans(n_clusters=K, random_state=2021).fit(tmp_data)
            # a higher Calinski-Harabasz score relates to a model with better defined clusters.
            score = calinski_harabasz_score(tmp_data, model.labels_)
        return score


    def _split_data(self, raw_data, split):
        data_dict = {}
        for i in range(split):
            data_dict['{}'.format(i)] = []

        if split == 1:
            list_data = []
            for row_idx in range(len(raw_data)):
                list_data.append(raw_data[row_idx])
            data_dict['0'] = list_data
            return data_dict

        line_index = 0
        for line in raw_data:
            line_index += 1
            print('split: {}'.format(line_index))
            x = range(len(line))
            my_pwlf = pwlf.PiecewiseLinFit(x, line)
            breaks = my_pwlf.fit(split)

            for i in range(len(breaks)-1):
                data_dict['{}'.format(i)].append(line[int(breaks[i]):int(breaks[i+1])+1])

        return data_dict


    def _get_split_data(self, data, layer, segment_idx, _type, split):
        if os.path.exists(os.path.join(self.dataset_dir, 'others')) is False: 
            os.makedirs(os.path.join(self.dataset_dir, 'others'))
        if layer == 0:
            search_dir = self.dataset_dir
            save_name = os.path.join(self.dataset_dir, 'others', f'{self.flag}{_type}_layer{layer},split{segment_idx}-thSeg,to{split}.pkl')
        else:
            search_dir = os.path.join(self.dataset_dir, 'others')
            save_name = os.path.join(search_dir, f'{self.flag}{_type}_layer{layer},split{segment_idx}-thSeg,to{split}.pkl')
            

        if os.path.exists(save_name) == True:
            f = open(save_name, 'rb')
            data_dict = pickle.load(f)
            f.close()
        else:
            data_dict = self._split_data(data, split)
            f = open(save_name, 'wb')
            pickle.dump(data_dict, f)
            f.close()

        data = []
        sample_num = len(data_dict['0'])
        for sample_idx in range(sample_num):
            for seg_idx in range(split):
                data.append(data_dict[f'{seg_idx}'][sample_idx])
        return data

    def _get_opti_split_num(self, data, layer, segment_idx, _type, max_loop=5, split_num=None):
        if split_num is None:
            loop_idx = 1
            best_score = -np.inf
            is_convergence = False
            while loop_idx <= max_loop:
                this_data = self._get_split_data(data, layer, segment_idx, _type, loop_idx)
                this_score = self._k_means(this_data, loop_idx)
                if this_score >= best_score:
                    best_score = this_score
                    best_split_num = loop_idx
                    best_splited_data = this_data
                else:
                    is_convergence = True
                    break
                print(f'score of {loop_idx}: {this_score}')
                print(f'best - score of: {best_split_num}, {best_score}\n')
                loop_idx += 1
            if is_convergence is False:
                print('best_split_num={}, not convergence but reach the max loop limit.'.format(best_split_num))
            else:
                print('converged\n')
            frag_nums = [best_split_num] * len(data)
            self.split_num = best_split_num
        else:
            best_splited_data = self._get_split_data(data, layer, segment_idx, _type, split_num)
            best_split_num = split_num
            frag_nums = [best_split_num] * len(data)
        return best_split_num, best_splited_data, frag_nums

    def _get_unsplited_data(self, layer, segment_idx, _type):
        if layer == 0:
            if _type == 'train':
                data = np.loadtxt(os.path.join(self.dataset_dir, self.dataset_name+'_TRAIN.tsv'))
            elif _type == 'test':
                data = np.loadtxt(os.path.join(self.dataset_dir, self.dataset_name+'_TEST.tsv'))
            elif _type == 'v':
                data = np.loadtxt(os.path.join(self.dataset_dir, self.dataset_name+'_TEST.tsv'))
            if self.normal_cluster is None:
                cluster = np.unique(data[:,0])
                max_num = 0
                self.normal_cluster = -1
                for c in cluster:
                    t = data[np.where(data[:,0]==c)]
                    if t.shape[0] > max_num:
                        max_num = t.shape[0]
                        self.normal_cluster = c
            print('\n\n', _type, '\nnormal cluster:', self.normal_cluster)
            if _type == 'train':
                self.train_label = data[np.where(data[:,0]==self.normal_cluster)][:,0]
                data = data[np.where(data[:,0]==self.normal_cluster)][:,1:]
                print('train sample num:', len(data), '\n')
                return data, self.train_label
            else:
                normal_data_idx= np.where(data[:,0]==self.normal_cluster)[0]
                abnormal_data_idx= np.where(data[:,0]!=self.normal_cluster)[0]
                np.random.seed(1)
                np.random.shuffle(abnormal_data_idx)
                abnormal_data_idx = abnormal_data_idx[0:int(len(normal_data_idx)*0.1)]
                data_idx = np.sort(np.concatenate((normal_data_idx, abnormal_data_idx),axis=0))

                label = data[data_idx,0].flatten()
                label[label==self.normal_cluster] = -2 # normal label is '-2'
                self.test_label = label
                data = data[data_idx,1:]
                print('test sample num:', len(data))
                print('normal: {}, abnormal: {}'.format(len(label[label==-2]), len(label[label!=-2])))
                return data, self.test_label
        else:
            file_name = os.path.join(self.dataset_dir, 'others', f'{self.flag}{_type}_layer{self.layer-1},split{self.prev_segment_idx}-thSeg,to{self.prev_split_num}.pkl')
            f = open(os.path.join(file_name), 'rb')
            all_data = pickle.load(f)
            f.close()
            data = all_data[f'{segment_idx}']
            if _type == 'train':
                return data, self.train_label
            elif _type == 'test':
                return data, self.test_label        

    def read_data(self, dataset_dir, dataset, split_num=None):
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_dir.split(os.sep)[-1]
        data, label = self._get_unsplited_data(self.layer, self.segment_idx, dataset)
        # split data
        split_num, splited_data, frag_nums = self._get_opti_split_num(data, self.layer, self.segment_idx, dataset, split_num=split_num)
        return splited_data, label, frag_nums, split_num