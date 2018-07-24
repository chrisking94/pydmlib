from  dataprocessor import *


class Sampler(RSDataProcessor):
    def __init__(self, features2process, name=''):
        RSDataProcessor.__init__(self, features2process, name, 'blue', 'yellow', 'highlight')

    def _sample(self, data, features, label):
        self.error('Not implemented!')

    def _process(self, data, features, label):
        spcount0 = data.shape[0]
        data = self._sample(data, features, label)
        self.msg('%d ==> %d' % (spcount0, data.shape[0]), 'sample quantity')
        return data


class SplUnder(Sampler):
    def __init__(self, features2process, feature_weights=None):
        """
        对于某一行数据，根据feature_weights中的权重来计算其重要性（见:param feature_weights）
        采样时首选重要性高的数据
        :param feature_weights:
            features = {f1,f2,...}
            如果feature_weights is dict(f1:w1,f2:w2,...)，则按importancei=∑(wi*(if <fi> not null then 1 else 0))
                计算后首选importance高的数据
            如果feature_weights==1,则首选空值少的某行
            如果feature_weights==None，则进行随机抽样
        """
        Sampler.__init__(self, features2process)
        self.feature_weights = feature_weights

    def _sample(self, data, features, label):
        feature_weights = self.feature_weights
        lables, cnts = np.unique(data[label], return_counts=True)
        mc_samplecount = cnts.min()
        minorclass = lables[cnts == mc_samplecount][0]
        lables, cnts = lables[lables != minorclass], cnts[lables != minorclass]
        newdata = data[data[label] == minorclass]
        for lbl, cnt in zip(lables, cnts):
            dtmp = data[data[label] == lbl]
            if feature_weights == None:
                newdata = newdata.append(dtmp.sample(mc_samplecount))
            elif feature_weights == 1:
                self.error('Not implemented!')
            elif isinstance(feature_weights, dict):
                self.error('Not implemented!')
            else:
                self.error('Invalid value for feature_weights!')
        return newdata


class SplMiddle(Sampler):
    def __init__(self, features2process):
        Sampler.__init__(self, features2process)

    def _sample(self, data, features, label):
        lables, cnts = np.unique(data[label], return_counts=True)
        avg = int(cnts.mean())
        sdata = pd.DataFrame(columns=data.columns)
        for lbl, cnt in zip(lables, cnts):
            if cnt > avg:
                # undersampling
                sdata = sdata.append(data[data[label] == lbl].sample(avg))
            elif cnt < avg:
                # oversampling
                sdata = sdata.append(data[data[label] == lbl])  # preserve original data
                sdata = sdata.append(data[data[label] == lbl].sample(avg - cnt, replace=True))
        return sdata


class SplAppoint(Sampler):
    def __init__(self, features2process, sample_count):
        """
        指定数目采样
        :param features2process:
        :param sample_count: 采样数
        """
        Sampler.__init__(self, features2process)
        self.sample_count = sample_count

    def _sample(self, data, features, label):
        y = data[label]
        labels = np.unique(y)
        if self.sample_count < 1:
            sample_count = int(data.shape[0] * self.sample_count)
        else:
            sample_count = self.sample_count
        avg_size = int(sample_count / labels.shape[0])
        sample_ret = data.sample(0)
        for label in labels:
            sample_ret = sample_ret.append(data[y == label].sample(avg_size))
        return sample_ret


