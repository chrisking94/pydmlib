from base import *


class Sampler(RSDataProcessor):
    def __init__(self, features2process, name='Sampler'):
        super(Sampler, self).__init__(features2process, name, 'blue', 'yellow', 'highlight')


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
        super(SplUnder, self).__init__(features2process, '向下采样')
        self.feature_weights = feature_weights

    def _process(self, data, features, label):
        feature_weights = self.feature_weights
        self.msg('采样前行数 %d' % data.shape[0])
        lables, cnts = np.unique(data[label], return_counts=True)
        self.msg('minor class count: %d' % cnts.min())
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
        self.msg('采样后行数 %d' % newdata.shape[0])
        return newdata


class SplMiddle(Sampler):
    def __init__(self, features2process):
        super(SplMiddle, self).__init__(features2process, '中间采样')

    def _process(self, data, features, label):
        self.msg('--sample count before undersampling %d' % data.shape[0])
        lables, cnts = np.unique(data[label], return_counts=True)
        avg = int(cnts.mean())
        self.msg(
            '\n--minor class count: %d \n--major class count: %d\n--average count: %d' % (cnts.min(), cnts.max(), avg))
        sdata = pd.DataFrame(columns=data.columns)
        for lbl, cnt in zip(lables, cnts):
            if cnt > avg:
                # undersampling
                sdata = sdata.append(data[data[label] == lbl].sample(avg))
            elif cnt < avg:
                # oversampling
                sdata = sdata.append(data[data[label] == lbl])  # preserve original data
                sdata = sdata.append(data[data[label] == lbl].sample(avg - cnt, replace=True))
        self.msg('--sample count after undersampling %d' % data.shape[0])
        return sdata


class SplAppoint(Sampler):
    def __init__(self, features2process, sample_count):
        """
        指定数目采样
        :param features2process:
        :param sample_count: 采样数
        """
        super(SplAppoint, self).__init__(features2process, '指定采样，样本大小=%.2f' % sample_count)
        self.sample_count = sample_count

    def _process(self, data, features, label):
        y = data[label]
        labels = np.unique(y)
        if self.sample_count < 1:
            sample_count = int(data.shape[0] * self.sample_count)
        else:
            sample_count = self.sample_count
        avg_size = int(sample_count / labels.shape[0])
        sample_ret = pd.DataFrame(columns=data.columns)
        for label in labels:
            sample_ret = sample_ret.append(data[y == label].sample(avg_size))
        return sample_ret


