from base import *


class Discretor(RSDataProcessor):
    def __init__(self, features2process, name='Discretor'):
        RSDataProcessor.__init__(self, features2process, name, 'red', 'white')


class DsctChi2(Discretor):
    def __init__(self, features2process, min_interval=1):
        """
        chi^2离散
        """
        Discretor.__init__(self, features2process, 'χ²离散')
        self.min_interval = min_interval
        self.min_epos = 0.05
        self.final_bin = []

    def _fit(self, x, y, min_interval=1):
        self.min_interval = min_interval
        x = np.floor(x)
        x = np.int32(x)
        min_val = np.min(x)
        bin_dict = {}
        bin_li = []
        for i in range(len(x)):
            pos = (x[i] - min_val) / min_interval * min_interval + min_val
            target = y[i]
            bin_dict.setdefault(pos, [0, 0])
            if target == 1:
                bin_dict[pos][0] += 1
            else:
                bin_dict[pos][1] += 1

        for key, val in bin_dict.items():
            t = [key]
            t.extend(val)
            bin_li.append(t)

        bin_li.sort(key=lambda x: x[0], reverse=False)

        L_index = 0
        R_index = 1
        self.final_bin.append(bin_li[L_index][0])
        while True:
            L = bin_li[L_index]
            R = bin_li[R_index]
            # using infomation gain;
            p1 = L[1] / (L[1] + L[2] + 0.0)
            p0 = L[2] / (L[1] + L[2] + 0.0)

            if p1 <= 1e-5 or p0 <= 1e-5:
                LGain = 0
            else:
                LGain = -p1 * np.log(p1) - p0 * np.log(p0)

            p1 = R[1] / (R[1] + R[2] + 0.0)
            p0 = R[2] / (R[1] + R[2] + 0.0)
            if p1 <= 1e-5 or p0 <= 1e-5:
                RGain = 0
            else:
                RGain = -p1 * np.log(p1) - p0 * np.log(p0)

            p1 = (L[1] + R[1]) / (L[1] + L[2] + R[1] + R[2] + 0.0)
            p0 = (L[2] + R[2]) / (L[1] + L[2] + R[1] + R[2] + 0.0)

            if p1 <= 1e-5 or p0 <= 1e-5:
                ALLGain = 0
            else:
                ALLGain = -p1 * np.log(p1) - p0 * np.log(p0)

            if np.absolute(ALLGain - LGain - RGain) <= self.min_epos:
                # concat the interval;
                bin_li[L_index][1] += R[1]
                bin_li[L_index][2] += R[2]
                R_index += 1

            else:
                L_index = R_index
                R_index = L_index + 1
                self.final_bin.append(bin_li[L_index][0])

            if R_index >= len(bin_li):
                break

    def _transform(self, x):
        res = []
        for e in x:
            index = self.get_Discretization_index(self.final_bin, e)
            res.append(index)

        res = np.asarray(res)
        return res

    def _process(self, data, features, label):
        """
        chi2离散
        :param data:
        :return:
        """
        X = data[features]
        y = data[label]
        for col in features:
            xs = X[col].values
            # preprocessing
            if xs.max() == xs.min():
                self.warning('column [%s] values are all [%f], all set to %d' % (col, xs.min(), 1))
                xs = xs - xs.min() + 1
            else:
                ys = y.values
                self._fit(xs, ys, self.min_interval)
            X[col] = self._transform(xs)
        data[features] = X
        return X

    def get_Discretization_index(self, Discretization_vals, val):
        index = -1
        for i in range(len(Discretization_vals)):
            e = Discretization_vals[i]
            if val <= e:
                index = i
                break

        return index


class DsctMonospace(Discretor):
    def __init__(self, features2process, bin_amount=10):
        """
        等宽离散
        :param bin_amount:默认分10桶
        """
        Discretor.__init__(self, features2process, '等宽离散，bin_amount=%d' % bin_amount)
        self.bin_amount = bin_amount

    def _process(self, data, features, label):
        """
        :param X: pandas.DataFrame([feature1, feature2, ...])
        """
        X, y = data[features], data[label]
        bs = self.bin_amount
        bs = (X.max() - X.min()) / bs
        bs[bs == 0] = 1  # 避免除数为0
        X = ((X - X.min()) / bs).round()
        data[features] = X
        return data


class DsctInfomationEntropy(Discretor):
    def __init__(self, features2process, min_infomation_gain=0.01):
        """
        基于信息熵的离散法
        原理： http://www.doc88.com/p-352263493037.html
        :param features2process:
        :param min_infomation_gain:
        """
        Discretor.__init__(self, features2process, '信息熵分裂')

    def _process(self, data, features, label):
        return data



