from base import *


class Discretor(RSDataProcessor):
    def __init__(self, name='Discretor'):
        super(Discretor, self).__init__(name, 'red', 'white')

    def fit_transform(X, y):
        raise Exception('error:Not implemented!')


class DsctChi2(Discretor):
    def __init__(self, min_interval=1):
        '''
        chi^2离散
        '''
        super(DsctChi2, self).__init__('χ²离散')
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

    def fit_transform(self, data, features=None):
        '''
        监督离散
        :param X:pandas.DataFrame
        :param y:pandas.Series
        '''
        self.starttimer()
        X, y = self._getXy(data, features)
        for col in X.columns:
            xs = X[col].values
            # preprocessing
            if (xs.max() == xs.min()):
                self.msg('%s-warning:column [%s] values are all [%f], all set to %d' % (self.name, col, xs.min(), 1))
                xs = xs - xs.min() + 1
            else:
                ys = y.values
                self._fit(xs, ys, self.min_interval)
            X[col] = self._transform(xs)
        self.msgtimecost()
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
    def __init__(self, bin_size=None):
        '''
        等宽离散
        :param bin_size:默认分10桶
        '''
        super(DsctMonospace, self).__init__('等宽离散')
        self.bin_size = bin_size

    def fit_transform(self, data, features=None):
        '''
        :param X: pandas.DataFrame([feature1, feature2, ...])
        '''
        self.starttimer()
        X, y = self._getXy(data, features)
        bs = self.bin_size
        if bs == None:
            bs = (X.max() - X.min()) / 10
        bs[bs == 0] = 1  # 避免除数为0
        X = ((X - X.min()) / bs).round()
        self.msgtimecost()
        return X


class DsctNone(Discretor):
    def __init__(self):
        '''
        不离散
        '''
        super(DsctNone, self).__init__('不离散')

    def fit_transform(self, X, y=None):
        return X

