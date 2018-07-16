from base import *
from sklearn.tree import DecisionTreeRegressor
from threading import Thread
import hashlib
import inspect


class CostEstimator(RSObject):
    save_folder = '%s/pydm/estimator/' % os.path.expanduser('~')
    if not os.path.exists(save_folder):
        try:
            os.makedirs(save_folder)
        except:
            save_folder = './pydm/estimator/'
            os.makedirs(save_folder)
    max_rows_saved = 1000
    estimators = {}

    def __init__(self, project_name, **kwargs):
        RSObject.__init__(self, project_name)
        self.project_name = project_name
        self.filepath = '%s%s.csv' % (self.save_folder, self.project_name)
        self.data = None
        self.load()

    def load(self):
        if os.path.exists(self.filepath):
            self.data = pd.read_csv(self.filepath)

    def save(self):
        if self.data.shape[0] > self.max_rows_saved:
            data = self.data.iloc[-1000:, :]
        else:
            data = self.data
        data.to_csv(self.filepath, index=False)


class TimeCostEstimator(CostEstimator):
    class Factors(list):
        # 对于非简单（非int, float, ...）因子，会展开__init__中的参数
        # 展开时会再次遇到非简单因子，如此递归，max_depth决定递归的深度，=1表示只展开第一层__init__
        max_depth = 1
        memorize_threshold = 1  # unit:s, cost大于prediction_threshold时才记录本次获得的经验

        def __init__(self, immutable_factors=()):
            list.__init__(self)
            self.immutable_factors = immutable_factors
            self.clear()

        def _append(self, obj, depth):
            if isinstance(obj, bool):
                list.append(self, int(obj))
            elif isinstance(obj, int) or isinstance(obj, float) or isinstance(obj, str):
                list.append(self, obj)
            elif isinstance(obj, pd.DataFrame):
                list.extend(self, obj.shape)
            elif depth < self.max_depth:  # 其他类型， 展开__init__中的参数
                for var_name in obj.__init__.__code__.co_varnames[1:]:  # 不包含self
                    if hasattr(obj, var_name):
                        self._append(getattr(obj, var_name), depth+1)

        def append(self, obj):
            self._append(obj, 0)

        def extend(self, iterable):
            for x in iterable:
                self.append(x)

        def clear(self):
            list.clear(self)
            self.extend(self.immutable_factors)

    def __init__(self, project_name, immutable_factors=(), **kwargs):
        """

        :param project_name:
        :param factor_names: list
        :param kwargs:
        """
        CostEstimator.__init__(self, project_name, **kwargs)
        self.time = 0
        self.factors = TimeCostEstimator.Factors(immutable_factors)
        self.predictor = None
        self.train()
        self.end_data = None

    def _get_machine_info(self):
        try:
            import psutil
            machine_info = [psutil.virtual_memory().percent, psutil.cpu_percent(1)]
        except ImportError:
            psutil = None
            machine_info = [0, 0]
        return machine_info

    def predict(self):
        self.time = time.time()
        self.factors.extend(self._get_machine_info())
        if self.predictor is not None:
            x = self.factors + [-1]
            if len(x) == self.data.shape[1]:
                self.data.loc[self.data.shape[0], :] = x
                data = self._get_dummies()
                x = data.iloc[-1, :-1]
                self.data.drop(self.data.shape[0] - 1, axis=0, inplace=True)
                try:
                    return self.predictor.predict(x.reshape(1, -1)) - 1
                except RuntimeError:
                    return -1
        return -1

    def memorize_experience(self):
        """
        memorize时才会清空factors
        :return:
        """
        cost = time.time() - self.time
        if cost > 1:
            new_experience = self.factors + [cost]
            if self.data is None or self.data.shape[0] == 0 or len(new_experience) != self.data.shape[1]:
                # factors发生变化，丢弃原来的数据
                self.data = pd.DataFrame(data=[new_experience])
            else:
                self.data.loc[self.data.shape[0], :] = new_experience
            self.factors.clear()  # 重置factors
            self.save()
            self.train()

    def train(self, time_out=0):
        """
        train predictor in new thread, wait while time out ,thread termination or exception occurs
        :param time_out: unit:s
        :return:
        """
        if self.data is None:
            return False
        elif self.data.shape[0] < 2 or self.data.shape[1] < 2:
            return False
        t = Thread(target=self._thread_train)
        t.start()
        t.join(time_out)
        return not t.isAlive()

    def _get_dummies(self):
        fact = self.data.columns[self.data.dtypes == 'object']
        if len(fact) > 0:
            data = self.data.drop(columns=fact)
            data = pd.concat([pd.get_dummies(self.data[fact]), data], axis=1)
        else:
            data = self.data
        return data

    def _thread_train(self):
        data = self._get_dummies()
        X, y = data.iloc[:, :-1], data.iloc[:, -1]
        if self.predictor is None:
            self.predictor = DecisionTreeRegressor()
        self.predictor.fit(X, y)

    @staticmethod
    def get_estimator(character, immutable_factors=()):
        if isinstance(character, TimeCostEstimator.get_estimator.__class__):
            s = inspect.getsource(character)
            character = hashlib.sha224(s.encode('utf-8')).hexdigest()
        if character in TimeCostEstimator.estimators.keys():
            return TimeCostEstimator.estimators.keys()
        else:
            return TimeCostEstimator(character, immutable_factors)


def test():
    return
    e = TimeCostEstimator('test')
    for i in range(1, 10):
        print(e.predict(i))
        time.sleep(i)
        e.memorize_experience()
    return

