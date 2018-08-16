from .base import *
from sklearn.tree import DecisionTreeRegressor
from threading import Thread
import hashlib
import inspect


class RSCostEstimator(RSObject):
    save_folder = '%s/pydm/estimator/' % os.path.expanduser('~')
    if not os.path.exists(save_folder):
        try:
            os.makedirs(save_folder)
        except Exception as e:
            save_folder = './pydm/estimator/'
            os.makedirs(save_folder)
    max_rows_saved = 1000
    estimators = {}

    def __init__(self, project_name, **kwargs):
        RSObject.__init__(self, 'CE-%s' % project_name)
        self.load_file = True
        self.project_name = project_name
        self.file_path = '%s%s.csv' % (self.save_folder, self.project_name)
        self.data = None
        self.new_experience = None
        self.__dict__.update(kwargs)
        if self.load_file:
            self.load()

    def load(self):
        if os.path.exists(self.file_path):
            self.data = pd.read_csv(self.file_path)

    def save(self):
        if self.data.shape[0] > self.max_rows_saved:
            self.data.drop(index=range(self.data.shape[0] - 40, self.data.shape[0]), inplace=True)
        self.data.to_csv(self.file_path, index=False)

    def save_new_exp(self):
        if self.new_experience is not None:
            if self.data is None or \
                    self.data.shape[0] == 0 or \
                    len(self.new_experience) != self.data.shape[1]:
                # overwrite
                self.data = pd.DataFrame(data=[self.new_experience])
                self.save()
            else:
                self.data.loc[self.data.shape[0], :] = self.new_experience
                # append
                s_out = ','.join([str(x) for x in self.new_experience])
                s_out = '%s\n' % s_out
                with open(self.file_path, 'a') as f:
                    f.write(s_out)

    @staticmethod
    def init():
        thread = Thread(target=RSCostEstimator._thread_init())
        thread.start()

    @staticmethod
    def _thread_init():
        """
        initialize subclasses here
        :return:
        """
        # manage experience files
        f_info = pd.DataFrame(columns=['file', 'size', 'delta_acc_day'])
        for f_name in os.listdir(RSCostEstimator.save_folder):
            if f_name.endswith('.csv'):
                f_path = '%s%s' % (RSCostEstimator.save_folder, f_name)
                f_access_time = datetime.datetime.fromtimestamp(os.path.getatime(f_path))
                f_size = os.path.getsize(f_path)
                now_time = datetime.datetime.now()
                delta_at = now_time - f_access_time
                if '#' in f_name:  # temp experience file
                    f_info.loc[f_info.shape[0], ] = [f_name, f_size, delta_at.days]
        # delete some functional experience files which are too old
        series_dad = f_info['delta_acc_day']
        f_del = f_info['file'][(series_dad-series_dad.mean()) > 3*series_dad.std()]
        for f_name in f_del.iteritems():
            f_path = '%s%s' % (RSCostEstimator.save_folder, f_name)
            print('%s was removed.' % f_path)
            os.remove(f_path)


class CETime(RSCostEstimator):
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
            elif isinstance(obj, int) or isinstance(obj, float):
                if np.isnan(obj):
                    list.append(self, -1)
                else:
                    list.append(self, obj)
            elif isinstance(obj, str):
                list.append(self, obj)
            elif isinstance(obj, pd.DataFrame):
                list.extend(self, obj.shape)
            elif depth < self.max_depth:  # 其他类型， 展开__init__中的参数
                for var_name in obj.__init__.__code__.co_varnames[1:]:  # 不包含self
                    if hasattr(obj, var_name):
                        self._append(getattr(obj, var_name), depth+1)
            else:
                list.append(self, 0)  # 设为缺省

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
        RSCostEstimator.__init__(self, project_name, **kwargs)
        self.factors = None
        self.predictor = None
        self.train(1)
        self.end_data = None

    def _get_machine_info(self):
        try:
            import psutil
            machine_info = [psutil.virtual_memory().percent, psutil.cpu_percent()]
        except ImportError:
            psutil = None
            machine_info = [0, 0]
        return machine_info

    def predict(self):
        try:
            self.factors.extend(self._get_machine_info())
            if self.predictor is not None:
                x = self.factors + [-1]
                if len(x) == self.data.shape[1]:
                    self.data.loc[self.data.shape[0], :] = x
                    data = self._get_processed_data()
                    x = data.iloc[-1, :-1]
                    self.data.drop(self.data.shape[0] - 1, axis=0, inplace=True)
                    try:
                        return self.predictor.predict(x.values.reshape(1, -1)) - 1
                    except ValueError as e:
                        # self.warning('prediction failed! %s' % e.__str__())
                        return -1
        except TypeError as e:
            return -1
        except Exception as e:
            raise e
        return -1

    def memorize_experience(self):
        """
        memorize时才会清空factors
        :return:
        """
        cost = time.time() - self._time_start
        if cost > 1 and self.factors is not None:
            self.new_experience = self.factors + [cost]
            self.save_new_exp()
            self.factors = None  # 重置factors
            self.train()

    def abandon_experience(self):
        self.factors = None

    def train(self, time_out=0):
        """
        train predictor in new thread, wait while time out ,thread termination or exception occurs
        :param time_out: unit:s
        :return:
        """
        if self.data is None:
            return False
        elif self.data.shape[0] < 1 or self.data.shape[1] < 2:
            return False
        t = Thread(target=self._thread_train)
        t.start()
        # t.setDaemon(True)
        # t.join(time_out)
        return not t.isAlive()

    def _get_processed_data(self):
        data = self.data
        if data.shape[0] > 2:
            # 去掉单值列
            data = data[data.columns[
                pd.Series([len(data[x].unique()) > 1 for x in data.columns[:-1]]+[True])]]
        # 编码factors
        fact = data.columns[data.dtypes == 'object']
        if len(fact) > 0:
            encoded = pd.get_dummies(data[fact].astype('str'))
            data = data.drop(columns=fact)
            data = pd.concat([encoded, data], axis=1)
        if data.shape[1] < 2:
            data = self.data.iloc[:, -2:]
        return data

    def _thread_train(self):
        data = self._get_processed_data()
        X, y = data.iloc[:, :-1], data.iloc[:, -1]
        if self.predictor is None:
            self.predictor = DecisionTreeRegressor()
        self.predictor.fit(X, y)
        if self.data.shape[0] > self.max_rows_saved:
            self.save()

    @staticmethod
    def get_estimator(character):
        if isinstance(character, CETime.get_estimator.__class__):
            s = inspect.getsource(character)
            rgx = re.compile(r'\s')
            s = rgx.sub('', s)
            character = hashlib.sha224(s.encode('utf-8')).hexdigest()
            character = '#%s' % character
        if character in CETime.estimators.keys():
            # 各个estimator的immutable_factors一般不同
            estimator = CETime.estimators[character]
        else:
            estimator = CETime(character)
            CETime.estimators[character] = estimator
        return estimator



