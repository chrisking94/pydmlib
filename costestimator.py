from base import *
from sklearn.tree import DecisionTreeRegressor


class CostEstimator(RSObject):
    save_folder = '%s/pydm/estimator/' % os.path.expanduser('~')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    max_rows_saved = 1000

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
    def __init__(self, project_name, **kwargs):
        """

        :param project_name:
        :param factor_names: list
        :param kwargs:
        """
        CostEstimator.__init__(self, project_name, **kwargs)
        self.time = 0

    def _get_machine_info(self):
        try:
            import psutil
            machine_info = [psutil.virtual_memory().percent, psutil.cpu_percent(1)]
        except ImportError:
            psutil = None
            machine_info = [0, 0]
        return machine_info

    def predict(self, *factors):
        # factors = self._get_machine_info() + [x for x in factors]
        new_line = []
        for f in factors:
            if isinstance(f, pd.DataFrame):
                new_line.extend(f.shape)
            else:
                new_line.append(f)
        new_line.append(0)  # y
        self.time = time.time()
        if self.data is None:
            self.data = pd.DataFrame(data=[new_line])
        else:
            self.data.loc[self.data.shape[0], :] = new_line
            str_cols = self.data.columns[self.data.dtypes == 'object']
            if len(str_cols) > 0:
                drp = self.data.drop(columns=str_cols)
                data = pd.get_dummies(self.data[str_cols])
                if drp.shape[0] > 0:
                    data = pd.concat([drp, data], axis=1)
            else:
                data = self.data
            X, y = data.iloc[:-1, :-1], data.iloc[:-1, -1]
            x = data.iloc[-1, :-1]
            if self.data.shape[0] > 2:
                clf = DecisionTreeRegressor()
                clf.fit(X, y)
                return clf.predict(x.reshape(1, -1))
        return 2

    def memorize_experience(self):
        cost = time.time() - self.time
        self.data.iloc[-1, -1] = cost
        self.save()


def test():
    return
    e = TimeCostEstimator('test')
    for i in range(1, 10):
        print(e.predict(i))
        time.sleep(i)
        e.memorize_experience()
    return

