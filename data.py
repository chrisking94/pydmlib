from  base import pd, time, RSObject, re, np, RSTable
import pymssql
from collections import Iterable


class RSDataMetaclass(type):
    def __new__(mcs, name, bases, attrs):
        func_dict = {}
        mcs._rfaf([pd.DataFrame], func_dict)
        func_dict = [(k, mcs._wrap_return(v)) for (k, v) in func_dict.items() if k not in set(attrs.keys())]
        attrs.update(func_dict)
        return type(name, bases, attrs)

    @staticmethod
    def _wrap_return(func):
        def wrappedfunc(self, *arg, **kwargs):
            ret = func(self, *arg, **kwargs)
            if isinstance(self, RSData):
                if isinstance(ret, pd.DataFrame):
                    try:
                        ret = self.__class__(data=ret)
                        ret.name = self.name
                    except:
                        self.warning('derived classes of RSData should have a __init__ function like \
                                     \n     def __init__(data=None, ...)\n')
                        ret = RSData(name=self.name, data=ret)
                    if not isinstance(ret.columns, RSData.Index):
                        ret.columns = RSData.Index(ret.columns)
                elif ret is None:
                    if not isinstance(self.columns, RSData.Index):
                        self.columns = RSData.Index(self.columns)
            return ret
        return wrappedfunc

    @staticmethod
    def _rfaf(classes_, dict_ret):  # recursively find all functions
        bases = []
        for c in classes_:
            funcs = [(k, v) for (k, v) in c.__dict__.items()
                     if v.__class__.__name__ == 'function'
                     and k not in dict_ret.keys()]
            dict_ret.update(funcs)
            bases.extend(c.__bases__)
        if bases.__len__() > 0:
            RSDataMetaclass._rfaf(bases, dict_ret)


class RSDataIndexMetaclass(type):
    def __new__(cls, name, bases, attrs):
        func_dict = {}
        cls._rfaf([pd.core.indexes.base.Index], func_dict)
        exclude_funcs = {'_reset_identity'}
        exclude_funcs.update(attrs.keys())
        func_dict = [(k, cls._wrap_return(v)) for (k, v) in func_dict.items() if k not in exclude_funcs]
        attrs.update(func_dict)
        return type(name, bases, attrs)

    @staticmethod
    def _wrap_return(func):
        def wrappedfunc(self, *arg, **kwargs):
            ret = func(self, *arg, **kwargs)
            if isinstance(self, RSData.Index) and isinstance(ret, pd.core.indexes.base.Index):
                ret = self.__class__(ret)
            return ret
        return wrappedfunc

    @staticmethod
    def _rfaf(classes_, dict_ret):  # recursively find all functions
        bases = []
        for c in classes_:
            funcs = [(k, v) for (k, v) in c.__dict__.items()
                     if v.__class__.__name__ == 'function'
                     and k not in dict_ret.keys()]
            dict_ret.update(funcs)
            bases.extend(c.__bases__)
        if bases.__len__() > 0:
            RSDataIndexMetaclass._rfaf(bases, dict_ret)


class RSData(pd.DataFrame, RSObject, metaclass=RSDataMetaclass):
    class Index(pd.core.indexes.base.Index):  # rewrite Index
        # 没有显式声明__init__函数时，调用__init__构造对象时会出现两种情况
        # 1.__init__的实参单一且为基类对象，会返回一个实参的拷贝，其类型为当前类的类型
        # 2.其他情况的实参，则直接调用基类__init__并传入实参
        rgx_col_label = re.compile(r'<([^>]+)>')  # 列名label捕获
        rgx_multi_label = re.compile(r'([\|\&-])')  # label集合运算捕获
        rgx_none_bracket = re.compile(r'[^\(\)]+')  # 非括号捕获
        rgx_multi_expr = re.compile(r'( \| | \& | - )')  # 多表达式捕获
        rgx_white_char = re.compile(r'\s')  # 白字符捕获

        def __getitem__(self, item):
            if item is None:
                return self
            elif isinstance(item, str):
                list_exprs = self.rgx_multi_expr.split(item)
                if item == '':
                    return self
                elif len(list_exprs) > 1:
                    # multi expressions
                    # e.g. item='@@l | @abc & bv,dc,zx'
                    # then list_exprs=['@@l', ' | ', '@abc', ' & ', 'bv,dc,zx']
                    item = self._gen_cols_from_expr_chips(list_exprs, self._get_cols_by_expr)
                elif item[0] == '@':
                    if len(item) > 2 and item[1] == '@':  # @@xc 使用标签匹配，返回带x,c标签的列
                        item = item[2:]
                        list_lbls = self.rgx_multi_label.split(item)
                        if len(list_lbls) > 1:
                            # multi label
                            # e.g. item='@@a|b&c-d'
                            # then list_lbls = ['a', '|', 'b', '&', 'c', '-', 'd']
                            cols = self._gen_cols_from_expr_chips(list_lbls, self._get_cols_by_label)
                        else:
                            cols = self._get_cols_by_label(item)
                    else:  # @使用正则表达式
                        regx = re.compile(item[1:])
                        cols = [x for x in self if regx.search(x) is not None]
                    item = cols
                elif item == 'S':
                    # universal set
                    item = self
                else:
                    item_dict = self._get_item_mapping_dict()
                    if item in item_dict.keys():
                        return item_dict[item]
                    else:
                        return item
            elif isinstance(item, tuple):
                # 可以进行集合运算
                # 例：('A|(B&C)', expA, expB, expC)
                item = self._gen_set(item)
            elif isinstance(item, list):
                # 返回交集，忽略标签
                item_dict = self._get_item_mapping_dict()
                item = [self.rgx_col_label.sub('', x) for x in item]
                item = [item_dict[x] for x in item if x in item_dict.keys()]
            else:
                item = pd.core.indexes.base.Index.__getitem__(self, item)
            if (not isinstance(item, str)) and isinstance(item, Iterable):
                item = pd.core.indexes.base.Index(item)
                return RSData.Index(item)
            else:
                return item

        def _get_item_mapping_dict(self):
            """
            dict(none_label = labeled)
            :return:
            """
            return dict(zip([self.rgx_col_label.sub('', x) for x in self], self))

        def _gen_set(self, tp):
            sl = []
            set_list = []
            ret = ([1])
            S = set(self)
            for i, exp in enumerate(tp[1:]):
                var_name = chr(65 + i)
                set_list.append(set(self.__getitem__(exp)))
                sl.append('%s = set_list[%d]' % (var_name, i))
            sl.append('ret[0] = %s' % tp[0])
            script = '\n'.join(sl)
            try:
                exec(script)
            except Exception:
                raise Exception('Expression [%s] is invalid!Please check expression and parameters.'
                                % tp[0])
            return ret[0]

        def _get_cols_by_label(self, s_label):
            item = set(s_label)
            cols = []
            for col in self:
                labels = self.rgx_col_label.findall(col)
                if labels.__len__() > 0:
                    labels = labels[0]
                    if 'x' not in labels:
                        if 'y' not in labels and 'r' not in labels:
                            labels = 'x%s' % labels
                    elif 'x' == labels:
                        labels = 'cx'
                else:
                    labels = 'cx'  # 默认为cx标签
                if item.issubset(set(labels)):
                    cols.append(col)
            return cols

        def _get_cols_by_expr(self, s_expr):
            if len(s_expr) > 1:
                if s_expr[0] == '@':
                    return list(self.__getitem__(s_expr))
                else:
                    # 'a, b, c, d...'
                    return [self.rgx_white_char.sub('', x) for x in s_expr.split(',')]
            elif s_expr == 'S':
                return list(self)
            else:
                return []

        def _gen_cols_from_expr_chips(self, chips, func_get_cols):
            l_exp = []  # expression builder
            cols = ['']
            for i, chip in enumerate(chips):
                if i % 2:
                    # operator
                    l_exp.append(chip)
                else:
                    # label
                    var_name = chr(65 + int(i / 2))
                    expr = self.rgx_none_bracket.findall(chip)[0]
                    l_exp.append(self.rgx_none_bracket.sub(var_name, chip))
                    cols.append(func_get_cols(expr))
            cols[0] = ''.join(l_exp)
            cols = self.__getitem__(tuple(cols))
            return cols

    def __init__(self, data=None, index=None, columns=None, dtype=None,
                 copy=False, name='RSData'):
        pd.DataFrame.__init__(self, data, index, columns, dtype, copy)
        RSObject.__init__(self, name, 'random', 'default', 'underline')
        if not isinstance(self.columns, RSData.Index):
            self.columns = RSData.Index(self.columns)

    def toDataFrame(self):
        return pd.DataFrame.copy(self)

    def decompose_column(self, col, sep, inplace=False):
        """
        拆分一列字符串数据，字符串服从某种规律，
                            Char
                    如： [['a,b,c']
                         ['F,a,b']]
                    会被拆分为：
                        Char_a  Char_b  Char_c  Char_F
                        True    True    True    False
                        True    True    False   True
        :param col: 要拆分的列
        :param sep: 字符串分隔符
        :param inplace:
        :return:
        """
        # 拆分 sNoGestationReason
        self.starttimer()
        if inplace:
            data = self
        else:
            data = self.__class__(name=self.name, data=pd.DataFrame.copy(self))
        self.msg('decomposing %s by %s ...' % (col, sep))
        org = data[col]
        rows = org.unique()
        reasons = set()
        for row in rows:
            if row is not np.nan:
                reasons.update(row.split(sep))
        for reason in reasons:
            try:
                data['%s_%s' % (col, reason)] = org.str.contains(reason)
            except:
                self.warning('bad item "%s" ,ignored.' % reason)
        data.pop(col)
        self.msgtimecost()
        return data

    def data(self, label):
        ys = self.columns['@@y']
        rgx = re.compile(label)
        y = ''
        for y_ in ys:
            if rgx.search(y_) is not None:
                y = y_
                break
        if y == '':
            self.warning('No such <y> column %s.' % label)
            ret = None
        else:
            ret = pd.concat([self['@@x'], self[y]], axis=1)
            ret.rename({y: 'label'}, axis=1, inplace=True)
        return ret

    def __getitem__(self, item):
        if not (isinstance(item, str)) and not isinstance(item, tuple) and not isinstance(item, list):
            item = pd.DataFrame.__getitem__(self, item)
            return item
        else:
            return pd.DataFrame.__getitem__(self, self.columns[item])

    def __rshift__(self, other):
        if isinstance(other, int):
            if other == 0:
                return None
            else:
                return self
        else:
            from wrapper import wrap, WrpUnknown
            wrp = wrap(None, other)
            if isinstance(wrp, WrpUnknown):
                pd.DataFrame.__rshift__(self, other)
            else:
                if hasattr(wrp, 'b_fit'):
                    wrp.b_fit = False  # step delay for wrapper
                return wrp(self)


class MSSqlData(RSData):
    def __init__(self, data=None, host='', user='', password='', database='',
                 table='', name=''):
        start = time.time()
        b_read_sql = False
        msg = ''
        if data is None:
            b_read_sql = True
            sql = '''SELECT * 
                    FROM %s
                ''' % table
            try:
                conn = pymssql.connect(host=host, user=user,
                                       password=password, database=database)
                data = pd.read_sql(sql, con=conn)
            except Exception as e:
                data = None
                msg = 'No data read from [%s], %s' % (table, e.__str__())
            else:
                conn.close()
        if name == '':
            name = '%s.%s' % (database, table)
        RSData.__init__(self, name=name, data=data)
        self.table = table
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        if b_read_sql:
            self.msgtimecost(start)
        if msg != '':
            self.warning(msg)

    def save(self, table='', **kwargs):
        """
        把self保存到数据库
        :if_exists: {‘fail’, ‘replace’, ‘append’}, default ‘fail’
        :return:
        """
        self.connect()
        if table == '':
            table = self.table
            if table == '':
                self.error('Please specify param table.')
        self.to_sql(self.table, self.conn, **kwargs)
        self.close()

    def connect(self):
        self.conn = pymssql.connect(host=self.host, user=self.user,
                                    password=self.password, database=self.database)

    def close(self):
        if self.conn is not None:
            self.conn.close()


class RSSeries(pd.Series, RSObject):
    def __repr__(self):
        return RSTable(self).__str__()


if __name__ == '__main__':
    data = [[1, 2, 5, 7], [3, 4, 6, 10], [5, 6, 7, 22]]
    data = RSData(data, columns=['<c>A', '<y>B', '<r>C', 'D'], name='R')
    print(data['(S - @@y)'])
    # print(data[('S-A-B', '@x', ['<c>A'])])
    # print(type(data.columns))
    # print(data.columns['@d'])
    # data.pop('<c>A')
    # print(type(data.columns))
    # print(data)
    # dt = data.data('B')
    # print(type(dt.columns))

    pass
