from .base import pd, time, RSObject, re, np, RSTable
from sqlalchemy.types import NVARCHAR, Float, Integer, SmallInteger
from sqlalchemy import create_engine
from collections import Iterable
from sqlalchemy.exc import ProgrammingError


class RSData(pd.DataFrame, RSObject):
    class Index(pd.core.indexes.base.Index):  # rewrite Index
        # 没有显式声明__init__函数时，调用__init__构造对象时会出现两种情况
        # 1.__init__的实参单一且为基类对象，会返回一个实参的拷贝，其类型为当前类的类型
        # 2.其他情况的实参则直接调用基类__init__并传入实参
        rgx_col_label = re.compile(r'<([^>]+)>')  # 列名label捕获
        rgx_multi_label = re.compile(r'([\|\&-])')  # label集合运算捕获
        rgx_none_bracket = re.compile(r'[^\(\)]+')  # 非括号捕获
        rgx_multi_expr = re.compile(r'( \| | \& | - )')  # 多表达式捕获
        rgx_white_char = re.compile(r'\s')  # 白字符捕获

        class Item(str):
            built_in_function_labels = set('xyr')
            built_in_type_labels = set('dcl')

            def __new__(cls, o):
                return str.__new__(cls, o)

            def __init__(self, o):
                """
                <*>name
                :param o:
                """
                str.__init__(self)
                self._name = RSData.Index.rgx_col_label.sub('', self)
                self._labels = RSData.Index.rgx_col_label.findall(o)
                if len(self._labels) > 0:
                    self._labels = set(self._labels[0])
                else:
                    self._labels = set()
                if len(self.built_in_function_labels & self._labels) == 0:
                    self._labels.add('x')
                if len(self.built_in_type_labels & self._labels) == 0:
                    self._labels.add('c')

            @property
            def name(self):
                return self._name

            @property
            def labels(self):
                return self._labels

        built_in_labels = Item.built_in_type_labels | Item.built_in_function_labels

        def __getitem__(self, item):
            item_ = item
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
                elif ',' in item:
                    # 'a, b, c' represent columns [a] [b] [c]
                    item = self._get_cols_by_expr(item)
                else:
                    item = self.Item(item)
                    for x in self:
                        x = self.Item(x)
                        if x.name == item.name:
                            return x
                    if item in self.built_in_labels:
                        itm = self._get_cols_by_label(item)
                        if len(itm) == 0:
                            item = None
                        elif len(itm) == 1:
                            item = itm[0]
                        else:
                            item = itm
                    else:
                        item = None
                    if item is None:
                        raise KeyError('[%s] not found!' % item_)
                    else:
                        return item
            elif isinstance(item, tuple):
                # 可以进行集合运算
                # 例：('A|(B&C)', expA, expB, expC)
                item = self._gen_set_by_tuple(item)
            elif isinstance(item, list):
                # 返回交集，忽略标签
                if len(item) > 0:
                    if isinstance(item[0], str):
                        names = set([self.Item(x).name for x in item])
                        item = [x for x in self if self.Item(x).name in names]
                    else:
                        item = pd.core.indexes.base.Index.__getitem__(self, item)
                else:
                    item = []
            else:
                item = pd.core.indexes.base.Index.__getitem__(self, item)
            if (not isinstance(item, str)) and isinstance(item, Iterable):
                item = pd.core.indexes.base.Index(item)
                return RSData.Index(item)
            else:
                return item

        def __contains__(self, item):
            if isinstance(item, tuple):
                return True
            elif isinstance(item, str):
                if len(item) > 1 and item[0] == '@':
                    return True
                elif item in self.built_in_labels:
                    return True
                else:
                    try:
                        item = self[item]
                    except KeyError:
                        return False
                    return super(RSData.Index, self).__contains__(item)
            else:
                return False

        def _gen_set_by_tuple(self, tp):
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
            cols = [x for x in self if item.issubset(self.Item(x).labels)]
            return cols

        def _get_cols_by_expr(self, s_expr):
            if len(s_expr) > 1:
                if s_expr[0] == '@':
                    return list(self.__getitem__(s_expr))
                else:
                    # 'a, b, c, d...'
                    cols = [self.rgx_white_char.sub('', x) for x in s_expr.split(',')]
                    return list(self[cols])
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

    def to_data_frame(self):
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
        self.start_timer()
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
            except Exception:
                self.warning('bad item "%s" ,ignored.' % reason)
        data.pop(col)
        self.msg_time_cost()
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
        return ret

    def _getitem_column(self, key):
        key = self.columns[key]
        if isinstance(key, str):
            return pd.DataFrame._getitem_column(self, key)
        else:
            return self._getitem_array(key)

    def _getitem_array(self, key):
        """
        <labels> will be ignored
        :param key:
        :return:
        """
        if isinstance(key, list):
            key = self.columns[key]
        return pd.DataFrame._getitem_array(self, key)

    def _set_item(self, key, value):
        try:
            key = self.columns[key]
            if not isinstance(key, str):
                self._setitem_array(key, value)
                return
        except KeyError:
            pass
        pd.DataFrame._set_item(self, key, value)

    def __rshift__(self, other):
        if isinstance(other, int):
            if other == 0:
                return None
            elif other == 1:
                self.msg(str(self.shape), 'shape')
            elif other == 2:
                self.msg(str(self.columns), 'columns')
            return self
        else:
            from .wrapper import wrap, WrpUnknown
            wrp = wrap(None, other)
            if isinstance(wrp, WrpUnknown):
                pd.DataFrame.__rshift__(self, other)
            else:
                if hasattr(wrp, 'b_fit'):
                    wrp.b_fit = False  # step delay for wrapper
                return wrp(self)

    def __setattr__(self, name, value):
        if name[0] == '_':
            RSObject.__setattr__(self, name, value)
        else:
            if name in self.columns:
                ns = self.columns[name]
                if isinstance(ns, str):
                    name = ns
                else:
                    self.warning('multi column[%s]: %s found, set %s as a new column/attribute.'
                                 % (name, ns.__str__(), name))
            pd.DataFrame.__setattr__(self, name, value)

    #################
    #   Properties  #
    #################
    @property
    def columns(self):
        cols = RSData.Index(self._data.axes[0])
        return cols

    @columns.setter
    def columns(self, v):
        if len(v) == len(self._data.axes[0]):
            self._data.axes[0] = pd.core.indexes.base.Index(v)
        else:
            raise ValueError('length of new column must be same with old one\'s.')

    @property
    def _constructor(self):
        def constructor(*args, **kwargs):
            data = RSData(*args, **kwargs)
            data.name = self.name
            return data
        return constructor


class MSSqlData(RSData):
    def __init__(self, data=None, host='', user='', password='', database='',
                 table='', name=''):
        start = time.time()
        b_read_sql = False
        msg = ''
        column_types = {}
        if host == '':
            engine = None
        else:
            engine = create_engine("mssql+pymssql://%s:%s@%s/%s" % (user, password, host, database))
        if data is None:
            b_read_sql = True
            sql = '''SELECT * 
                    FROM %s
                ''' % table
            try:
                con = engine.connect()
                data = pd.read_sql(sql, con=con)
                #  dtype mapping
                sql = '''
                            SELECT COLUMN_NAME, DATA_TYPE
                            FROM INFORMATION_SCHEMA.columns
                            WHERE TABLE_NAME=\'%s\' and TABLE_CATALOG=\'%s\'
                            ''' % (table, database)
                ct = pd.read_sql(sql, con)
                column_types = dict(zip(ct['COLUMN_NAME'], ct['DATA_TYPE']))
                con.close()
            except ProgrammingError as e:
                if e.code == 'f405':
                    pass  # no such table
                else:
                    raise e
            except Exception as e:
                raise e
        if name == '':
            name = '%s.%s' % (database, table)
        RSData.__init__(self, data=data, name=name)
        self.engine = engine
        self.table = table
        self.column_types = column_types
        if b_read_sql:
            if msg == '':  # no exception occurred
                self.msg_time_cost(start)
            else:
                self.warning(msg)

    def save(self, table='', **kwargs):
        """
        把self保存到数据库
        :if_exists: {‘fail’, ‘replace’, ‘append’}, default ‘fail’
        :return:
        """
        con = self.engine.connect()
        if table == '':
            table = self.table
            if table == '':
                self.error('Please specify param table.')
        self.to_sql(self.table, con, **kwargs)
        con.close()

    #################
    #   Properties  #
    #################
    pass


class RSSeries(pd.Series, RSObject):
    def __repr__(self):
        return RSTable(self).__str__()


if __name__ == '__main__':
    dt = [[1, 2, 5, 7], [3, 4, 6, 10], [5, 6, 7, 22]]
    dt = RSData(dt, columns=['<c>A', '<y>B', '<r>C', 'D'], name='R')
    print(dt['(S - @@y)'])
    # print(data[('S-A-B', '@x', ['<c>A'])])
    # print(type(data.columns))
    # print(data.columns['@d'])
    # data.pop('<c>A')
    # print(type(data.columns))
    # print(data)
    # dt = data.data('B')
    # print(type(dt.columns))

    pass
