from base import *
from misc import *
import sklearn.model_selection as cv
from wrapper import IWrap


class ModelTester(RSObject):
    def __init__(self, name='ModelTester'):
        RSObject.__init__(self, name, 'red', 'default', 'highlight')

    def report(self):
        self.error('Not implemented!')


class ProcessorSequence(RSList, RSDataProcessor):
    def __init__(self, copyfrom=()):
        RSDataProcessor.__init__(self, None, 'ProcessorSequence', 'random', 'random')
        for i, procr in enumerate(copyfrom):
            if not isinstance(procr, RSDataProcessor):
                copyfrom[i] = IWrap(None, procr)
        RSList.__init__(self, copyfrom)
        self.report_line = []
        self.report_title = []
        self.b_contains_continue = False
        self.report_table = None

    def fit_transform(self, data):
        self.report_line = []
        #  从后往前找到第一个可用检查点
        tmplist = self[:-1]
        tmplist.reverse()
        available_checkpoint = None
        for procr in tmplist:
            if isinstance(procr, TCCheckPoint):
                if procr.is_available():
                    available_checkpoint = procr
        for procr in self:
            if available_checkpoint is not None:  # 从检查点开始执行
                if available_checkpoint == procr:
                    available_checkpoint = None
            if available_checkpoint is None:
                #  运行处理器
                if isinstance(procr, TCContinue):
                    self.b_contains_continue = True
                data = procr.fit_transform(data)
                self.report_line.extend(procr.get_report())
                self.report_title.extend(procr.get_report_title())
            else:
                #  CheckPoint之前，只添加报告，不做运算
                self.report_line.extend(procr.get_report())
                self.report_title.extend(procr.get_report_title())
        self.report_table = pd.DataFrame(self.report_line, index=self.report_title)
        return data

    def _wrapProcr(self, procr):
        if isinstance(procr, RSDataProcessor):
            return procr
        else:
            return IWrap(None, procr)

    def reset(self, from_=0):
        """
        从from_节点起，重置它和它之后的所有CheckPoint
        :param from_: 节点索引，或id
        :return:
        """
        index = self.get_index(from_)
        for x in self[index:]:
            if isinstance(x, TCCheckPoint):
                x.reset()

    def log(self):
        self._submsg('report', 'green',
                     '\n%s' % self.report_table.__str__())

    def get_report(self):
        return self.report_line

    def get_report_title(self, *args):
        return self.report_title

    def contains_continue(self):
        return self.b_contains_continue

    def info(self):
        return pd.DataFrame(data=[(x.id, x.name) for x in self],
                            columns=('id', 'name'))

    def __setitem__(self, key, value):
        """
        会把key之后的CheckPoint全部reset
        :param key:
        :param value:
        :return:
        """
        index = self.get_index(key)
        self.reset(index+1)
        RSList.__setitem__(self, key, self._wrapProcr(value))

    def insert(self, index, object):
        """
        会把index及之后的CheckPoint全部reset
        :param index:
        :param object:
        :return:
        """
        index = self.get_index(index)
        self.reset(index)
        RSList.insert(self, index, self._wrapProcr(object))

    def append(self, object):
        RSList.append(self, self._wrapProcr(object))


class MTAutoGrid(ModelTester, RSList):
    def __init__(self, data_procr_grid=None, copyfrom=()):
        """
        自动化网格测试
        :param data_procr_grid: 数据处理器表，表末尾必须是分类器，结构示例如下：
                            [[procr11, procr12],
                             [procr21, procr22, procr23],
                             procr3,
                             ...
                             [procrn1, procrn2, ...]]
        """
        RSList.__init__(self, copyfrom)
        ModelTester.__init__(self, 'AutoGridModelTester')
        if data_procr_grid is not None:
            #  format processor grid
            if data_procr_grid.__len__() == 0 or not isinstance(data_procr_grid[-1], TCEnd):
                data_procr_grid.append(TCEnd())
            if not isinstance(data_procr_grid[0], TCStart):
                data_procr_grid.insert(0, TCStart())
            self.data_procr_grid = data_procr_grid
            self.report_table = None  # pd.DataFrame()
            self.check_points = ProcessorSequence()  # 所有检查点的列表
            self._gen_process_table(data_procr_grid, ProcessorSequence())  # 存入self

    def _gen_process_table(self, data_procr_grid, processor_sequence):
        """
        生成处理流程表，存放在self中
        :param data_procr_grid:  处理器网格
        :param processor_sequence: list,  用于存放一条处理序列的列表
        :return: 遇到TCBreak返回False，否则返回True
        """
        cur_procrs = data_procr_grid[0]
        if not isinstance(cur_procrs, list):
            cur_procrs = [cur_procrs]
        if cur_procrs.__len__() > 1:
            # 多处理器分叉，设置数据检查点
            processor_sequence.append(TCCheckPoint())
        for i, procr in enumerate(cur_procrs):
            ps = processor_sequence.copy()
            if isinstance(procr, TCCheckPoint):
                procr = procr.copy()
                ps.append(procr)
                self.check_points.append(procr)
                if isinstance(procr, TCBreak):  # break
                    self.append(ps)
                    return False
                elif isinstance(procr, TCEnd):  # end
                    self.append(ps)
                    return True
            else:
                ps.append(procr)
            if data_procr_grid.__len__() > 1:
                if not self._gen_process_table(data_procr_grid[1:], ps):
                    return False
        return True

    def fit_transform(self, data):
        """
        开始测试
        :param data:输入数据集
        :return:
        """
        self.starttimer()
        b_head_done = False
        self.report_table = None
        cdata = data
        for i, sequence in enumerate(self):
            self.msgtime(msg='开始执行第%d/%d个测试序列。' % ((i+1), self.__len__()))
            cdata = sequence.fit_transform(data)
            # 生成表头
            if not sequence.contains_continue() and not b_head_done:
                b_head_done = True
                self.report_table = pd.DataFrame(self.report_table, columns=sequence.get_report_title())
            # 添加记录
            if self.report_table is None:
                self.report_table = pd.DataFrame([sequence.get_report()])
            else:
                self.report_table.loc[self.report_table.shape[0], :] = sequence.get_report()
            # 输出报告
            sequence.log()
            self.msgtimecost(msg='第%d/%d个测试完成。' % ((i+1), self.__len__()))
        self.msgtimecost(msg='总耗时。')
        self.msgtime(msg='测试完成！调用log()可获取测试日志。')
        self.log()
        return cdata

    def data(self, id_index=-1):
        """
        返回CheckPoint中保存的数据
        :param id_index: int:
                                    id(>=100): 可通过log()查看
                                    index(<100): 在self.check_points中的索引
                                str:
                                    TCCheckPoint().name
        :return:
        """
        return self.check_points[id_index].data

    def log(self):
        return self.report_table

    def read_checkpoints(self):
        return pd.DataFrame(data=[(x.id, x.strid, x.name, x.info()) for x in self.check_points],
                            columns=['id', 'strid', 'Name', 'Info'])

    def show(self, what='log'):
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            if what == 'log':
                print(self.log())
            elif what == 'checkpoints':
                print(self.read_checkpoints())

    def savelog(self, filepath='', btimesuffix=False):
        """
        save log as .csv
        :param filepath: don't carry a .csv suffix, e.g.usr/xxx is fine.
        :param btimesuffix: append a time suffix after filename,e.g filename --> filename_2018-05-22 14:53:00
        :return:
        """
        if filepath == '':
            filepath = '%s_%s.csv' % (self.name, self.strtime())
        else:
            if btimesuffix:
                filepath = '%s_%s.csv' % (filepath, self.strtime())
        self.report_table.to_csv(filepath, sep='\t')

    def run_sequence(self, index):
        """
        运行测试序列表中的一条测试序列
        :param index:
        :return:
        """
        pass

    def info(self):
        pass


class TesterController(RSDataProcessor):
    def __init__(self, name='TesterController'):
        RSDataProcessor.__init__(self, None, name, 'white', 'black', 'default')


class TCCheckPoint(TesterController):
    def __init__(self, name = ''):
        if name == '':
            TesterController.__init__(self, 'TCCP')
            self.name = 'TCCP-%d' % self.id
            self.strid = ''
        else:
            self.strid = name
            TesterController.__init__(self, 'TCCP-%s' % name)
        self.data = None
        self.infolist = None
        self.copy_count = 0

    def fit_transform(self, data):
        self.msgtime(self._colorstr('++check+point' * 8, 0, self.msgforecolor, self.msgbackcolor))
        if self.data is None:
            self.data = data
        return self.data

    def is_me(self, id_name):
        if isinstance(id_name, str):
            return id_name == self.strid
        else:
            return TesterController.is_me(self, id_name)

    def is_available(self):
        return self.data is not None

    def copy(self):
        if self.strid == '':
            return self.__class__()
        else:
            self.copy_count += 1
            return self.__class__('%s%d' % (self.strid, self.copy_count))

    def set_infolist(self, info):
        self.infolist = info.copy()

    def info(self):
        return self.infolist.__str__()

    def reset(self):
        self.data = None


class TCBreak(TCCheckPoint):
    def __init__(self, name='TC-Break'):
        TCCheckPoint.__init__(self, name)

    def fit_transform(self, data):
        data = TCCheckPoint.fit_transform(self, data)
        self.msgtime(self._colorstr('**break*point' * 8, 0, self.msgforecolor, self.msgbackcolor))
        return data


class TCContinue(TesterController):
    def __init__(self):
        TesterController.__init__(self, 'TC-Continue')

    def fit_transform(self, data):
        self.msg('continue.')
        return data


class TCSkip(TesterController):
    def __init__(self, processor):
        TesterController.__init__(self, 'TCSkip-%s' % processor.name)
        self.processor = processor
        self.b_runonce = False

    def fit_transform(self, data):
        if self.b_runonce:
            self.b_runonce = False
            return self.processor.fit_transform(data)
        else:
            self.msg('skipped.')
            return None

    def reset(self):
        self.b_runonce = True


class TCStart(TCCheckPoint):
    def __init__(self, name='Start'):
        TCCheckPoint.__init__(self, name)

    def fit_transform(self, data):
        data = TCCheckPoint.fit_transform(self, data)
        self.msgtime(self._colorstr('--start-point' * 8, 0, self.msgforecolor, self.msgbackcolor))
        return data


class TCEnd(TCCheckPoint):
    def __init__(self, name='End'):
        TCCheckPoint.__init__(self, name)

    def fit_transform(self, data):
        data = TCCheckPoint.fit_transform(self, data)
        self.msgtime(self._colorstr('--end-point' * 9, 0, self.msgforecolor, self.msgbackcolor))
        return data


def test():
    return
    grid = [
        [TCCheckPoint(), TCCheckPoint()],
        TCContinue(),
        [TCCheckPoint(), TCCheckPoint()]
    ]
    mt = MTAutoGrid(grid)
    for s in mt:
        print(s.copy().__class__.__name__)
    pass