from base import *
from misc import *
import sklearn.model_selection as cv


class ModelTester(RSObject):
    def __init__(self, name='ModelTester'):
        RSObject.__init__(self, name, 'red', 'default', 'highlight')

    def report(self):
        self.error('Not implemented!')


class MTAutoGrid(ModelTester):
    def __init__(self, data_procr_grid):
        """
        自动化网格测试
        :param data_procr_grid: 数据处理器表，表末尾必须是分类器，结构示例如下：
                            [[procr11, procr12],
                             [procr21, procr22, procr23],
                             procr3,
                             ...
                             [clf1, clf2, ...]]
        """
        ModelTester.__init__(self, 'AutoGridModelTester')
        self.data_procr_grid = data_procr_grid
        self.report_table = None  # pd.DataFrame()
        self.process_table = []
        self.check_points = []  # 输出检查点
        self._gen_process_table(data_procr_grid, [])  # 存入self.process_table

    def _gen_process_table(self, data_procr_grid, processor_sequence):
        """
        生成处理流程表，存放在process_table中
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
            if i==0:
                ps = processor_sequence.copy()
            else:
                ps = [TCContinue('TCContinue-%s' % x.name) for x in processor_sequence[:-1]]
                ps.append(processor_sequence[-1])  # 复制数据检查点
            ps.append(procr)
            if isinstance(procr, TesterController):
                if isinstance(procr, TCBreak):  # break
                    self.check_points.append(procr)
                    self.process_table.append(ps)
                    return False
                elif isinstance(procr, TCEnd):  # end
                    self.check_points.append(procr)
                    self.process_table.append(ps)
                    return True
                elif isinstance(procr, TCCheckPoint):
                    self.check_points.append(procr)
            if data_procr_grid.__len__() > 1:
                if not self._gen_process_table(data_procr_grid[1:], ps):
                    return False
            else:
                cp = TCCheckPoint()
                self.check_points.append(cp)
                ps.append(cp)
                self.process_table.append(ps)
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
        for i, sequence in enumerate(self.process_table):
            cdata = data
            self.msgtime(msg='开始执行第%d/%d个测试序列。' % ((i+1), self.process_table.__len__()))
            b_contains_continue = False
            report_table_head = []
            report_line = []
            for procr in sequence:
                cdata = procr.fit_transform(cdata)
                if isinstance(procr, TesterController):
                    if isinstance(procr, TCContinue):  # continue
                        b_contains_continue = True
                    elif isinstance(procr, TCCheckPoint):
                        procr.set_infolist(report_line)
                report_line.extend(procr.get_report())
                if not b_contains_continue and not b_head_done:
                    report_table_head.extend(procr.get_report_title())
            # 生成表头
            if not b_contains_continue:
                if not b_head_done:
                    b_head_done = True
                    self.report_table = pd.DataFrame(self.report_table, columns=report_table_head)
            # 添加记录
            if self.report_table is None:
                self.report_table = pd.DataFrame([report_line])
            else:
                self.report_table.loc[self.report_table.shape[0], :] = report_line
            # 输出报告
            current_row = self.report_table.loc[self.report_table.shape[0] - 1]
            self._submsg('report', 'green',
                         '\n%s' % current_row.__str__())
            self.msgtimecost(msg='第%d/%d个测试完成。' % ((i+1), self.process_table.__len__()))
        self.msgtimecost(msg='总耗时。')
        self.msgtime(msg='测试完成！调用log()可获取测试日志。')
        self.log()
        return cdata

    def data(self, check_point_id=-1):
        """
        返回CheckPoint中保存的数据
        :param check_point_id: int:
                                    id(>=100): 可通过log()查看
                                    index(<100): 在self.check_points中的索引
                                str:
                                    TCCheckPoint().name
        :return:
        """
        if isinstance(check_point_id, int) and check_point_id <100:
            return self.check_points[check_point_id].data
        else:
            checkpoints = [x for x in self.check_points if x.is_me(check_point_id)]
            if checkpoints.__len__()>0:
                return checkpoints[0].data
            else:
                self.warning('No such CheckPoint[id=%d]' % check_point_id)

    def log(self):
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            return self.report_table

    def read_checkpoints(self):
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            return pd.DataFrame(data=[(x.id, x.strid, x.name, x.info()) for x in self.check_points],
                                columns=['id', 'strid', 'Name', 'Info'])

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


class TesterController(RSDataProcessor):
    def __init__(self, name='TesterController'):
        RSDataProcessor.__init__(self, None, name, 'white', 'black', 'default')


class TCCheckPoint(TesterController):
    def __init__(self, name = ''):
        if name == '':
            TesterController.__init__(self, 'TCCP-%d')
            self.name = self.name % self.id
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

    def is_me(self, id):
        if isinstance(id, str):
            return id == self.strid
        else:
            return id == self.id

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


class TCBreak(TCCheckPoint):
    def __init__(self, name='TC-Break'):
        TCCheckPoint.__init__(self, name)

    def fit_transform(self, data):
        data = TCCheckPoint.fit_transform(self, data)
        self.msgtime(self._colorstr('**break*point' * 8, 0, self.msgforecolor, self.msgbackcolor))
        return data


class TCContinue(TesterController):
    def __init__(self, name='TC-Continue'):
        TesterController.__init__(self, name)

    def fit_transform(self, data):
        self.msg('skipped.')
        return data


class TCEnd(TCCheckPoint):
    def __init__(self, name='TC-End'):
        TCCheckPoint.__init__(self, name)

    def fit_transform(self, data):
        data = TCCheckPoint.fit_transform(self, data)
        self.msgtime(self._colorstr('--end-point' * 10, 0, self.msgforecolor, self.msgbackcolor))
        return data


