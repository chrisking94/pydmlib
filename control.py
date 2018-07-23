from .base import *
from threading import Thread


class RSControl(RSObject):
    controls = {}
    thread = None
    buffer = []
    interval = 5  # ms
    iTimer = 0  # ms
    s_out = ''

    class RSControlThread(Thread):
        def __init__(self, *args, **kwargs):
            Thread.__init__(self, *args, **kwargs)
            self.state = 'running'  # running

        def pause(self):
            self.state = 'pause'
            while self.state != 'paused':
                pass

        def resume(self):
            self.state = 'running'

        def terminate(self):
            self.state = 'terminated'

        def run(self):
            last_s = ''
            while self.state != 'terminated':
                if self.state == 'running':
                    t = time.time()
                    RSControl.buffer = []
                    for ctrl in RSControl.controls.values():
                        if ctrl.visible:
                            RSControl.buffer.append(ctrl.refresh())
                    s = ''.join(RSControl.buffer)
                    if s != last_s:
                        len_diff = len(last_s) - len(s)
                        if len_diff > 0:
                            print('%s%s' % (s, ' ' * len_diff), end='\r')
                        else:
                            print(s, end='\r')
                        last_s = s
                        RSControl.s_out = s
                    interval_s = RSControl.interval / 1000.0  # unit:s
                    delta_t = time.time() - t
                    if delta_t < interval_s:
                        time.sleep(interval_s - delta_t)
                    delta_t = time.time() - t
                    RSControl.iTimer += int(delta_t * 1000)
                elif self.state == 'pause':
                    n = len(last_s)
                    if n > 0:
                        print(' ' * n, end='\r')  # 清行
                        last_s = ''
                    self.state = 'paused'
                elif self.state == 'paused':
                    continue

    def __init__(self, **kwargs):
        RSObject.__init__(self)
        self.visible = True
        self.wait_interval_index_dict = {}  # unit:ms
        RSControl.controls[self.name] = self
        self.__dict__.update(kwargs)
        if self.name in RSControl.controls.keys():
            self.name = '%s_%d' % (self.name, self.id)

    def refresh(self):
        """
        refresh
        :return: str
        """
        return self.__str__()

    def wait(self, t):
        """
        wait for a while
        firstly run in 2nd wait invoking
        :param t: ms
        :return: boolean, whether can go on
        """
        if t == 0:
            return True
        t_index = int(self.iTimer / t)
        if t in self.wait_interval_index_dict.keys():
            if t_index > self.wait_interval_index_dict[t]:
                self.wait_interval_index_dict[t] = t_index
                return True
        else:
            self.wait_interval_index_dict[t] = t_index
        return False

    def _destroy(self):
        if self.name in RSControl.controls.keys():
            RSControl.controls.pop(self.name)

    def destroy(self):
        self._destroy()

    def __del__(self):
        self._destroy()

    @staticmethod
    def init():
        if RSControl.thread is None:
            RSControl.thread = RSControl.RSControlThread()
            RSControl.thread.start()


class CStandbyCursor(RSControl):
    chars = '-\\|/'
    
    def __init__(self, **kwargs):
        RSControl.__init__(self, **kwargs)
        self._i = 0

    def __str__(self):
        if self.wait(400):
            if self._i < len(self.chars)-1:
                self._i += 1
            else:
                self._i = 0
        return self.chars[self._i]


class CTimer(RSControl):
    def __init__(self, **kwargs):
        RSControl.__init__(self, **kwargs)
        self.t = time.time()
        self.st = '0s'

    def reset(self):
        self.t = time.time()

    def __str__(self):
        if self.wait(1000):
            t = int(time.time() - self.t)
            m, s = divmod(t, 60)
            h, m = divmod(m, 60)
            st = '%02d:%02d:%02d' % (h, m, s)
            self.st = st
        return self.st


class CLabel(RSControl):
    def __init__(self, **kwargs):
        self._text = ''
        RSControl.__init__(self, **kwargs)

    ##############
    # Properties #
    ##############
    @property
    def text(self):
        if callable(self._text):
            return self._text()
        else:
            return self._text

    @text.setter
    def text(self, text):
        self._text = text

    def __str__(self):
        return self.text


class CProgressBar(RSControl):
    fill_char = '❚'
    null_char = '❚'

    def __init__(self, **kwargs):
        self._width = 0
        self._s = ''
        self._percentage = 1
        self.percentage = 0
        self.width = 40  # unit:char
        RSControl.__init__(self, **kwargs)

    ##############
    # Properties #
    ##############

    @property
    def percentage(self):
        return self._percentage

    @percentage.setter
    def percentage(self, value):
        """
        set percentage
        :param value: float
        :return:
        """
        if self._width == 0:
            self._s = ''
            return
        if value < 0:
            value = 0
        elif value > 100:
            value = 100
        if value != self._percentage:
            self._percentage = value
            i = int(self._width * self._percentage / 100.0)
            null_block = self.null_char * (self._width - i)
            if self.fill_char == self.null_char:
                null_block = self.colorstr(null_block, 0, 7, 8)
            self._s = '[%s%s%d%%]' % (self.fill_char * i,
                                      null_block,
                                      self.percentage)

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        if value < 0:
            self._width = 0
            self._s = ''
        else:
            self._width = value
            self._percentage -= 1  # refresh
            self.percentage = self._percentage + 1

    def __str__(self):
        return self._s


class CTimeProgressBar(CProgressBar):
    def __init__(self, time_=0, **kwargs):
        """

        :param time: s
        :param kwargs:
        """
        CProgressBar.__init__(self, **kwargs)
        self.time_ = time_
        self.start_ = time.time()

    def reset(self, time_):
        """
        reset time
        :param time_:
        :return:
        """
        self.start_ = time.time()
        self.time_ = time_

    def __str__(self):
        if self.time_ == 0:
            self.percentage = 100
        else:
            self.percentage = (time.time() - self.start_) * 100 / self.time_
        return CProgressBar.__str__(self)


def test():
    return
    RSControl.init()
    t = CTimer(visible=True)
    p = CProgressBar(percentage=50, visible=True)
    while 1:
        pass
    pass

