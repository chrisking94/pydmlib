from base import *
from threading import Thread


class RSControl(RSObject):
    controls = {}
    thread = None
    buffer = []
    interval = 20  # ms
    iTimer = 0  # ms
    s_out = ''

    class RSControlThread(Thread):
        def __init__(self, *args, **kwargs):
            Thread.__init__(self, *args, **kwargs)
            self.state = 'running'  # running

        def pause(self):
            self.state = 'pause'

        def resume(self):
            self.state = 'running'

        def terminate(self):
            self.state = 'terminated'

        def run(self):
            last_s = ''
            while self.state != 'terminated':
                if self.state == 'running':
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
                time.sleep(RSControl.interval / 1000.0)
                RSControl.iTimer += RSControl.interval

    def __init__(self, **kwargs):
        RSObject.__init__(self)
        self.visible = True
        self.t_until = -1  # ms
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
        :param t: ms
        :return: boolean, whether can go on
        """
        if self.t_until == -1:
            self.t_until = RSControl.iTimer + t
            return False
        elif RSControl.iTimer >= self.t_until:
            self.t_until = -1
            return True
        else:
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

    @staticmethod
    def print(s, **kwargs):
        print(' ' * len(RSControl.s_out), end='\r')  # 清行
        print(s, **kwargs)

    @staticmethod
    def show(*args, **kwargs):
        RSControl.thread.pause()
        print(' ' * len(RSControl.s_out), end='\r')  # 清行
        plt.show(*args, **kwargs)
        RSControl.thread.resume()


RSControl.init()


class CStandbyCursor(RSControl):
    def __init__(self, **kwargs):
        RSControl.__init__(self, **kwargs)
        self._i = 0
        self._chars = '-\\|/'

    def __str__(self):
        if self.wait(400):
            if self._i < len(self._chars)-1:
                self._i += 1
            else:
                self._i = 0
        return self._chars[self._i]


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
    null_char = '⌷'

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
            self._s = '[%s%s%d%%]' % (CProgressBar.fill_char * i,
                                      self.colorstr(CProgressBar.fill_char * (self._width - i), 0, 7, 8),
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
            self._percentage -= 1
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

