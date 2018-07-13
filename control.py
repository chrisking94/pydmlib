from base import *
import _thread


class RSControl(RSObject):
    controls = {}
    thread = None
    buffer = []
    interval = 20  # ms
    iTimer = 0  # ms
    s_out = ''

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
            RSControl.thread = _thread.start_new_thread(RSControl._process_refresh, ())

    @staticmethod
    def print(s, **kwargs):
        print(' ' * len(RSControl.s_out), end='\r')  # 清行
        print(s, **kwargs)

    @staticmethod
    def _process_refresh():
        s = ''
        last_s = ''
        while True:
            RSControl.buffer = []
            for ctrl in RSControl.controls.values():
                if ctrl.visible:
                    RSControl.buffer.append(ctrl.refresh())
            s = ''.join(RSControl.buffer)
            if s != last_s:
                RSControl.print(s, end='\r')
                last_s = s
                RSControl.s_out = s
            time.sleep(RSControl.interval / 1000.0)
            RSControl.iTimer += RSControl.interval


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
    fill_char = '■'
    null_char = '□'

    def __init__(self, **kwargs):
        self._width = 0
        self._s = ''
        self._percentage = 1
        self.percentage = 0
        self.width = 20  # unit:char
        RSControl.__init__(self, **kwargs)

    ##############
    # Properties #
    ##############

    @property
    def percentage(self):
        return self._percentage

    @percentage.setter
    def percentage(self, value):
        if 0 <= value <= 100:
            if value != self._percentage:
                self._percentage = value
                i = int(self._width * self._percentage / 100.0)
                self._s = '[%s%s]' % (CProgressBar.fill_char * i,
                                      CProgressBar.null_char * (self._width - i))
        else:
            self.error('invalid percentage %s.' % value)

    @property
    def width(self):
        return self._width + 2

    @width.setter
    def width(self, value):
        if value < 2:
            self._width = 0
            self._s = ''
        else:
            self._width = value - 2
            self._percentage -= 1
            self.percentage = self._percentage + 1

    def __str__(self):
        return self._s


def test():
    return
    RSControl.init()
    t = CTimer(visible=True)
    p = CProgressBar(percentage=50, visible=True)
    while 1:
        pass
    pass

