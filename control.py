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
        self.error('Must implement refresh()!')

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

    def __str__(self):
        return self.refresh()

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
        self._char = '-'

    def refresh(self):
        if self.wait(400):
            if self._char == '-':
                self._char = '\\'
            elif self._char == '\\':
                self._char = '|'
            elif self._char == '|':
                self._char = '/'
            else:
                self._char = '-'
        return self._char


class CTimer(RSControl):
    def __init__(self, **kwargs):
        RSControl.__init__(self, **kwargs)
        self.t = time.time()
        self.st = '0s'

    def refresh(self):
        if self.wait(1000):
            t = int(time.time() - self.t)
            m, s = divmod(t, 60)
            h, m = divmod(m, 60)
            st = '%02d:%02d:%02d' % (h, m, s)
            self.st = st
        return self.st

    def reset(self):
        self.t = time.time()


class CLabel(RSControl):
    def __init__(self, **kwargs):
        self.text = ''
        RSControl.__init__(self, **kwargs)

    def refresh(self):
        if callable(self.text):
            return self.text()
        else:
            return self.text

