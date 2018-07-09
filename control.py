from base import *
import _thread


class RSControl(RSObject):
    controls = {}
    thread = None
    buffer = []
    interval = 20  # ms
    iTimer = 0  # ms

    def __init__(self, name):
        if name in RSControl.controls.keys():
            name = '%s_%d' % (name, RSObject.id_count)
        RSObject.__init__(self, name)
        self.visible = True
        self.t_until = -1  # ms
        RSControl.controls[self.name] = self

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
        :return: boolean, can go on
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
                print(s, end='\r')
                last_s = s
            time.sleep(RSControl.interval / 1000.0)
            RSControl.iTimer += RSControl.interval


RSControl.init()


class CStandbyCursor(RSControl):
    def __init__(self, name=''):
        RSControl.__init__(self, name)
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
    def __init__(self, name=''):
        RSControl.__init__(self, name)
        self.t = time.time()
        self.st = '0s'

    def refresh(self):
        if self.wait(1000):
            t = int(time.time() - self.t)
            m, s = divmod(t, 60)
            h, m = divmod(m, 60)
            st = '%02d:%02d:%02d' % (h, m, s)
            self.st = 'time: %s' % st
        return self.st

    def reset(self):
        self.t = time.time()

