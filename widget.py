#####################
#   Static Widgets  #
#####################
from utils import cfg


class RSRect(object):
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self._width = width
        self._height = height

    #################
    #   Properties  #
    #################
    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, v):
        if v == -1:
            self._width = cfg.stage.width - self.x
        else:
            self._width = v

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, v):
        if v == -1:
            self._height = cfg.stage.height - self.y
        else:
            self._height = v

    @property
    def right(self):
        return self.x + self.width

    @property
    def bottom(self):
        return self.y + self.height


class Widget(RSRect):
    def __init__(self, x, y, width=-1, height=-1):
        RSRect.__init__(self, x, y, width, height)
