from .treeplot import TreePlot
from .base import RSObject
from abc import abstractmethod
import matplotlib.animation as ani
from .misc import RSPlot


class Animation(RSPlot):
    def __init__(self):
        RSPlot.__init__(self)
        self.redraw_list = []
        self.ani = None

    def run(self):
        fig = self.plot()
        self.ani = ani.FuncAnimation(fig, self.redraw, frames=1, interval=100)

    @abstractmethod
    def redraw(self, frame_data):
        pass

    def add_redraw(self, node):
        self.redraw_list.append(node)


class TreeAni(Animation):
    def __init__(self, *args, **kwargs):
        Animation.__init__(self)

    def redraw(self, frame_data):
        redraw_list = self.redraw_list.copy()
        self.redraw_list = []
        if len(redraw_list) > 0:
            for node in redraw_list:
                node.redraw()


