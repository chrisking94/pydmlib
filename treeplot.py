from .misc import RSPlot, RSObject, np
from random import random
import matplotlib.animation as ani


class TPObject(RSObject):
    b_vertical = True

    def __init__(self, name, color, **kwargs):
        RSObject.__init__(self, name=name, msgforecolor=color)
        self._left = 0
        self._top = 0
        self._width = 0
        self._height = 0
        self.__dict__.update(kwargs)

    ##############
    # Properties #
    ##############
    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, v):
        self._left = v

    @property
    def top(self):
        return self._top

    @top.setter
    def top(self, v):
        self._top = v

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, v):
        self._width = v

    @property
    def height(self):
        return self._width

    @height.setter
    def height(self, v):
        self._height = v

    @property
    def right(self):
        return self.left + self.width

    @property
    def bottom(self):
        return self.top + self.height

    @staticmethod
    def _(x, y):
        if TPObject.b_vertical:
            return x, y
        else:
            return y, x


class TPArc(TPObject):
    f_arrow_offset = 0.05

    def __init__(self, color, name='', start=None, end=None):
        self.start = start  # node
        self.end = end  # node
        TPObject.__init__(self, name, color)
        self.ax = None
        self.graph = None
        self.text = None
        self.rad = (random()*0.4 - 0.2)

    def draw(self, ax):
        ex_kwargs = None
        if self.start is None:
            ex_kwargs = dict(xy=self._(self.end.left+self.end.width/2, self.end.top))
        elif self.end is not None:
            ex_kwargs = dict(arrowprops=dict(arrowstyle="<|-",
                                             connectionstyle="arc3,rad=%f" % self.rad,
                                             fc="black"),
                             xy=self._(self.start.left + self.start.width / 2, self.start.top),
                             xytext=self._(self.end.left + self.end.width / 2, self.end.top))
        if ex_kwargs is not None:
            self.graph = ax.annotate(self.end.name,
                                     size=20, va="top", ha="center",
                                     xycoords='data',
                                     textcoords='data',
                                     bbox=dict(boxstyle="round4", fc='black', alpha=.25),
                                     color=self.end.msgforecolor,
                                     **ex_kwargs)
            if self.is_valid():
                self.text = ax.annotate(self.name,
                                        xy=self._(self.left+self.width/2.0,
                                                  self.top+self.height/2.0),
                                        xycoords='data',
                                        color=self._fore_color,
                                        ha='center')
        self.ax = ax

    def redraw(self):
        if self.graph is not None:
            self.graph.remove()
            del self.graph
            if self.text is not None:
                self.text.remove()
                del self.text
        if self.ax is not None:
            self.draw(self.ax)

    def is_valid(self):
        return self.start is not None and self.end is not None

    ##############
    # properties #
    ##############
    @property
    def left(self):
        if self.is_valid():
            return self.start.left if self.start.left < self.end.left else self.end.left

    @property
    def top(self):
        if self.is_valid():
            return self.start.top

    @property
    def width(self):
        if self.is_valid():
            return abs(self.start.left - self.end.left)

    @property
    def height(self):
        if self.is_valid():
            return self.end.top - self.start.top


class TPNode(TPObject):
    def __init__(self, name, color, **kwargs):
        self.children = []
        self._parent = None
        TPObject.__init__(self, name, color, **kwargs)
        self.arc = TPArc(color, start=self.parent, end=self)
        self.text = None
        self.text_height = 0.2

    def calc_rect(self):
        if len(self.children) > 0:
            self.b_vertical = True
            max_child_height = 0
            self.width = 0
            for child in self.children:
                child.left = self.left + self.width
                child.top = self.top + 1
                child.calc_rect()
                self.width += child.width + 1
                if child.height > max_child_height:
                    max_child_height = child.height
            self.height = max_child_height + 1
            self.width -= 1

    def draw(self, ax):
        self.arc.draw(ax)
        for child in self.children:
            child.draw(ax)
        pass

    def redraw(self):
        self.arc.redraw()

    def add_child(self, node):
        node.parent = self
        self.children.append(node)

    ##############
    # Properties #
    ##############
    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, node):
        self._parent = node
        self.arc.start = node


class TreePlot(RSPlot):
    def __init__(self, tree, width, height, **kwargs):
        RSObject.__init__(self)
        self.tree = tree
        self.width = width
        self.height = height
        self.b_vertical = True
        self.__dict__.update(kwargs)

    def plot(self, ax=None, **kwargs):
        fig = plt.figure(figsize=(self.width, self.height))
        ax = fig.add_subplot(111)
        ax.axis('off')
        self.tree.top = 0.1
        self.tree.calc_rect()
        TPObject.b_vertical = self.b_vertical
        ax.set_xlim(-0.1, self.tree.width + 0.1)
        ax.set_ylim(-0.1, self.tree.height / 2 + 0.1)
        self.tree.draw(ax)
        return fig


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    a = TPNode('A', 'red')
    b = TPNode('B', 'blue')
    c = TPNode('C', 'green')
    d = TPNode('D', 'blue')
    e = TPNode('E', 'red')
    f = TPNode('F', 'yellow')
    g = TPNode('G', 'cyan')
    h = TPNode('h', 'pink')
    i = TPNode('i', 'white')
    j = TPNode('j', 'black')
    k = TPNode('k', 'pink')
    l = TPNode('l', 'yellow')
    m = TPNode('m', 'cyan')

    a.add_child(b)

    a.add_child(k)
    a.add_child(l)

    a.add_child(c)

    b.add_child(d)
    b.add_child(e)
    b.add_child(m)

    c.add_child(f)
    f.add_child(g)
    f.add_child(h)
    f.add_child(i)
    f.add_child(j)

    tp = TreePlot(a, 5, 5, b_vertical=True)
    tp.plot()
    plt.show()

