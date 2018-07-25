from misc import RSPlot, RSObject
import matplotlib.pyplot as plt


class TPObject(RSObject):
    def __init__(self, name, color, **kwargs):
        RSObject.__init__(self, name=name, msgforecolor=color)
        self.left = 0
        self.top = 0
        self.width = 0
        self.height = 0
        self.__dict__.update(kwargs)


class TPArc(TPObject):
    f_arrow_offset = 0.05

    def __init__(self, color, name='', start=None, end=None):
        self.start = start  # node
        self.end = end  # node
        RSObject.__init__(self, name, color)

    def draw(self, ax):
        if self.start is not None and self.end is not None:
            y_offset = self.start.text_height / 2
            ax.arrow(self.start.left + self.start.width/2, self.start.top+y_offset,
                     self.end.left + self.end.width/2-self.start.left-self.start.width/2,
                     self.end.top-self.start.top-y_offset,
                     length_includes_head=True, head_width=.05, head_length=.25,
                     fc=self.msgforecolor, ec=self.msgforecolor)


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
            last_child = None
            max_child_height = 0
            self.width = 0
            for child in self.children:
                if last_child is None:
                    child.left = self.left
                else:
                    child.left = last_child.left + last_child.width + 1
                child.top = self.top + 1
                child.calc_rect()
                self.width += child.width + 1
                if child.height > max_child_height:
                    max_child_height = child.height
                last_child = child
            self.height = max_child_height + 1
            self.width -= 1

    def draw(self, ax):
        self.text = ax.text(self.left + self.width/2, self.top, self.name,
                            fontsize=15, verticalalignment='bottom', horizontalalignment='center',
                            color=self.msgforecolor,
                            bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.3))
        self.arc.draw(ax)
        for child in self.children:
            child.draw(ax)
        pass

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
        self.__dict__.update(kwargs)

    def plot(self, ax=None, **kwargs):
        fig = plt.figure(figsize=(self.width, self.height))
        ax = fig.add_subplot(111)
        self.tree.calc_rect()
        ax.set_xlim(-0.1, self.tree.width + 0.1)
        ax.set_ylim(-0.1, self.tree.height + 0.1)
        self.tree.draw(ax)


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

    a.add_child(b)
    a.add_child(c)
    b.add_child(d)
    b.add_child(e)
    c.add_child(f)
    f.add_child(g)
    f.add_child(h)

    p = TreePlot(a, 5, 5)
    p.plot()
    plt.show()

