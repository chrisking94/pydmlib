import matplotlib.pyplot as ppt


class RSPlotManager(object):
    def __init__(self):
        self.__dict__.update(ppt.__dict__)
        self.__dict__.update(RSPlotManager.__dict__)

    def show(*args, **kwargs):
        from control import RSControl
        RSControl.thread.pause()
        ppt.show(*args, **kwargs)
        RSControl.thread.resume()


plt = RSPlotManager()


def printf(s, *args, **kwargs):
    from control import RSControl
    RSControl.thread.pause()
    if isinstance(s, str):
        try:
            print(s % args, **kwargs)
        except Exception as e:
            print(s, *args, **kwargs)
    else:
        print(str(s), **kwargs)
    RSControl.thread.resume()

