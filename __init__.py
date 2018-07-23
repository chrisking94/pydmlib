from .outlierhandler import *
from .factorencoder import *
from .discretor import *
from .featureclassifier import *
from .featurecombiner import *
from .featureselector import *
from .misc import *
from .integration import *
from .nanhandler import *
from .normalizer import *
from .sampler import *
from .transformer import *
from .reporter import *
from .formatconverter import *
from .wrapper import *
from .data import *
from .costestimator import RSCostEstimator
import base
import misc
import wrapper
import data
import control
import integration
import costestimator


############################
#   module initialization  #
############################
from utils import get_option_configurator

cfg = get_option_configurator('./pydmlib.cfg')


def set_option(*args, **kwargs):
    for i in range(0, len(args), 2):
        k, v = args[i], args[i+1]
        if k == 'dp.msg_mode':
            RSDataProcessor.s_msg_mode = v
        elif k == 'dp.progressbar0':
            RSDataProcessor.progressbar.null_char = v
        elif k == 'dp.progressbar1':
            RSDataProcessor.progressbar.fill_char = v
        elif k == 'dp.cursor':
            RSDataProcessor.cursor.chars = v
        elif k == 'plot.enable':
            RSPlot.b_enable = v
        else:
            pd.set_option(k, v, **kwargs)
        cfg.set('Config', k, v)
    cfg.write()


if cfg.has_section('Config'):
    for option in cfg.items('Config'):
        set_option(*option)
else:
    cfg.add_section('Config')

RSControl.init()
RSCostEstimator.init()  # initialize statically


############################
#          debug           #
############################
if __name__ == '__main__':
    base.test()
    integration.test()
    misc.test()
    wrapper.test()
    data.test()
    control.test()
    costestimator.test()

