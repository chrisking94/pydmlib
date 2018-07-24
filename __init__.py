from  outlierhandler import *
from  factorencoder import *
from  discretor import *
from  featureclassifier import *
from  featurecombiner import *
from  featureselector import *
from  misc import *
from  integration import *
from  nanhandler import *
from  normalizer import *
from  sampler import *
from  transformer import *
from  reporter import *
from  formatconverter import *
from  wrapper import *
from  data import *


############################
#   module initialization  #
############################
from  utils import GlobalOption
from  control import RSControl
from  costestimator import RSCostEstimator

cfg = GlobalOption('./pydmlib.cfg')


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
    from  base import test as base_test
    from  misc import test as misc_test
    from  wrapper import test as wrapper_test
    from  data import test as data_test
    from  control import test as control_test
    from  integration import test as integration_test
    from  costestimator import test as costestimator_test
    from  utils import test as utils_test
    base_test()
    integration_test()
    misc_test()
    wrapper_test()
    data_test()
    control_test()
    costestimator_test()
    utils_test()

