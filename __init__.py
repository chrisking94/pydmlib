from outlierhandler import *
from factorencoder import *
from discretor import *
from featureclassifier import *
from featurecombiner import *
from featureselector import *
from misc import *
from integration import *
from nanhandler import *
from normalizer import *
from sampler import *
from transformer import *
from reporter import *
from formatconverter import *
from wrapper import *
from data import *
from costestimator import RSCostEstimator
import base
import misc
import wrapper
import data
import control
import integration
import costestimator

RSControl.init()
RSCostEstimator.init()  # initialize statically

base.test()
integration.test()
misc.test()
wrapper.test()
data.test()
control.test()
costestimator.test()

