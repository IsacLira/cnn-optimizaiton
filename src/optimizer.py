import scipy.io
from scipy.io import loadmat
import numpy as np
import pandas as pd

import torch
from torch import Tensor
from torch.autograd import Variable
from torch.nn import functional as F
from torch import nn

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score as auc,
                            accuracy_score as acc,
                            confusion_matrix)
from sklearn.model_selection import train_test_split as split

from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax import RangeParameter, SearchSpace, RangeParameter

#!pip install mpu
import mpu

# Necessary if you want consistent results
import random

# from matplotlib.ticker import MaxNLocator

