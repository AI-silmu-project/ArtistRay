import os
from os.path import exists, join, basename, splitext

git_repo_url = 'https://github.com/foolwood/SiamMask.git'
project_name = splitext(basename(git_repo_url))[0]
  
import sys
sys.path.append(project_name)
sys.path.append(join(project_name, 'experiments', 'siammask_sharp'))
import time
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["axes.grid"] = False

from IPython.display import YouTubeVideo

import cv2
import torchvision
import cv2
import numpy as np
import torch
torch.set_grad_enabled(False)

from types import SimpleNamespace
from custom import Custom
from tools.test import *