import torch
from torch import tensor
import torch.nn as nn
from torch.nn import *
import torchvision
import torchvision.models as models
from torchvision.ops.stochastic_depth import stochastic_depth
import time
import builtins
import operator

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.relu148 = ReLU(inplace=True)
        self.conv2d148 = Conv2d(1920, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d149 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu149 = ReLU(inplace=True)
        self.conv2d149 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x525):
        x526=self.relu148(x525)
        x527=self.conv2d148(x526)
        x528=self.batchnorm2d149(x527)
        x529=self.relu149(x528)
        x530=self.conv2d149(x529)
        return x530

m = M().eval()
x525 = torch.randn(torch.Size([1, 1920, 7, 7]))
start = time.time()
output = m(x525)
end = time.time()
print(end-start)
