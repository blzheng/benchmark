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
        self.batchnorm2d90 = BatchNorm2d(544, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu90 = ReLU(inplace=True)
        self.conv2d90 = Conv2d(544, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x321):
        x322=self.batchnorm2d90(x321)
        x323=self.relu90(x322)
        x324=self.conv2d90(x323)
        return x324

m = M().eval()
x321 = torch.randn(torch.Size([1, 544, 7, 7]))
start = time.time()
output = m(x321)
end = time.time()
print(end-start)
