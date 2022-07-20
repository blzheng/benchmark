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
        self.conv2d24 = Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d24 = BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x91):
        x92=self.conv2d24(x91)
        x93=self.batchnorm2d24(x92)
        x94=torch.nn.functional.relu(x93,inplace=True)
        return x94

m = M().eval()
x91 = torch.randn(torch.Size([1, 96, 25, 25]))
start = time.time()
output = m(x91)
end = time.time()
print(end-start)
