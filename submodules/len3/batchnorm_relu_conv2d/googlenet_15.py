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
        self.batchnorm2d46 = BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d47 = Conv2d(160, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x170):
        x171=self.batchnorm2d46(x170)
        x172=torch.nn.functional.relu(x171,inplace=True)
        x173=self.conv2d47(x172)
        return x173

m = M().eval()
x170 = torch.randn(torch.Size([1, 160, 7, 7]))
start = time.time()
output = m(x170)
end = time.time()
print(end-start)
