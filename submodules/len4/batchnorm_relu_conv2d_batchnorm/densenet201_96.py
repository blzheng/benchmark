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
        self.batchnorm2d196 = BatchNorm2d(1856, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu196 = ReLU(inplace=True)
        self.conv2d196 = Conv2d(1856, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d197 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x692):
        x693=self.batchnorm2d196(x692)
        x694=self.relu196(x693)
        x695=self.conv2d196(x694)
        x696=self.batchnorm2d197(x695)
        return x696

m = M().eval()
x692 = torch.randn(torch.Size([1, 1856, 7, 7]))
start = time.time()
output = m(x692)
end = time.time()
print(end-start)
