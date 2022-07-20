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
        self.relu91 = ReLU(inplace=True)
        self.conv2d98 = Conv2d(2048, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d98 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x318, x320):
        x321=operator.add(x318, x320)
        x322=self.relu91(x321)
        x323=self.conv2d98(x322)
        x324=self.batchnorm2d98(x323)
        return x324

m = M().eval()
x318 = torch.randn(torch.Size([1, 2048, 7, 7]))
x320 = torch.randn(torch.Size([1, 2048, 7, 7]))
start = time.time()
output = m(x318, x320)
end = time.time()
print(end-start)
