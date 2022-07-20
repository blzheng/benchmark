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
        self.conv2d98 = Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d98 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x322):
        x323=self.conv2d98(x322)
        x324=self.batchnorm2d98(x323)
        return x324

m = M().eval()
x322 = torch.randn(torch.Size([1, 2048, 7, 7]))
start = time.time()
output = m(x322)
end = time.time()
print(end-start)
