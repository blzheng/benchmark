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
        self.conv2d113 = Conv2d(2112, 352, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d67 = BatchNorm2d(352, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x334, x322):
        x335=self.conv2d113(x334)
        x336=self.batchnorm2d67(x335)
        x337=operator.add(x336, x322)
        return x337

m = M().eval()
x334 = torch.randn(torch.Size([1, 2112, 7, 7]))
x322 = torch.randn(torch.Size([1, 352, 7, 7]))
start = time.time()
output = m(x334, x322)
end = time.time()
print(end-start)
