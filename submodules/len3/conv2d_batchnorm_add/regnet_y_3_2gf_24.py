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
        self.conv2d109 = Conv2d(1512, 1512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d67 = BatchNorm2d(1512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x343, x331):
        x344=self.conv2d109(x343)
        x345=self.batchnorm2d67(x344)
        x346=operator.add(x331, x345)
        return x346

m = M().eval()
x343 = torch.randn(torch.Size([1, 1512, 7, 7]))
x331 = torch.randn(torch.Size([1, 1512, 7, 7]))
start = time.time()
output = m(x343, x331)
end = time.time()
print(end-start)
