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
        self.conv2d253 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d165 = BatchNorm2d(2304, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x814, x799):
        x815=operator.add(x814, x799)
        x816=self.conv2d253(x815)
        x817=self.batchnorm2d165(x816)
        return x817

m = M().eval()
x814 = torch.randn(torch.Size([1, 384, 7, 7]))
x799 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x814, x799)
end = time.time()
print(end-start)
