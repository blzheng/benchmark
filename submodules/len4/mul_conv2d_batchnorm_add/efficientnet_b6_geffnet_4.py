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
        self.conv2d207 = Conv2d(2064, 344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d123 = BatchNorm2d(344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x612, x617, x606):
        x618=operator.mul(x612, x617)
        x619=self.conv2d207(x618)
        x620=self.batchnorm2d123(x619)
        x621=operator.add(x620, x606)
        return x621

m = M().eval()
x612 = torch.randn(torch.Size([1, 2064, 7, 7]))
x617 = torch.randn(torch.Size([1, 2064, 1, 1]))
x606 = torch.randn(torch.Size([1, 344, 7, 7]))
start = time.time()
output = m(x612, x617, x606)
end = time.time()
print(end-start)
