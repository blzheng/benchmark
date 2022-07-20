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
        self.conv2d203 = Conv2d(344, 2064, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d121 = BatchNorm2d(2064, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x605, x591):
        x606=operator.add(x605, x591)
        x607=self.conv2d203(x606)
        x608=self.batchnorm2d121(x607)
        return x608

m = M().eval()
x605 = torch.randn(torch.Size([1, 344, 7, 7]))
x591 = torch.randn(torch.Size([1, 344, 7, 7]))
start = time.time()
output = m(x605, x591)
end = time.time()
print(end-start)
