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
        self.conv2d52 = Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d30 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x156):
        x157=self.conv2d52(x156)
        x158=self.batchnorm2d30(x157)
        return x158

m = M().eval()
x156 = torch.randn(torch.Size([1, 384, 28, 28]))
start = time.time()
output = m(x156)
end = time.time()
print(end-start)
