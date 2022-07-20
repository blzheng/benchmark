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
        self.conv2d67 = Conv2d(1056, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d68 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x241):
        x242=self.conv2d67(x241)
        x243=self.batchnorm2d68(x242)
        return x243

m = M().eval()
x241 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x241)
end = time.time()
print(end-start)
