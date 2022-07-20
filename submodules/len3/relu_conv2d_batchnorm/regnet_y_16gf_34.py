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
        self.relu68 = ReLU(inplace=True)
        self.conv2d89 = Conv2d(1232, 3024, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d55 = BatchNorm2d(3024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x280):
        x281=self.relu68(x280)
        x282=self.conv2d89(x281)
        x283=self.batchnorm2d55(x282)
        return x283

m = M().eval()
x280 = torch.randn(torch.Size([1, 1232, 14, 14]))
start = time.time()
output = m(x280)
end = time.time()
print(end-start)
