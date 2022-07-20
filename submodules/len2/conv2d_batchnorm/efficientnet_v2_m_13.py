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
        self.conv2d13 = Conv2d(192, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d13 = BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x47):
        x48=self.conv2d13(x47)
        x49=self.batchnorm2d13(x48)
        return x49

m = M().eval()
x47 = torch.randn(torch.Size([1, 192, 56, 56]))
start = time.time()
output = m(x47)
end = time.time()
print(end-start)
