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
        self.conv2d55 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d55 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x182):
        x183=self.conv2d55(x182)
        x184=self.batchnorm2d55(x183)
        return x184

m = M().eval()
x182 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x182)
end = time.time()
print(end-start)
