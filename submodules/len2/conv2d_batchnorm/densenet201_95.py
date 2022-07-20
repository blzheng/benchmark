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
        self.conv2d192 = Conv2d(1792, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d193 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x680):
        x681=self.conv2d192(x680)
        x682=self.batchnorm2d193(x681)
        return x682

m = M().eval()
x680 = torch.randn(torch.Size([1, 1792, 7, 7]))
start = time.time()
output = m(x680)
end = time.time()
print(end-start)
