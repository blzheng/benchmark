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
        self.conv2d93 = Conv2d(144, 864, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d55 = BatchNorm2d(864, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x289, x274):
        x290=operator.add(x289, x274)
        x291=self.conv2d93(x290)
        x292=self.batchnorm2d55(x291)
        return x292

m = M().eval()
x289 = torch.randn(torch.Size([1, 144, 14, 14]))
x274 = torch.randn(torch.Size([1, 144, 14, 14]))
start = time.time()
output = m(x289, x274)
end = time.time()
print(end-start)
