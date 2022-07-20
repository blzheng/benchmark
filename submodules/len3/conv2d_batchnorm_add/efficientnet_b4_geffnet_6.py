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
        self.conv2d158 = Conv2d(2688, 448, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d94 = BatchNorm2d(448, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x469, x457):
        x470=self.conv2d158(x469)
        x471=self.batchnorm2d94(x470)
        x472=operator.add(x471, x457)
        return x472

m = M().eval()
x469 = torch.randn(torch.Size([1, 2688, 7, 7]))
x457 = torch.randn(torch.Size([1, 448, 7, 7]))
start = time.time()
output = m(x469, x457)
end = time.time()
print(end-start)
