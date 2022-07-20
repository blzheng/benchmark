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
        self.relu136 = ReLU(inplace=True)
        self.conv2d142 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d142 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu139 = ReLU(inplace=True)

    def forward(self, x469):
        x470=self.relu136(x469)
        x471=self.conv2d142(x470)
        x472=self.batchnorm2d142(x471)
        x473=self.relu139(x472)
        return x473

m = M().eval()
x469 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x469)
end = time.time()
print(end-start)
