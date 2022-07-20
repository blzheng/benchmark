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
        self.relu132 = ReLU(inplace=True)
        self.conv2d132 = Conv2d(1088, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d133 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu133 = ReLU(inplace=True)
        self.conv2d133 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x469):
        x470=self.relu132(x469)
        x471=self.conv2d132(x470)
        x472=self.batchnorm2d133(x471)
        x473=self.relu133(x472)
        x474=self.conv2d133(x473)
        return x474

m = M().eval()
x469 = torch.randn(torch.Size([1, 1088, 7, 7]))
start = time.time()
output = m(x469)
end = time.time()
print(end-start)
