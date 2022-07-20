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
        self.batchnorm2d133 = BatchNorm2d(1760, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu133 = ReLU(inplace=True)
        self.conv2d133 = Conv2d(1760, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x470):
        x471=self.batchnorm2d133(x470)
        x472=self.relu133(x471)
        x473=self.conv2d133(x472)
        return x473

m = M().eval()
x470 = torch.randn(torch.Size([1, 1760, 14, 14]))
start = time.time()
output = m(x470)
end = time.time()
print(end-start)
