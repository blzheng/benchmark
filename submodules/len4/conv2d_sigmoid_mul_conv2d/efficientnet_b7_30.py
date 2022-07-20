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
        self.conv2d150 = Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid30 = Sigmoid()
        self.conv2d151 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x470, x467):
        x471=self.conv2d150(x470)
        x472=self.sigmoid30(x471)
        x473=operator.mul(x472, x467)
        x474=self.conv2d151(x473)
        return x474

m = M().eval()
x470 = torch.randn(torch.Size([1, 56, 1, 1]))
x467 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x470, x467)
end = time.time()
print(end-start)
