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

    def forward(self, x468, x460):
        x469=operator.add(x468, x460)
        x470=self.relu136(x469)
        x471=self.conv2d142(x470)
        return x471

m = M().eval()
x468 = torch.randn(torch.Size([1, 1024, 14, 14]))
x460 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x468, x460)
end = time.time()
print(end-start)
