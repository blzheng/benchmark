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
        self.conv2d151 = Conv2d(50, 1200, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid30 = Sigmoid()
        self.conv2d152 = Conv2d(1200, 200, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x473, x470):
        x474=self.conv2d151(x473)
        x475=self.sigmoid30(x474)
        x476=operator.mul(x475, x470)
        x477=self.conv2d152(x476)
        return x477

m = M().eval()
x473 = torch.randn(torch.Size([1, 50, 1, 1]))
x470 = torch.randn(torch.Size([1, 1200, 14, 14]))
start = time.time()
output = m(x473, x470)
end = time.time()
print(end-start)
