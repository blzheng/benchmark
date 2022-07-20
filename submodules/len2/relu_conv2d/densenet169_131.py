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

    def forward(self, x469):
        x470=self.relu132(x469)
        x471=self.conv2d132(x470)
        return x471

m = M().eval()
x469 = torch.randn(torch.Size([1, 1088, 7, 7]))
start = time.time()
output = m(x469)
end = time.time()
print(end-start)
