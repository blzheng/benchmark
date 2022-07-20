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
        self.relu89 = ReLU(inplace=True)
        self.conv2d89 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x318):
        x319=self.relu89(x318)
        x320=self.conv2d89(x319)
        return x320

m = M().eval()
x318 = torch.randn(torch.Size([1, 128, 7, 7]))
start = time.time()
output = m(x318)
end = time.time()
print(end-start)
