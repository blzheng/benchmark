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
        self.relu92 = ReLU(inplace=True)
        self.conv2d119 = Conv2d(336, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x361, x375):
        x376=operator.add(x361, x375)
        x377=self.relu92(x376)
        x378=self.conv2d119(x377)
        return x378

m = M().eval()
x361 = torch.randn(torch.Size([1, 336, 14, 14]))
x375 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x361, x375)
end = time.time()
print(end-start)
