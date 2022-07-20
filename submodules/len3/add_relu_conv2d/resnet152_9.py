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
        self.relu28 = ReLU(inplace=True)
        self.conv2d33 = Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x106, x98):
        x107=operator.add(x106, x98)
        x108=self.relu28(x107)
        x109=self.conv2d33(x108)
        return x109

m = M().eval()
x106 = torch.randn(torch.Size([1, 512, 28, 28]))
x98 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x106, x98)
end = time.time()
print(end-start)
