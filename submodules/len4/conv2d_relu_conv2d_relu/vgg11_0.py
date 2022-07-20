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
        self.conv2d2 = Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu2 = ReLU(inplace=True)
        self.conv2d3 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3 = ReLU(inplace=True)

    def forward(self, x6):
        x7=self.conv2d2(x6)
        x8=self.relu2(x7)
        x9=self.conv2d3(x8)
        x10=self.relu3(x9)
        return x10

m = M().eval()
x6 = torch.randn(torch.Size([1, 128, 56, 56]))
start = time.time()
output = m(x6)
end = time.time()
print(end-start)
