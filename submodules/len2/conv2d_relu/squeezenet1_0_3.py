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
        self.conv2d3 = Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3 = ReLU(inplace=True)

    def forward(self, x5):
        x8=self.conv2d3(x5)
        x9=self.relu3(x8)
        return x9

m = M().eval()
x5 = torch.randn(torch.Size([1, 16, 54, 54]))
start = time.time()
output = m(x5)
end = time.time()
print(end-start)
