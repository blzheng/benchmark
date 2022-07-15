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
        self.conv2d56 = Conv2d(576, 144, kernel_size=(1, 1), stride=(1, 1))
        self.relu43 = ReLU()

    def forward(self, x176):
        x177=self.conv2d56(x176)
        x178=self.relu43(x177)
        return x178

m = M().eval()
x176 = torch.randn(torch.Size([1, 576, 1, 1]))
start = time.time()
output = m(x176)
end = time.time()
print(end-start)
