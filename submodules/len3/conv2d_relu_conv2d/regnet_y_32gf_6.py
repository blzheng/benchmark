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
        self.conv2d35 = Conv2d(696, 174, kernel_size=(1, 1), stride=(1, 1))
        self.relu27 = ReLU()
        self.conv2d36 = Conv2d(174, 696, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x110):
        x111=self.conv2d35(x110)
        x112=self.relu27(x111)
        x113=self.conv2d36(x112)
        return x113

m = M().eval()
x110 = torch.randn(torch.Size([1, 696, 1, 1]))
start = time.time()
output = m(x110)
end = time.time()
print(end-start)
