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
        self.conv2d4 = Conv2d(224, 8, kernel_size=(1, 1), stride=(1, 1))
        self.relu3 = ReLU()

    def forward(self, x12):
        x13=self.conv2d4(x12)
        x14=self.relu3(x13)
        return x14

m = M().eval()
x12 = torch.randn(torch.Size([1, 224, 1, 1]))
start = time.time()
output = m(x12)
end = time.time()
print(end-start)
