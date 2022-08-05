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
        self.relu91 = ReLU(inplace=True)
        self.conv2d98 = Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x320, x322):
        x323=operator.add(x320, x322)
        x324=self.relu91(x323)
        x325=self.conv2d98(x324)
        return x325

m = M().eval()
x320 = torch.randn(torch.Size([1, 2048, 28, 28]))
x322 = torch.randn(torch.Size([1, 2048, 28, 28]))
start = time.time()
output = m(x320, x322)
end = time.time()
print(end-start)
