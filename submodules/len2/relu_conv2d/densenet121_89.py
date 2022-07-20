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
        self.relu90 = ReLU(inplace=True)
        self.conv2d90 = Conv2d(544, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x322):
        x323=self.relu90(x322)
        x324=self.conv2d90(x323)
        return x324

m = M().eval()
x322 = torch.randn(torch.Size([1, 544, 7, 7]))
start = time.time()
output = m(x322)
end = time.time()
print(end-start)
