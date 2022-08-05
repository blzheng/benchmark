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
        self.relu49 = ReLU(inplace=True)
        self.conv2d55 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x181):
        x182=self.relu49(x181)
        x183=self.conv2d55(x182)
        return x183

m = M().eval()
x181 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x181)
end = time.time()
print(end-start)
