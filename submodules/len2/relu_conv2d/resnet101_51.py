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
        self.relu52 = ReLU(inplace=True)
        self.conv2d56 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x182):
        x183=self.relu52(x182)
        x184=self.conv2d56(x183)
        return x184

m = M().eval()
x182 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x182)
end = time.time()
print(end-start)