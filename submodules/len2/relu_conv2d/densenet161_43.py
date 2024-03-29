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
        self.relu44 = ReLU(inplace=True)
        self.conv2d44 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x159):
        x160=self.relu44(x159)
        x161=self.conv2d44(x160)
        return x161

m = M().eval()
x159 = torch.randn(torch.Size([1, 192, 14, 14]))
start = time.time()
output = m(x159)
end = time.time()
print(end-start)
