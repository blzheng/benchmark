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
        self.relu139 = ReLU(inplace=True)
        self.conv2d144 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x475):
        x476=self.relu139(x475)
        x477=self.conv2d144(x476)
        return x477

m = M().eval()
x475 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x475)
end = time.time()
print(end-start)
