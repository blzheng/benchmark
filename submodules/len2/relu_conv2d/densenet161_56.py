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
        self.relu57 = ReLU(inplace=True)
        self.conv2d57 = Conv2d(816, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x205):
        x206=self.relu57(x205)
        x207=self.conv2d57(x206)
        return x207

m = M().eval()
x205 = torch.randn(torch.Size([1, 816, 14, 14]))
start = time.time()
output = m(x205)
end = time.time()
print(end-start)