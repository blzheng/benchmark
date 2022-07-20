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
        self.conv2d62 = Conv2d(224, 896, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid11 = Sigmoid()

    def forward(self, x194):
        x195=self.conv2d62(x194)
        x196=self.sigmoid11(x195)
        return x196

m = M().eval()
x194 = torch.randn(torch.Size([1, 224, 1, 1]))
start = time.time()
output = m(x194)
end = time.time()
print(end-start)
