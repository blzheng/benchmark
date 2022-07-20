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
        self.relu54 = ReLU(inplace=True)
        self.conv2d54 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x194):
        x195=self.relu54(x194)
        x196=self.conv2d54(x195)
        return x196

m = M().eval()
x194 = torch.randn(torch.Size([1, 192, 14, 14]))
start = time.time()
output = m(x194)
end = time.time()
print(end-start)
