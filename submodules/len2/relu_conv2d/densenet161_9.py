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
        self.relu10 = ReLU(inplace=True)
        self.conv2d10 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x37):
        x38=self.relu10(x37)
        x39=self.conv2d10(x38)
        return x39

m = M().eval()
x37 = torch.randn(torch.Size([1, 192, 56, 56]))
start = time.time()
output = m(x37)
end = time.time()
print(end-start)
