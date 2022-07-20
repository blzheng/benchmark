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
        self.conv2d29 = Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), bias=False)

    def forward(self, x107):
        x108=torch.nn.functional.relu(x107,inplace=True)
        x109=self.conv2d29(x108)
        return x109

m = M().eval()
x107 = torch.randn(torch.Size([1, 96, 25, 25]))
start = time.time()
output = m(x107)
end = time.time()
print(end-start)
