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
        self.relu147 = ReLU(inplace=True)
        self.conv2d147 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x521):
        x522=self.relu147(x521)
        x523=self.conv2d147(x522)
        return x523

m = M().eval()
x521 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x521)
end = time.time()
print(end-start)
