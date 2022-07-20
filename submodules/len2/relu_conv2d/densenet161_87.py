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
        self.relu88 = ReLU(inplace=True)
        self.conv2d88 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x313):
        x314=self.relu88(x313)
        x315=self.conv2d88(x314)
        return x315

m = M().eval()
x313 = torch.randn(torch.Size([1, 192, 14, 14]))
start = time.time()
output = m(x313)
end = time.time()
print(end-start)
