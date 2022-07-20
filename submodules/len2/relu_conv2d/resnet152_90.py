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
        self.relu91 = ReLU(inplace=True)
        self.conv2d95 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x312):
        x313=self.relu91(x312)
        x314=self.conv2d95(x313)
        return x314

m = M().eval()
x312 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x312)
end = time.time()
print(end-start)
