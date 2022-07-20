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
        self.conv2d52 = Conv2d(32, 768, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid6 = Sigmoid()

    def forward(self, x168):
        x169=self.conv2d52(x168)
        x170=self.sigmoid6(x169)
        return x170

m = M().eval()
x168 = torch.randn(torch.Size([1, 32, 1, 1]))
start = time.time()
output = m(x168)
end = time.time()
print(end-start)
