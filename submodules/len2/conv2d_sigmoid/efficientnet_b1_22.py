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
        self.conv2d112 = Conv2d(80, 1920, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid22 = Sigmoid()

    def forward(self, x344):
        x345=self.conv2d112(x344)
        x346=self.sigmoid22(x345)
        return x346

m = M().eval()
x344 = torch.randn(torch.Size([1, 80, 1, 1]))
start = time.time()
output = m(x344)
end = time.time()
print(end-start)
