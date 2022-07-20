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
        self.conv2d31 = Conv2d(208, 52, kernel_size=(1, 1), stride=(1, 1))
        self.relu23 = ReLU()
        self.conv2d32 = Conv2d(52, 208, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid5 = Sigmoid()

    def forward(self, x96):
        x97=self.conv2d31(x96)
        x98=self.relu23(x97)
        x99=self.conv2d32(x98)
        x100=self.sigmoid5(x99)
        return x100

m = M().eval()
x96 = torch.randn(torch.Size([1, 208, 1, 1]))
start = time.time()
output = m(x96)
end = time.time()
print(end-start)
