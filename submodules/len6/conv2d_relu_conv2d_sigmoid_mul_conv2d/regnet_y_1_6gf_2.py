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
        self.conv2d15 = Conv2d(120, 12, kernel_size=(1, 1), stride=(1, 1))
        self.relu11 = ReLU()
        self.conv2d16 = Conv2d(12, 120, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid2 = Sigmoid()
        self.conv2d17 = Conv2d(120, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x46, x45):
        x47=self.conv2d15(x46)
        x48=self.relu11(x47)
        x49=self.conv2d16(x48)
        x50=self.sigmoid2(x49)
        x51=operator.mul(x50, x45)
        x52=self.conv2d17(x51)
        return x52

m = M().eval()
x46 = torch.randn(torch.Size([1, 120, 1, 1]))
x45 = torch.randn(torch.Size([1, 120, 28, 28]))
start = time.time()
output = m(x46, x45)
end = time.time()
print(end-start)
