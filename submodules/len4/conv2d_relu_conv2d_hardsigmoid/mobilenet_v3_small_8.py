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
        self.conv2d48 = Conv2d(576, 144, kernel_size=(1, 1), stride=(1, 1))
        self.relu13 = ReLU()
        self.conv2d49 = Conv2d(144, 576, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid8 = Hardsigmoid()

    def forward(self, x140):
        x141=self.conv2d48(x140)
        x142=self.relu13(x141)
        x143=self.conv2d49(x142)
        x144=self.hardsigmoid8(x143)
        return x144

m = M().eval()
x140 = torch.randn(torch.Size([1, 576, 1, 1]))
start = time.time()
output = m(x140)
end = time.time()
print(end-start)
