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
        self.conv2d26 = Conv2d(208, 26, kernel_size=(1, 1), stride=(1, 1))
        self.relu19 = ReLU()

    def forward(self, x80):
        x81=self.conv2d26(x80)
        x82=self.relu19(x81)
        return x82

m = M().eval()
x80 = torch.randn(torch.Size([1, 208, 1, 1]))
start = time.time()
output = m(x80)
end = time.time()
print(end-start)
