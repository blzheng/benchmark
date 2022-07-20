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
        self.relu146 = ReLU(inplace=True)
        self.conv2d146 = Conv2d(1056, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x518):
        x519=self.relu146(x518)
        x520=self.conv2d146(x519)
        return x520

m = M().eval()
x518 = torch.randn(torch.Size([1, 1056, 7, 7]))
start = time.time()
output = m(x518)
end = time.time()
print(end-start)
