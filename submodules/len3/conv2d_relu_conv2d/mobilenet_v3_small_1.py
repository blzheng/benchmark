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
        self.conv2d13 = Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1))
        self.relu6 = ReLU()
        self.conv2d14 = Conv2d(24, 96, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x38):
        x39=self.conv2d13(x38)
        x40=self.relu6(x39)
        x41=self.conv2d14(x40)
        return x41

m = M().eval()
x38 = torch.randn(torch.Size([1, 96, 1, 1]))
start = time.time()
output = m(x38)
end = time.time()
print(end-start)
