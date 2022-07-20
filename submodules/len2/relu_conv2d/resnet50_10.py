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
        self.relu10 = ReLU(inplace=True)
        self.conv2d13 = Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x41):
        x42=self.relu10(x41)
        x43=self.conv2d13(x42)
        return x43

m = M().eval()
x41 = torch.randn(torch.Size([1, 128, 28, 28]))
start = time.time()
output = m(x41)
end = time.time()
print(end-start)