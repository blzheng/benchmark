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
        self.conv2d10 = Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
        self.relu10 = ReLU(inplace=True)
        self.conv2d11 = Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
        self.relu11 = ReLU(inplace=True)

    def forward(self, x25):
        x26=self.conv2d10(x25)
        x27=self.relu10(x26)
        x28=self.conv2d11(x27)
        x29=self.relu11(x28)
        return x29

m = M().eval()
x25 = torch.randn(torch.Size([1, 256, 27, 27]))
start = time.time()
output = m(x25)
end = time.time()
print(end-start)
