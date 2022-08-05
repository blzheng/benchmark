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
        self.conv2d15 = Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x49):
        x50=self.relu10(x49)
        x51=self.conv2d15(x50)
        return x51

m = M().eval()
x49 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x49)
end = time.time()
print(end-start)
