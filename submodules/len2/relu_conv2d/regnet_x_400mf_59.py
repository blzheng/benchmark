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
        self.relu59 = ReLU(inplace=True)
        self.conv2d64 = Conv2d(400, 400, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x206):
        x207=self.relu59(x206)
        x208=self.conv2d64(x207)
        return x208

m = M().eval()
x206 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x206)
end = time.time()
print(end-start)
