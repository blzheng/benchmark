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
        self.conv2d67 = Conv2d(1280, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x203):
        x204=self.conv2d67(x203)
        return x204

m = M().eval()
x203 = torch.randn(torch.Size([1, 1280, 14, 14]))
start = time.time()
output = m(x203)
end = time.time()
print(end-start)
