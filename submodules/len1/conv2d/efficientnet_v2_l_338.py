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
        self.conv2d338 = Conv2d(640, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x1085):
        x1086=self.conv2d338(x1085)
        return x1086

m = M().eval()
x1085 = torch.randn(torch.Size([1, 640, 7, 7]))
start = time.time()
output = m(x1085)
end = time.time()
print(end-start)
