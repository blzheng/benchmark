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
        self.conv2d195 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x690):
        x691=self.conv2d195(x690)
        return x691

m = M().eval()
x690 = torch.randn(torch.Size([1, 128, 7, 7]))
start = time.time()
output = m(x690)
end = time.time()
print(end-start)
