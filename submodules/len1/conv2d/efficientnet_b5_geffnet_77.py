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
        self.conv2d77 = Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x230):
        x231=self.conv2d77(x230)
        return x231

m = M().eval()
x230 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x230)
end = time.time()
print(end-start)
