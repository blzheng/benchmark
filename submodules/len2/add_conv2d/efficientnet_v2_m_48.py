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
        self.conv2d234 = Conv2d(512, 3072, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x747, x732):
        x748=operator.add(x747, x732)
        x749=self.conv2d234(x748)
        return x749

m = M().eval()
x747 = torch.randn(torch.Size([1, 512, 7, 7]))
x732 = torch.randn(torch.Size([1, 512, 7, 7]))
start = time.time()
output = m(x747, x732)
end = time.time()
print(end-start)
