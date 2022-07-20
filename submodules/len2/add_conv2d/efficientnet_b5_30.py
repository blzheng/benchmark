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
        self.conv2d188 = Conv2d(512, 3072, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x587, x572):
        x588=operator.add(x587, x572)
        x589=self.conv2d188(x588)
        return x589

m = M().eval()
x587 = torch.randn(torch.Size([1, 512, 7, 7]))
x572 = torch.randn(torch.Size([1, 512, 7, 7]))
start = time.time()
output = m(x587, x572)
end = time.time()
print(end-start)
