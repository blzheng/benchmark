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
        self.conv2d108 = Conv2d(144, 1512, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid20 = Sigmoid()

    def forward(self, x340, x337):
        x341=self.conv2d108(x340)
        x342=self.sigmoid20(x341)
        x343=operator.mul(x342, x337)
        return x343

m = M().eval()
x340 = torch.randn(torch.Size([1, 144, 1, 1]))
x337 = torch.randn(torch.Size([1, 1512, 7, 7]))
start = time.time()
output = m(x340, x337)
end = time.time()
print(end-start)
