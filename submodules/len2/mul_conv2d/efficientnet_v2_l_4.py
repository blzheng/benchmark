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
        self.conv2d57 = Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x190, x185):
        x191=operator.mul(x190, x185)
        x192=self.conv2d57(x191)
        return x192

m = M().eval()
x190 = torch.randn(torch.Size([1, 768, 1, 1]))
x185 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x190, x185)
end = time.time()
print(end-start)
