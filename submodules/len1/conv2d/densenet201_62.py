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
        self.conv2d62 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x223):
        x224=self.conv2d62(x223)
        return x224

m = M().eval()
x223 = torch.randn(torch.Size([1, 128, 14, 14]))
start = time.time()
output = m(x223)
end = time.time()
print(end-start)