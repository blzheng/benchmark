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
        self.conv2d14 = Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384)

    def forward(self, x138, x128):
        x139=operator.add(x138, x128)
        x141=self.conv2d14(x139)
        return x141

m = M().eval()
x138 = torch.randn(torch.Size([1, 384, 14, 14]))
x128 = torch.randn(torch.Size([1, 384, 14, 14]))
start = time.time()
output = m(x138, x128)
end = time.time()
print(end-start)
