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
        self.conv2d80 = Conv2d(864, 36, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x239):
        x240=x239.mean((2, 3),keepdim=True)
        x241=self.conv2d80(x240)
        return x241

m = M().eval()
x239 = torch.randn(torch.Size([1, 864, 14, 14]))
start = time.time()
output = m(x239)
end = time.time()
print(end-start)
