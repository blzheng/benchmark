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
        self.conv2d126 = Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x373):
        x374=x373.mean((2, 3),keepdim=True)
        x375=self.conv2d126(x374)
        return x375

m = M().eval()
x373 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x373)
end = time.time()
print(end-start)
