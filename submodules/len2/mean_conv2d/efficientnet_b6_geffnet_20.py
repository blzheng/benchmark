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
        self.conv2d100 = Conv2d(864, 36, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x299):
        x300=x299.mean((2, 3),keepdim=True)
        x301=self.conv2d100(x300)
        return x301

m = M().eval()
x299 = torch.randn(torch.Size([1, 864, 14, 14]))
start = time.time()
output = m(x299)
end = time.time()
print(end-start)
