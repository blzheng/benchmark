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
        self.conv2d121 = Conv2d(1632, 68, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x359):
        x360=x359.mean((2, 3),keepdim=True)
        x361=self.conv2d121(x360)
        return x361

m = M().eval()
x359 = torch.randn(torch.Size([1, 1632, 7, 7]))
start = time.time()
output = m(x359)
end = time.time()
print(end-start)
