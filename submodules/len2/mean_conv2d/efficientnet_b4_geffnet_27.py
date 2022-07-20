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
        self.conv2d136 = Conv2d(1632, 68, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x404):
        x405=x404.mean((2, 3),keepdim=True)
        x406=self.conv2d136(x405)
        return x406

m = M().eval()
x404 = torch.randn(torch.Size([1, 1632, 7, 7]))
start = time.time()
output = m(x404)
end = time.time()
print(end-start)
