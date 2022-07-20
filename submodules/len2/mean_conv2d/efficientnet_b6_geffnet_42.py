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
        self.conv2d210 = Conv2d(2064, 86, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x627):
        x628=x627.mean((2, 3),keepdim=True)
        x629=self.conv2d210(x628)
        return x629

m = M().eval()
x627 = torch.randn(torch.Size([1, 2064, 7, 7]))
start = time.time()
output = m(x627)
end = time.time()
print(end-start)
