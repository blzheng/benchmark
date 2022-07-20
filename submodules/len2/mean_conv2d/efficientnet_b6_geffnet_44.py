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
        self.conv2d220 = Conv2d(3456, 144, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x656):
        x657=x656.mean((2, 3),keepdim=True)
        x658=self.conv2d220(x657)
        return x658

m = M().eval()
x656 = torch.randn(torch.Size([1, 3456, 7, 7]))
start = time.time()
output = m(x656)
end = time.time()
print(end-start)
