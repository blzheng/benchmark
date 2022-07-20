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
        self.conv2d146 = Conv2d(1632, 68, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x434):
        x435=x434.mean((2, 3),keepdim=True)
        x436=self.conv2d146(x435)
        return x436

m = M().eval()
x434 = torch.randn(torch.Size([1, 1632, 7, 7]))
start = time.time()
output = m(x434)
end = time.time()
print(end-start)
