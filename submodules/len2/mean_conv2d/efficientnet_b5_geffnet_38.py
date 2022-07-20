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
        self.conv2d190 = Conv2d(3072, 128, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x566):
        x567=x566.mean((2, 3),keepdim=True)
        x568=self.conv2d190(x567)
        return x568

m = M().eval()
x566 = torch.randn(torch.Size([1, 3072, 7, 7]))
start = time.time()
output = m(x566)
end = time.time()
print(end-start)
