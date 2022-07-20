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
        self.conv2d136 = Conv2d(44, 1056, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid27 = Sigmoid()

    def forward(self, x425, x422):
        x426=self.conv2d136(x425)
        x427=self.sigmoid27(x426)
        x428=operator.mul(x427, x422)
        return x428

m = M().eval()
x425 = torch.randn(torch.Size([1, 44, 1, 1]))
x422 = torch.randn(torch.Size([1, 1056, 7, 7]))
start = time.time()
output = m(x425, x422)
end = time.time()
print(end-start)
