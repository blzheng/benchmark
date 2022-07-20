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
        self.conv2d217 = Conv2d(3456, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x641, x646):
        x647=operator.mul(x641, x646)
        x648=self.conv2d217(x647)
        return x648

m = M().eval()
x641 = torch.randn(torch.Size([1, 3456, 7, 7]))
x646 = torch.randn(torch.Size([1, 3456, 1, 1]))
start = time.time()
output = m(x641, x646)
end = time.time()
print(end-start)
