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
        self.conv2d223 = Conv2d(576, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x699, x684):
        x700=operator.add(x699, x684)
        x701=self.conv2d223(x700)
        return x701

m = M().eval()
x699 = torch.randn(torch.Size([1, 576, 7, 7]))
x684 = torch.randn(torch.Size([1, 576, 7, 7]))
start = time.time()
output = m(x699, x684)
end = time.time()
print(end-start)
