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
        self.conv2d249 = Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x745):
        x746=x745.mean((2, 3),keepdim=True)
        x747=self.conv2d249(x746)
        return x747

m = M().eval()
x745 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x745)
end = time.time()
print(end-start)
