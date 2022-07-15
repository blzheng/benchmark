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
        self.conv2d336 = Conv2d(160, 3840, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x1078):
        x1079=self.conv2d336(x1078)
        return x1079

m = M().eval()
x1078 = torch.randn(torch.Size([1, 160, 1, 1]))
start = time.time()
output = m(x1078)
end = time.time()
print(end-start)
