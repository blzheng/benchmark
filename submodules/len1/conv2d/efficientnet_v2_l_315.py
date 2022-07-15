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
        self.conv2d315 = Conv2d(3840, 160, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x1012):
        x1013=self.conv2d315(x1012)
        return x1013

m = M().eval()
x1012 = torch.randn(torch.Size([1, 3840, 1, 1]))
start = time.time()
output = m(x1012)
end = time.time()
print(end-start)
