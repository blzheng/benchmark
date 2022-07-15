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
        self.conv2d111 = Conv2d(1920, 80, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x342):
        x343=self.conv2d111(x342)
        return x343

m = M().eval()
x342 = torch.randn(torch.Size([1, 1920, 1, 1]))
start = time.time()
output = m(x342)
end = time.time()
print(end-start)
