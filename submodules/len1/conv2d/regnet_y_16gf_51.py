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
        self.conv2d51 = Conv2d(1232, 308, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x160):
        x161=self.conv2d51(x160)
        return x161

m = M().eval()
x160 = torch.randn(torch.Size([1, 1232, 1, 1]))
start = time.time()
output = m(x160)
end = time.time()
print(end-start)
