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
        self.conv2d80 = Conv2d(864, 36, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x240):
        x241=self.conv2d80(x240)
        return x241

m = M().eval()
x240 = torch.randn(torch.Size([1, 864, 1, 1]))
start = time.time()
output = m(x240)
end = time.time()
print(end-start)
