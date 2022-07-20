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
        self.conv2d21 = Conv2d(256, 16, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x72):
        x73=self.conv2d21(x72)
        return x73

m = M().eval()
x72 = torch.randn(torch.Size([1, 256, 1, 1]))
start = time.time()
output = m(x72)
end = time.time()
print(end-start)