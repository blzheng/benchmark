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
        self.conv2d37 = Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x111):
        x112=self.conv2d37(x111)
        return x112

m = M().eval()
x111 = torch.randn(torch.Size([1, 480, 1, 1]))
start = time.time()
output = m(x111)
end = time.time()
print(end-start)