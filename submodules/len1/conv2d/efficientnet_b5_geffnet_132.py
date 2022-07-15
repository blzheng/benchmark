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
        self.conv2d132 = Conv2d(1056, 176, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x394):
        x395=self.conv2d132(x394)
        return x395

m = M().eval()
x394 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x394)
end = time.time()
print(end-start)
