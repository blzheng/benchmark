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
        self.conv2d15 = Conv2d(48, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x35):
        x38=self.conv2d15(x35)
        return x38

m = M().eval()
x35 = torch.randn(torch.Size([1, 48, 13, 13]))
start = time.time()
output = m(x35)
end = time.time()
print(end-start)
