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
        self.conv2d75 = Conv2d(768, 32, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x225):
        x226=self.conv2d75(x225)
        return x226

m = M().eval()
x225 = torch.randn(torch.Size([1, 768, 1, 1]))
start = time.time()
output = m(x225)
end = time.time()
print(end-start)
