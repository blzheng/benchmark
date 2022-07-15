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
        self.conv2d101 = Conv2d(36, 864, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x302):
        x303=self.conv2d101(x302)
        return x303

m = M().eval()
x302 = torch.randn(torch.Size([1, 36, 1, 1]))
start = time.time()
output = m(x302)
end = time.time()
print(end-start)
