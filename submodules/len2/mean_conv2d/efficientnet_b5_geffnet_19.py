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
        self.conv2d95 = Conv2d(768, 32, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x284):
        x285=x284.mean((2, 3),keepdim=True)
        x286=self.conv2d95(x285)
        return x286

m = M().eval()
x284 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x284)
end = time.time()
print(end-start)
