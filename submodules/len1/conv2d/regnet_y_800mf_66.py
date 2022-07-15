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
        self.conv2d66 = Conv2d(784, 784, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=49, bias=False)

    def forward(self, x206):
        x207=self.conv2d66(x206)
        return x207

m = M().eval()
x206 = torch.randn(torch.Size([1, 784, 14, 14]))
start = time.time()
output = m(x206)
end = time.time()
print(end-start)
