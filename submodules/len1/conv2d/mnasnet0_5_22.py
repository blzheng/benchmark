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
        self.conv2d22 = Conv2d(144, 144, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=144, bias=False)

    def forward(self, x63):
        x64=self.conv2d22(x63)
        return x64

m = M().eval()
x63 = torch.randn(torch.Size([1, 144, 28, 28]))
start = time.time()
output = m(x63)
end = time.time()
print(end-start)
