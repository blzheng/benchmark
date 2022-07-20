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
        self.conv2d11 = Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)

    def forward(self, x105, x95):
        x106=operator.add(x105, x95)
        x108=self.conv2d11(x106)
        return x108

m = M().eval()
x105 = torch.randn(torch.Size([1, 768, 14, 14]))
x95 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x105, x95)
end = time.time()
print(end-start)
