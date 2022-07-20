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
        self.conv2d21 = Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)

    def forward(self, x210, x200):
        x211=operator.add(x210, x200)
        x213=self.conv2d21(x211)
        return x213

m = M().eval()
x210 = torch.randn(torch.Size([1, 768, 7, 7]))
x200 = torch.randn(torch.Size([1, 768, 7, 7]))
start = time.time()
output = m(x210, x200)
end = time.time()
print(end-start)
