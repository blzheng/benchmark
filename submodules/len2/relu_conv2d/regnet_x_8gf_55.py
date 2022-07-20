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
        self.relu55 = ReLU(inplace=True)
        self.conv2d59 = Conv2d(720, 720, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=6, bias=False)

    def forward(self, x191):
        x192=self.relu55(x191)
        x193=self.conv2d59(x192)
        return x193

m = M().eval()
x191 = torch.randn(torch.Size([1, 720, 14, 14]))
start = time.time()
output = m(x191)
end = time.time()
print(end-start)
