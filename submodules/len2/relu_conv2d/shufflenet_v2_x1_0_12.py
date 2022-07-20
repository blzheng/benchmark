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
        self.relu28 = ReLU(inplace=True)
        self.conv2d44 = Conv2d(232, 232, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=232, bias=False)

    def forward(self, x279):
        x280=self.relu28(x279)
        x281=self.conv2d44(x280)
        return x281

m = M().eval()
x279 = torch.randn(torch.Size([1, 232, 14, 14]))
start = time.time()
output = m(x279)
end = time.time()
print(end-start)
