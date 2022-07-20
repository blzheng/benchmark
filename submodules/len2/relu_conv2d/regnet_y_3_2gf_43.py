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
        self.relu57 = ReLU(inplace=True)
        self.conv2d75 = Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False)

    def forward(self, x235):
        x236=self.relu57(x235)
        x237=self.conv2d75(x236)
        return x237

m = M().eval()
x235 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x235)
end = time.time()
print(end-start)
