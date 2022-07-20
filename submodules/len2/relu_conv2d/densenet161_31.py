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
        self.relu32 = ReLU(inplace=True)
        self.conv2d32 = Conv2d(624, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x116):
        x117=self.relu32(x116)
        x118=self.conv2d32(x117)
        return x118

m = M().eval()
x116 = torch.randn(torch.Size([1, 624, 28, 28]))
start = time.time()
output = m(x116)
end = time.time()
print(end-start)
