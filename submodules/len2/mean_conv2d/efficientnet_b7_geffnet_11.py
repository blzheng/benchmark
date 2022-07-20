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
        self.conv2d54 = Conv2d(288, 12, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x164):
        x165=x164.mean((2, 3),keepdim=True)
        x166=self.conv2d54(x165)
        return x166

m = M().eval()
x164 = torch.randn(torch.Size([1, 288, 28, 28]))
start = time.time()
output = m(x164)
end = time.time()
print(end-start)
