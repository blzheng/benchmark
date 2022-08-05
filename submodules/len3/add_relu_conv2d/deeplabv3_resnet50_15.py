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
        self.relu46 = ReLU(inplace=True)
        self.conv2d53 = Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x172, x164):
        x173=operator.add(x172, x164)
        x174=self.relu46(x173)
        x175=self.conv2d53(x174)
        return x175

m = M().eval()
x172 = torch.randn(torch.Size([1, 2048, 28, 28]))
x164 = torch.randn(torch.Size([1, 2048, 28, 28]))
start = time.time()
output = m(x172, x164)
end = time.time()
print(end-start)
