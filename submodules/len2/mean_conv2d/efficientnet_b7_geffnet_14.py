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
        self.conv2d69 = Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x208):
        x209=x208.mean((2, 3),keepdim=True)
        x210=self.conv2d69(x209)
        return x210

m = M().eval()
x208 = torch.randn(torch.Size([1, 480, 28, 28]))
start = time.time()
output = m(x208)
end = time.time()
print(end-start)
