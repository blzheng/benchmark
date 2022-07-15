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
        self.conv2d19 = Conv2d(72, 72, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=72, bias=False)

    def forward(self, x54):
        x55=self.conv2d19(x54)
        return x55

m = M().eval()
x54 = torch.randn(torch.Size([1, 72, 28, 28]))
start = time.time()
output = m(x54)
end = time.time()
print(end-start)
