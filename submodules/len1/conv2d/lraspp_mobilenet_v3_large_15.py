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
        self.conv2d15 = Conv2d(120, 120, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=120, bias=False)

    def forward(self, x43):
        x44=self.conv2d15(x43)
        return x44

m = M().eval()
x43 = torch.randn(torch.Size([1, 120, 28, 28]))
start = time.time()
output = m(x43)
end = time.time()
print(end-start)
