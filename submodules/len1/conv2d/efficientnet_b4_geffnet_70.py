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
        self.conv2d70 = Conv2d(672, 672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=672, bias=False)

    def forward(self, x208):
        x209=self.conv2d70(x208)
        return x209

m = M().eval()
x208 = torch.randn(torch.Size([1, 672, 14, 14]))
start = time.time()
output = m(x208)
end = time.time()
print(end-start)
