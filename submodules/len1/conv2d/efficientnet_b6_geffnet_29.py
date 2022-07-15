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
        self.conv2d29 = Conv2d(240, 240, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=240, bias=False)

    def forward(self, x88):
        x89=self.conv2d29(x88)
        return x89

m = M().eval()
x88 = torch.randn(torch.Size([1, 240, 56, 56]))
start = time.time()
output = m(x88)
end = time.time()
print(end-start)
