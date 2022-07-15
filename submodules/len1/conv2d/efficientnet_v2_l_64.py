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
        self.conv2d64 = Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)

    def forward(self, x214):
        x215=self.conv2d64(x214)
        return x215

m = M().eval()
x214 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x214)
end = time.time()
print(end-start)
