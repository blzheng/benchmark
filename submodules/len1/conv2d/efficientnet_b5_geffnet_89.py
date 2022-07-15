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
        self.conv2d89 = Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)

    def forward(self, x266):
        x267=self.conv2d89(x266)
        return x267

m = M().eval()
x266 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x266)
end = time.time()
print(end-start)
