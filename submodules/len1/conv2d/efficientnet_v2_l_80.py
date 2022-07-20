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
        self.conv2d80 = Conv2d(768, 48, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x266):
        x267=self.conv2d80(x266)
        return x267

m = M().eval()
x266 = torch.randn(torch.Size([1, 768, 1, 1]))
start = time.time()
output = m(x266)
end = time.time()
print(end-start)