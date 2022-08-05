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
        self.conv2d68 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x207):
        x208=self.conv2d68(x207)
        return x208

m = M().eval()
x207 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x207)
end = time.time()
print(end-start)
