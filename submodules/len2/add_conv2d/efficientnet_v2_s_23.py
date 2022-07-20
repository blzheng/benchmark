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
        self.conv2d114 = Conv2d(256, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x362, x347):
        x363=operator.add(x362, x347)
        x364=self.conv2d114(x363)
        return x364

m = M().eval()
x362 = torch.randn(torch.Size([1, 256, 7, 7]))
x347 = torch.randn(torch.Size([1, 256, 7, 7]))
start = time.time()
output = m(x362, x347)
end = time.time()
print(end-start)
