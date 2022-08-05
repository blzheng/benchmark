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
        self.conv2d104 = Conv2d(2048, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x344):
        x345=self.conv2d104(x344)
        return x345

m = M().eval()
x344 = torch.randn(torch.Size([1, 2048, 28, 28]))
start = time.time()
output = m(x344)
end = time.time()
print(end-start)
