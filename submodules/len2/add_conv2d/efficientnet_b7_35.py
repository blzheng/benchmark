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
        self.conv2d207 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x650, x635):
        x651=operator.add(x650, x635)
        x652=self.conv2d207(x651)
        return x652

m = M().eval()
x650 = torch.randn(torch.Size([1, 384, 7, 7]))
x635 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x650, x635)
end = time.time()
print(end-start)
