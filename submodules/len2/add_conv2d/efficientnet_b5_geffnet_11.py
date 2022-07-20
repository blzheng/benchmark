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
        self.conv2d78 = Conv2d(128, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x232, x218):
        x233=operator.add(x232, x218)
        x234=self.conv2d78(x233)
        return x234

m = M().eval()
x232 = torch.randn(torch.Size([1, 128, 14, 14]))
x218 = torch.randn(torch.Size([1, 128, 14, 14]))
start = time.time()
output = m(x232, x218)
end = time.time()
print(end-start)
