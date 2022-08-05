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
        self.conv2d67 = Conv2d(1280, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x186, x189, x192, x195, x202):
        x203=torch.cat([x186, x189, x192, x195, x202],dim=1)
        x204=self.conv2d67(x203)
        return x204

m = M().eval()
x186 = torch.randn(torch.Size([1, 256, 14, 14]))
x189 = torch.randn(torch.Size([1, 256, 14, 14]))
x192 = torch.randn(torch.Size([1, 256, 14, 14]))
x195 = torch.randn(torch.Size([1, 256, 14, 14]))
x202 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x186, x189, x192, x195, x202)
end = time.time()
print(end-start)
