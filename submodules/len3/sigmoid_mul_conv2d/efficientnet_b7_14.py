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
        self.sigmoid14 = Sigmoid()
        self.conv2d71 = Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x219, x215):
        x220=self.sigmoid14(x219)
        x221=operator.mul(x220, x215)
        x222=self.conv2d71(x221)
        return x222

m = M().eval()
x219 = torch.randn(torch.Size([1, 480, 1, 1]))
x215 = torch.randn(torch.Size([1, 480, 28, 28]))
start = time.time()
output = m(x219, x215)
end = time.time()
print(end-start)
