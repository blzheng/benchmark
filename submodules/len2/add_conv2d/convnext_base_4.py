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
        self.conv2d10 = Conv2d(512, 512, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=512)

    def forward(self, x94, x84):
        x95=operator.add(x94, x84)
        x97=self.conv2d10(x95)
        return x97

m = M().eval()
x94 = torch.randn(torch.Size([1, 512, 14, 14]))
x84 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x94, x84)
end = time.time()
print(end-start)
