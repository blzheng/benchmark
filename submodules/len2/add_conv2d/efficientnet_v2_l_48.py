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
        self.conv2d213 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x686, x671):
        x687=operator.add(x686, x671)
        x688=self.conv2d213(x687)
        return x688

m = M().eval()
x686 = torch.randn(torch.Size([1, 384, 7, 7]))
x671 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x686, x671)
end = time.time()
print(end-start)
