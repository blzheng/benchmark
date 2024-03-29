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
        self.conv2d130 = Conv2d(1536, 1536, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1536, bias=False)

    def forward(self, x414):
        x415=self.conv2d130(x414)
        return x415

m = M().eval()
x414 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x414)
end = time.time()
print(end-start)
