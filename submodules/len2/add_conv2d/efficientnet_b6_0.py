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
        self.conv2d9 = Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)

    def forward(self, x26, x14):
        x27=operator.add(x26, x14)
        x28=self.conv2d9(x27)
        return x28

m = M().eval()
x26 = torch.randn(torch.Size([1, 32, 112, 112]))
x14 = torch.randn(torch.Size([1, 32, 112, 112]))
start = time.time()
output = m(x26, x14)
end = time.time()
print(end-start)
