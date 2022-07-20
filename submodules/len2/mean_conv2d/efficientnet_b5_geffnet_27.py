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
        self.conv2d135 = Conv2d(1056, 44, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x403):
        x404=x403.mean((2, 3),keepdim=True)
        x405=self.conv2d135(x404)
        return x405

m = M().eval()
x403 = torch.randn(torch.Size([1, 1056, 7, 7]))
start = time.time()
output = m(x403)
end = time.time()
print(end-start)
