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
        self.conv2d86 = Conv2d(2016, 2016, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=36, bias=False)

    def forward(self, x270):
        x271=self.conv2d86(x270)
        return x271

m = M().eval()
x270 = torch.randn(torch.Size([1, 2016, 14, 14]))
start = time.time()
output = m(x270)
end = time.time()
print(end-start)
