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
        self.conv2d32 = Conv2d(512, 512, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=512)

    def forward(self, x336, x326):
        x337=operator.add(x336, x326)
        x339=self.conv2d32(x337)
        return x339

m = M().eval()
x336 = torch.randn(torch.Size([1, 512, 14, 14]))
x326 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x336, x326)
end = time.time()
print(end-start)
