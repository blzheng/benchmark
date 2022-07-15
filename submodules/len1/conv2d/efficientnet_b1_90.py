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
        self.conv2d90 = Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)

    def forward(self, x276):
        x277=self.conv2d90(x276)
        return x277

m = M().eval()
x276 = torch.randn(torch.Size([1, 1152, 7, 7]))
start = time.time()
output = m(x276)
end = time.time()
print(end-start)
