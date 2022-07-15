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
        self.conv2d39 = Conv2d(116, 116, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=116, bias=False)

    def forward(self, x256):
        x257=self.conv2d39(x256)
        return x257

m = M().eval()
x256 = torch.randn(torch.Size([1, 116, 14, 14]))
start = time.time()
output = m(x256)
end = time.time()
print(end-start)
