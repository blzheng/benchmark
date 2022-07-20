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
        self.relu37 = ReLU(inplace=True)
        self.conv2d42 = Conv2d(400, 400, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=25, bias=False)

    def forward(self, x133):
        x134=self.relu37(x133)
        x135=self.conv2d42(x134)
        return x135

m = M().eval()
x133 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x133)
end = time.time()
print(end-start)
