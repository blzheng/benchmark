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
        self.relu100 = ReLU(inplace=True)
        self.conv2d129 = Conv2d(2904, 2904, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x393, x407):
        x408=operator.add(x393, x407)
        x409=self.relu100(x408)
        x410=self.conv2d129(x409)
        return x410

m = M().eval()
x393 = torch.randn(torch.Size([1, 2904, 14, 14]))
x407 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x393, x407)
end = time.time()
print(end-start)
