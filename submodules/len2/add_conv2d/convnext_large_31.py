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
        self.conv2d39 = Conv2d(1536, 1536, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=1536)

    def forward(self, x408, x398):
        x409=operator.add(x408, x398)
        x411=self.conv2d39(x409)
        return x411

m = M().eval()
x408 = torch.randn(torch.Size([1, 1536, 7, 7]))
x398 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x408, x398)
end = time.time()
print(end-start)
