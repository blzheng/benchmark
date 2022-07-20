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
        self.conv2d43 = Conv2d(576, 144, kernel_size=(1, 1), stride=(1, 1))
        self.relu12 = ReLU()
        self.conv2d44 = Conv2d(144, 576, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x125):
        x126=self.conv2d43(x125)
        x127=self.relu12(x126)
        x128=self.conv2d44(x127)
        return x128

m = M().eval()
x125 = torch.randn(torch.Size([1, 576, 1, 1]))
start = time.time()
output = m(x125)
end = time.time()
print(end-start)
