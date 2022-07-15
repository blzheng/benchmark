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
        self.conv2d175 = Conv2d(1824, 76, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x523):
        x524=self.conv2d175(x523)
        return x524

m = M().eval()
x523 = torch.randn(torch.Size([1, 1824, 1, 1]))
start = time.time()
output = m(x523)
end = time.time()
print(end-start)
