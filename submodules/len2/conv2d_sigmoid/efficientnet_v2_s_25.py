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
        self.conv2d147 = Conv2d(64, 1536, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid25 = Sigmoid()

    def forward(self, x468):
        x469=self.conv2d147(x468)
        x470=self.sigmoid25(x469)
        return x470

m = M().eval()
x468 = torch.randn(torch.Size([1, 64, 1, 1]))
start = time.time()
output = m(x468)
end = time.time()
print(end-start)
