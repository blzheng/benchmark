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
        self.conv2d92 = Conv2d(44, 1056, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid13 = Sigmoid()

    def forward(self, x297):
        x298=self.conv2d92(x297)
        x299=self.sigmoid13(x298)
        return x299

m = M().eval()
x297 = torch.randn(torch.Size([1, 44, 1, 1]))
start = time.time()
output = m(x297)
end = time.time()
print(end-start)
