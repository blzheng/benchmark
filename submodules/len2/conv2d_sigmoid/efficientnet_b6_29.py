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
        self.conv2d146 = Conv2d(50, 1200, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid29 = Sigmoid()

    def forward(self, x457):
        x458=self.conv2d146(x457)
        x459=self.sigmoid29(x458)
        return x459

m = M().eval()
x457 = torch.randn(torch.Size([1, 50, 1, 1]))
start = time.time()
output = m(x457)
end = time.time()
print(end-start)
