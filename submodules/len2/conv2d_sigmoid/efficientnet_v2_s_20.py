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
        self.conv2d122 = Conv2d(64, 1536, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid20 = Sigmoid()

    def forward(self, x388):
        x389=self.conv2d122(x388)
        x390=self.sigmoid20(x389)
        return x390

m = M().eval()
x388 = torch.randn(torch.Size([1, 64, 1, 1]))
start = time.time()
output = m(x388)
end = time.time()
print(end-start)
