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
        self.conv2d73 = Conv2d(110, 440, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid13 = Sigmoid()

    def forward(self, x228):
        x229=self.conv2d73(x228)
        x230=self.sigmoid13(x229)
        return x230

m = M().eval()
x228 = torch.randn(torch.Size([1, 110, 1, 1]))
start = time.time()
output = m(x228)
end = time.time()
print(end-start)
