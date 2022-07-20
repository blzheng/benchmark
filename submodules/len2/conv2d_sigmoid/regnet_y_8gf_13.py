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
        self.conv2d72 = Conv2d(224, 896, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid13 = Sigmoid()

    def forward(self, x226):
        x227=self.conv2d72(x226)
        x228=self.sigmoid13(x227)
        return x228

m = M().eval()
x226 = torch.randn(torch.Size([1, 224, 1, 1]))
start = time.time()
output = m(x226)
end = time.time()
print(end-start)
