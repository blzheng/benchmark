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
        self.conv2d71 = Conv2d(1232, 308, kernel_size=(1, 1), stride=(1, 1))
        self.relu55 = ReLU()
        self.conv2d72 = Conv2d(308, 1232, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid13 = Sigmoid()

    def forward(self, x224, x223):
        x225=self.conv2d71(x224)
        x226=self.relu55(x225)
        x227=self.conv2d72(x226)
        x228=self.sigmoid13(x227)
        x229=operator.mul(x228, x223)
        return x229

m = M().eval()
x224 = torch.randn(torch.Size([1, 1232, 1, 1]))
x223 = torch.randn(torch.Size([1, 1232, 14, 14]))
start = time.time()
output = m(x224, x223)
end = time.time()
print(end-start)
