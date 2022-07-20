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
        self.conv2d260 = Conv2d(160, 3840, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid52 = Sigmoid()

    def forward(self, x818, x815):
        x819=self.conv2d260(x818)
        x820=self.sigmoid52(x819)
        x821=operator.mul(x820, x815)
        return x821

m = M().eval()
x818 = torch.randn(torch.Size([1, 160, 1, 1]))
x815 = torch.randn(torch.Size([1, 3840, 7, 7]))
start = time.time()
output = m(x818, x815)
end = time.time()
print(end-start)
