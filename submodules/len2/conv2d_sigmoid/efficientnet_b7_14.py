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
        self.conv2d70 = Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid14 = Sigmoid()

    def forward(self, x218):
        x219=self.conv2d70(x218)
        x220=self.sigmoid14(x219)
        return x220

m = M().eval()
x218 = torch.randn(torch.Size([1, 20, 1, 1]))
start = time.time()
output = m(x218)
end = time.time()
print(end-start)
