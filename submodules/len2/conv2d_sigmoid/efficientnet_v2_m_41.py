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
        self.conv2d232 = Conv2d(128, 3072, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid41 = Sigmoid()

    def forward(self, x741):
        x742=self.conv2d232(x741)
        x743=self.sigmoid41(x742)
        return x743

m = M().eval()
x741 = torch.randn(torch.Size([1, 128, 1, 1]))
start = time.time()
output = m(x741)
end = time.time()
print(end-start)
