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
        self.conv2d157 = Conv2d(64, 1536, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid27 = Sigmoid()

    def forward(self, x500, x497):
        x501=self.conv2d157(x500)
        x502=self.sigmoid27(x501)
        x503=operator.mul(x502, x497)
        return x503

m = M().eval()
x500 = torch.randn(torch.Size([1, 64, 1, 1]))
x497 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x500, x497)
end = time.time()
print(end-start)
