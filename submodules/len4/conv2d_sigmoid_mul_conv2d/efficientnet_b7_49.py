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
        self.conv2d245 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid49 = Sigmoid()
        self.conv2d246 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x772, x769):
        x773=self.conv2d245(x772)
        x774=self.sigmoid49(x773)
        x775=operator.mul(x774, x769)
        x776=self.conv2d246(x775)
        return x776

m = M().eval()
x772 = torch.randn(torch.Size([1, 96, 1, 1]))
x769 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x772, x769)
end = time.time()
print(end-start)
