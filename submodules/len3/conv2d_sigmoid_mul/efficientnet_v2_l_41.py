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
        self.conv2d241 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid41 = Sigmoid()

    def forward(self, x776, x773):
        x777=self.conv2d241(x776)
        x778=self.sigmoid41(x777)
        x779=operator.mul(x778, x773)
        return x779

m = M().eval()
x776 = torch.randn(torch.Size([1, 96, 1, 1]))
x773 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x776, x773)
end = time.time()
print(end-start)
