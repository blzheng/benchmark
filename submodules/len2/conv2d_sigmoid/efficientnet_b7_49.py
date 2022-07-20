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

    def forward(self, x772):
        x773=self.conv2d245(x772)
        x774=self.sigmoid49(x773)
        return x774

m = M().eval()
x772 = torch.randn(torch.Size([1, 96, 1, 1]))
start = time.time()
output = m(x772)
end = time.time()
print(end-start)
