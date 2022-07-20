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
        self.conv2d306 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid54 = Sigmoid()

    def forward(self, x984, x981):
        x985=self.conv2d306(x984)
        x986=self.sigmoid54(x985)
        x987=operator.mul(x986, x981)
        return x987

m = M().eval()
x984 = torch.randn(torch.Size([1, 96, 1, 1]))
x981 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x984, x981)
end = time.time()
print(end-start)