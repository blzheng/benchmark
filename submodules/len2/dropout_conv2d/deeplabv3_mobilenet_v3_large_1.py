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
        self.dropout1 = Dropout(p=0.1, inplace=False)
        self.conv2d71 = Conv2d(10, 21, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x215):
        x216=self.dropout1(x215)
        x217=self.conv2d71(x216)
        return x217

m = M().eval()
x215 = torch.randn(torch.Size([1, 10, 28, 28]))
start = time.time()
output = m(x215)
end = time.time()
print(end-start)
