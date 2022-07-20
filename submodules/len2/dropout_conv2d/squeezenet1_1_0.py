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
        self.dropout0 = Dropout(p=0.5, inplace=False)
        self.conv2d25 = Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x61):
        x62=self.dropout0(x61)
        x63=self.conv2d25(x62)
        return x63

m = M().eval()
x61 = torch.randn(torch.Size([1, 512, 13, 13]))
start = time.time()
output = m(x61)
end = time.time()
print(end-start)
