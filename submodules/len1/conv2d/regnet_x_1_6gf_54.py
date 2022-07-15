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
        self.conv2d54 = Conv2d(912, 912, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=38, bias=False)

    def forward(self, x174):
        x175=self.conv2d54(x174)
        return x175

m = M().eval()
x174 = torch.randn(torch.Size([1, 912, 14, 14]))
start = time.time()
output = m(x174)
end = time.time()
print(end-start)
