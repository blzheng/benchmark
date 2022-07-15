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
        self.conv2d223 = Conv2d(2304, 2304, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2304, bias=False)

    def forward(self, x702):
        x703=self.conv2d223(x702)
        return x703

m = M().eval()
x702 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x702)
end = time.time()
print(end-start)
