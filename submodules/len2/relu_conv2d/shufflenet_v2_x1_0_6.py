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
        self.relu15 = ReLU(inplace=True)
        self.conv2d24 = Conv2d(116, 116, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=116, bias=False)

    def forward(self, x145):
        x146=self.relu15(x145)
        x147=self.conv2d24(x146)
        return x147

m = M().eval()
x145 = torch.randn(torch.Size([1, 116, 14, 14]))
start = time.time()
output = m(x145)
end = time.time()
print(end-start)
