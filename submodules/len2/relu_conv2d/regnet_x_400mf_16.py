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
        self.relu16 = ReLU(inplace=True)
        self.conv2d20 = Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=10, bias=False)

    def forward(self, x61):
        x62=self.relu16(x61)
        x63=self.conv2d20(x62)
        return x63

m = M().eval()
x61 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x61)
end = time.time()
print(end-start)
