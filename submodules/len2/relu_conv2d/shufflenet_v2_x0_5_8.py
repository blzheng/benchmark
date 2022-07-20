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
        self.relu19 = ReLU(inplace=True)
        self.conv2d30 = Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48, bias=False)

    def forward(self, x189):
        x190=self.relu19(x189)
        x191=self.conv2d30(x190)
        return x191

m = M().eval()
x189 = torch.randn(torch.Size([1, 48, 14, 14]))
start = time.time()
output = m(x189)
end = time.time()
print(end-start)