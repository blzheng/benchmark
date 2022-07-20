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
        self.conv2d18 = Conv2d(384, 768, kernel_size=(2, 2), stride=(2, 2))
        self.conv2d19 = Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)

    def forward(self, x188):
        x189=self.conv2d18(x188)
        x191=self.conv2d19(x189)
        return x191

m = M().eval()
x188 = torch.randn(torch.Size([1, 384, 14, 14]))
start = time.time()
output = m(x188)
end = time.time()
print(end-start)
