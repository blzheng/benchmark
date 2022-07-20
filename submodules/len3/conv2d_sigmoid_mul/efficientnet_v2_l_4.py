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
        self.conv2d56 = Conv2d(48, 768, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid4 = Sigmoid()

    def forward(self, x188, x185):
        x189=self.conv2d56(x188)
        x190=self.sigmoid4(x189)
        x191=operator.mul(x190, x185)
        return x191

m = M().eval()
x188 = torch.randn(torch.Size([1, 48, 1, 1]))
x185 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x188, x185)
end = time.time()
print(end-start)
