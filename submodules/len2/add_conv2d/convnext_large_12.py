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
        self.conv2d18 = Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)

    def forward(self, x182, x172):
        x183=operator.add(x182, x172)
        x185=self.conv2d18(x183)
        return x185

m = M().eval()
x182 = torch.randn(torch.Size([1, 768, 14, 14]))
x172 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x182, x172)
end = time.time()
print(end-start)
