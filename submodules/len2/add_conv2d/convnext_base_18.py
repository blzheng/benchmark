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
        self.conv2d24 = Conv2d(512, 512, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=512)

    def forward(self, x248, x238):
        x249=operator.add(x248, x238)
        x251=self.conv2d24(x249)
        return x251

m = M().eval()
x248 = torch.randn(torch.Size([1, 512, 14, 14]))
x238 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x248, x238)
end = time.time()
print(end-start)
