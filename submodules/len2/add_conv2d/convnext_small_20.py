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
        self.conv2d26 = Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384)

    def forward(self, x270, x260):
        x271=operator.add(x270, x260)
        x273=self.conv2d26(x271)
        return x273

m = M().eval()
x270 = torch.randn(torch.Size([1, 384, 14, 14]))
x260 = torch.randn(torch.Size([1, 384, 14, 14]))
start = time.time()
output = m(x270, x260)
end = time.time()
print(end-start)
