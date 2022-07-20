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
        self.relu86 = ReLU(inplace=True)
        self.conv2d86 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x306):
        x307=self.relu86(x306)
        x308=self.conv2d86(x307)
        return x308

m = M().eval()
x306 = torch.randn(torch.Size([1, 192, 14, 14]))
start = time.time()
output = m(x306)
end = time.time()
print(end-start)
