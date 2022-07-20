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
        self.relu13 = ReLU(inplace=True)
        self.conv2d17 = Conv2d(288, 288, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=18, bias=False)

    def forward(self, x51):
        x52=self.relu13(x51)
        x53=self.conv2d17(x52)
        return x53

m = M().eval()
x51 = torch.randn(torch.Size([1, 288, 28, 28]))
start = time.time()
output = m(x51)
end = time.time()
print(end-start)
