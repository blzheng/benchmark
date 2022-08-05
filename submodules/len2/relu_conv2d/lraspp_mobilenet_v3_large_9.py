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
        self.relu11 = ReLU(inplace=True)
        self.conv2d20 = Conv2d(120, 120, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=120, bias=False)

    def forward(self, x57):
        x58=self.relu11(x57)
        x59=self.conv2d20(x58)
        return x59

m = M().eval()
x57 = torch.randn(torch.Size([1, 120, 28, 28]))
start = time.time()
output = m(x57)
end = time.time()
print(end-start)
