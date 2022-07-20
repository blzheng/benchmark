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
        self.relu125 = ReLU(inplace=True)
        self.conv2d125 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x444):
        x445=self.relu125(x444)
        x446=self.conv2d125(x445)
        return x446

m = M().eval()
x444 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x444)
end = time.time()
print(end-start)
