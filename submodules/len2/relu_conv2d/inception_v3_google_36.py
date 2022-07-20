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
        self.conv2d67 = Conv2d(192, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)

    def forward(self, x229):
        x230=torch.nn.functional.relu(x229,inplace=True)
        x231=self.conv2d67(x230)
        return x231

m = M().eval()
x229 = torch.randn(torch.Size([1, 192, 12, 12]))
start = time.time()
output = m(x229)
end = time.time()
print(end-start)
