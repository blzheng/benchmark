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
        self.relu34 = ReLU(inplace=True)
        self.conv2d53 = Conv2d(232, 232, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=232, bias=False)

    def forward(self, x345):
        x346=self.relu34(x345)
        x347=self.conv2d53(x346)
        return x347

m = M().eval()
x345 = torch.randn(torch.Size([1, 232, 7, 7]))
start = time.time()
output = m(x345)
end = time.time()
print(end-start)
