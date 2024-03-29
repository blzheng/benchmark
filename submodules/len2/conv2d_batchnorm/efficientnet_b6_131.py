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
        self.conv2d219 = Conv2d(3456, 3456, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=3456, bias=False)
        self.batchnorm2d131 = BatchNorm2d(3456, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x687):
        x688=self.conv2d219(x687)
        x689=self.batchnorm2d131(x688)
        return x689

m = M().eval()
x687 = torch.randn(torch.Size([1, 3456, 7, 7]))
start = time.time()
output = m(x687)
end = time.time()
print(end-start)
