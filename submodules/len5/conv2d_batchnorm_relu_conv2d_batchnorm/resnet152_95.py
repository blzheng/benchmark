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
        self.conv2d146 = Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchnorm2d146 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu142 = ReLU(inplace=True)
        self.conv2d147 = Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d147 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x483):
        x484=self.conv2d146(x483)
        x485=self.batchnorm2d146(x484)
        x486=self.relu142(x485)
        x487=self.conv2d147(x486)
        x488=self.batchnorm2d147(x487)
        return x488

m = M().eval()
x483 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x483)
end = time.time()
print(end-start)
