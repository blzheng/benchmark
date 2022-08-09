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
import sys
import os

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.conv2d0 = Conv2d(3, 96, kernel_size=(4, 4), stride=(4, 4))
        self.layernorm0 = LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        self.layernorm1 = LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        self.layernorm2 = LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        self.linear0 = Linear(in_features=96, out_features=384, bias=True)
        self.gelu0 = GELU(approximate='none')
        self.dropout0 = Dropout(p=0.0, inplace=False)
        self.linear1 = Linear(in_features=384, out_features=96, bias=True)
        self.dropout1 = Dropout(p=0.0, inplace=False)
        self.layernorm3 = LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        self.layernorm4 = LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        self.linear2 = Linear(in_features=96, out_features=384, bias=True)
        self.gelu1 = GELU(approximate='none')
        self.dropout2 = Dropout(p=0.0, inplace=False)
        self.linear3 = Linear(in_features=384, out_features=96, bias=True)
        self.dropout3 = Dropout(p=0.0, inplace=False)
        self.layernorm5 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.linear4 = Linear(in_features=384, out_features=192, bias=False)
        self.layernorm6 = LayerNorm((192,), eps=1e-05, elementwise_affine=True)
        self.layernorm7 = LayerNorm((192,), eps=1e-05, elementwise_affine=True)
        self.linear5 = Linear(in_features=192, out_features=768, bias=True)
        self.gelu2 = GELU(approximate='none')
        self.dropout4 = Dropout(p=0.0, inplace=False)
        self.linear6 = Linear(in_features=768, out_features=192, bias=True)
        self.dropout5 = Dropout(p=0.0, inplace=False)
        self.layernorm8 = LayerNorm((192,), eps=1e-05, elementwise_affine=True)
        self.layernorm9 = LayerNorm((192,), eps=1e-05, elementwise_affine=True)
        self.linear7 = Linear(in_features=192, out_features=768, bias=True)
        self.gelu3 = GELU(approximate='none')
        self.dropout6 = Dropout(p=0.0, inplace=False)
        self.linear8 = Linear(in_features=768, out_features=192, bias=True)
        self.dropout7 = Dropout(p=0.0, inplace=False)
        self.layernorm10 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear9 = Linear(in_features=768, out_features=384, bias=False)
        self.layernorm11 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.layernorm12 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.linear10 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu4 = GELU(approximate='none')
        self.dropout8 = Dropout(p=0.0, inplace=False)
        self.linear11 = Linear(in_features=1536, out_features=384, bias=True)
        self.dropout9 = Dropout(p=0.0, inplace=False)
        self.layernorm13 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.layernorm14 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.linear12 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu5 = GELU(approximate='none')
        self.dropout10 = Dropout(p=0.0, inplace=False)
        self.linear13 = Linear(in_features=1536, out_features=384, bias=True)
        self.dropout11 = Dropout(p=0.0, inplace=False)
        self.layernorm15 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.layernorm16 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.linear14 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu6 = GELU(approximate='none')
        self.dropout12 = Dropout(p=0.0, inplace=False)
        self.linear15 = Linear(in_features=1536, out_features=384, bias=True)
        self.dropout13 = Dropout(p=0.0, inplace=False)
        self.layernorm17 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.layernorm18 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.linear16 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu7 = GELU(approximate='none')
        self.dropout14 = Dropout(p=0.0, inplace=False)
        self.linear17 = Linear(in_features=1536, out_features=384, bias=True)
        self.dropout15 = Dropout(p=0.0, inplace=False)
        self.layernorm19 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.layernorm20 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.linear18 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu8 = GELU(approximate='none')
        self.dropout16 = Dropout(p=0.0, inplace=False)
        self.linear19 = Linear(in_features=1536, out_features=384, bias=True)
        self.dropout17 = Dropout(p=0.0, inplace=False)
        self.layernorm21 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.layernorm22 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.linear20 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu9 = GELU(approximate='none')
        self.dropout18 = Dropout(p=0.0, inplace=False)
        self.linear21 = Linear(in_features=1536, out_features=384, bias=True)
        self.dropout19 = Dropout(p=0.0, inplace=False)
        self.layernorm23 = LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
        self.linear22 = Linear(in_features=1536, out_features=768, bias=False)
        self.layernorm24 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.layernorm25 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear23 = Linear(in_features=768, out_features=3072, bias=True)
        self.gelu10 = GELU(approximate='none')
        self.dropout20 = Dropout(p=0.0, inplace=False)
        self.linear24 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout21 = Dropout(p=0.0, inplace=False)
        self.layernorm26 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.layernorm27 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear25 = Linear(in_features=768, out_features=3072, bias=True)
        self.gelu11 = GELU(approximate='none')
        self.dropout22 = Dropout(p=0.0, inplace=False)
        self.linear26 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout23 = Dropout(p=0.0, inplace=False)
        self.layernorm28 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.adaptiveavgpool2d0 = AdaptiveAvgPool2d(output_size=1)
        self.linear27 = Linear(in_features=768, out_features=1000, bias=True)
        self.relative_position_bias_table0 = torch.rand(torch.Size([169, 3])).to(torch.float32)
        self.relative_position_index0 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight0 = torch.rand(torch.Size([288, 96])).to(torch.float32)
        self.weight1 = torch.rand(torch.Size([96, 96])).to(torch.float32)
        self.bias0 = torch.rand(torch.Size([288])).to(torch.float32)
        self.bias1 = torch.rand(torch.Size([96])).to(torch.float32)
        self.relative_position_bias_table1 = torch.rand(torch.Size([169, 3])).to(torch.float32)
        self.relative_position_index1 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight2 = torch.rand(torch.Size([288, 96])).to(torch.float32)
        self.weight3 = torch.rand(torch.Size([96, 96])).to(torch.float32)
        self.bias2 = torch.rand(torch.Size([288])).to(torch.float32)
        self.bias3 = torch.rand(torch.Size([96])).to(torch.float32)
        self.relative_position_bias_table2 = torch.rand(torch.Size([169, 6])).to(torch.float32)
        self.relative_position_index2 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight4 = torch.rand(torch.Size([576, 192])).to(torch.float32)
        self.weight5 = torch.rand(torch.Size([192, 192])).to(torch.float32)
        self.bias4 = torch.rand(torch.Size([576])).to(torch.float32)
        self.bias5 = torch.rand(torch.Size([192])).to(torch.float32)
        self.relative_position_bias_table3 = torch.rand(torch.Size([169, 6])).to(torch.float32)
        self.relative_position_index3 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight6 = torch.rand(torch.Size([576, 192])).to(torch.float32)
        self.weight7 = torch.rand(torch.Size([192, 192])).to(torch.float32)
        self.bias6 = torch.rand(torch.Size([576])).to(torch.float32)
        self.bias7 = torch.rand(torch.Size([192])).to(torch.float32)
        self.relative_position_bias_table4 = torch.rand(torch.Size([169, 12])).to(torch.float32)
        self.relative_position_index4 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight8 = torch.rand(torch.Size([1152, 384])).to(torch.float32)
        self.weight9 = torch.rand(torch.Size([384, 384])).to(torch.float32)
        self.bias8 = torch.rand(torch.Size([1152])).to(torch.float32)
        self.bias9 = torch.rand(torch.Size([384])).to(torch.float32)
        self.relative_position_bias_table5 = torch.rand(torch.Size([169, 12])).to(torch.float32)
        self.relative_position_index5 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight10 = torch.rand(torch.Size([1152, 384])).to(torch.float32)
        self.weight11 = torch.rand(torch.Size([384, 384])).to(torch.float32)
        self.bias10 = torch.rand(torch.Size([1152])).to(torch.float32)
        self.bias11 = torch.rand(torch.Size([384])).to(torch.float32)
        self.relative_position_bias_table6 = torch.rand(torch.Size([169, 12])).to(torch.float32)
        self.relative_position_index6 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight12 = torch.rand(torch.Size([1152, 384])).to(torch.float32)
        self.weight13 = torch.rand(torch.Size([384, 384])).to(torch.float32)
        self.bias12 = torch.rand(torch.Size([1152])).to(torch.float32)
        self.bias13 = torch.rand(torch.Size([384])).to(torch.float32)
        self.relative_position_bias_table7 = torch.rand(torch.Size([169, 12])).to(torch.float32)
        self.relative_position_index7 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight14 = torch.rand(torch.Size([1152, 384])).to(torch.float32)
        self.weight15 = torch.rand(torch.Size([384, 384])).to(torch.float32)
        self.bias14 = torch.rand(torch.Size([1152])).to(torch.float32)
        self.bias15 = torch.rand(torch.Size([384])).to(torch.float32)
        self.relative_position_bias_table8 = torch.rand(torch.Size([169, 12])).to(torch.float32)
        self.relative_position_index8 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight16 = torch.rand(torch.Size([1152, 384])).to(torch.float32)
        self.weight17 = torch.rand(torch.Size([384, 384])).to(torch.float32)
        self.bias16 = torch.rand(torch.Size([1152])).to(torch.float32)
        self.bias17 = torch.rand(torch.Size([384])).to(torch.float32)
        self.relative_position_bias_table9 = torch.rand(torch.Size([169, 12])).to(torch.float32)
        self.relative_position_index9 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight18 = torch.rand(torch.Size([1152, 384])).to(torch.float32)
        self.weight19 = torch.rand(torch.Size([384, 384])).to(torch.float32)
        self.bias18 = torch.rand(torch.Size([1152])).to(torch.float32)
        self.bias19 = torch.rand(torch.Size([384])).to(torch.float32)
        self.relative_position_bias_table10 = torch.rand(torch.Size([169, 24])).to(torch.float32)
        self.relative_position_index10 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight20 = torch.rand(torch.Size([2304, 768])).to(torch.float32)
        self.weight21 = torch.rand(torch.Size([768, 768])).to(torch.float32)
        self.bias20 = torch.rand(torch.Size([2304])).to(torch.float32)
        self.bias21 = torch.rand(torch.Size([768])).to(torch.float32)
        self.relative_position_bias_table11 = torch.rand(torch.Size([169, 24])).to(torch.float32)
        self.relative_position_index11 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight22 = torch.rand(torch.Size([2304, 768])).to(torch.float32)
        self.weight23 = torch.rand(torch.Size([768, 768])).to(torch.float32)
        self.bias22 = torch.rand(torch.Size([2304])).to(torch.float32)
        self.bias23 = torch.rand(torch.Size([768])).to(torch.float32)

    def forward(self, x):
        x0=x
        if x0 is None:
            print('x0: {}'.format(x0))
        elif isinstance(x0, torch.Tensor):
            print('x0: {}'.format(x0.shape))
        elif isinstance(x0, tuple):
            tuple_shapes = '('
            for item in x0:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x0: {}'.format(tuple_shapes))
        else:
            print('x0: {}'.format(x0))
        x1=self.conv2d0(x0)
        if x1 is None:
            print('x1: {}'.format(x1))
        elif isinstance(x1, torch.Tensor):
            print('x1: {}'.format(x1.shape))
        elif isinstance(x1, tuple):
            tuple_shapes = '('
            for item in x1:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x1: {}'.format(tuple_shapes))
        else:
            print('x1: {}'.format(x1))
        x2=torch.permute(x1, [0, 2, 3, 1])
        if x2 is None:
            print('x2: {}'.format(x2))
        elif isinstance(x2, torch.Tensor):
            print('x2: {}'.format(x2.shape))
        elif isinstance(x2, tuple):
            tuple_shapes = '('
            for item in x2:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x2: {}'.format(tuple_shapes))
        else:
            print('x2: {}'.format(x2))
        x3=self.layernorm0(x2)
        if x3 is None:
            print('x3: {}'.format(x3))
        elif isinstance(x3, torch.Tensor):
            print('x3: {}'.format(x3.shape))
        elif isinstance(x3, tuple):
            tuple_shapes = '('
            for item in x3:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x3: {}'.format(tuple_shapes))
        else:
            print('x3: {}'.format(x3))
        x4=self.layernorm1(x3)
        if x4 is None:
            print('x4: {}'.format(x4))
        elif isinstance(x4, torch.Tensor):
            print('x4: {}'.format(x4.shape))
        elif isinstance(x4, tuple):
            tuple_shapes = '('
            for item in x4:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x4: {}'.format(tuple_shapes))
        else:
            print('x4: {}'.format(x4))
        x7=operator.getitem(self.relative_position_bias_table0, self.relative_position_index0)
        if x7 is None:
            print('x7: {}'.format(x7))
        elif isinstance(x7, torch.Tensor):
            print('x7: {}'.format(x7.shape))
        elif isinstance(x7, tuple):
            tuple_shapes = '('
            for item in x7:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x7: {}'.format(tuple_shapes))
        else:
            print('x7: {}'.format(x7))
        x8=x7.view(49, 49, -1)
        if x8 is None:
            print('x8: {}'.format(x8))
        elif isinstance(x8, torch.Tensor):
            print('x8: {}'.format(x8.shape))
        elif isinstance(x8, tuple):
            tuple_shapes = '('
            for item in x8:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x8: {}'.format(tuple_shapes))
        else:
            print('x8: {}'.format(x8))
        x9=x8.permute(2, 0, 1)
        if x9 is None:
            print('x9: {}'.format(x9))
        elif isinstance(x9, torch.Tensor):
            print('x9: {}'.format(x9.shape))
        elif isinstance(x9, tuple):
            tuple_shapes = '('
            for item in x9:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x9: {}'.format(tuple_shapes))
        else:
            print('x9: {}'.format(x9))
        x10=x9.contiguous()
        if x10 is None:
            print('x10: {}'.format(x10))
        elif isinstance(x10, torch.Tensor):
            print('x10: {}'.format(x10.shape))
        elif isinstance(x10, tuple):
            tuple_shapes = '('
            for item in x10:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x10: {}'.format(tuple_shapes))
        else:
            print('x10: {}'.format(x10))
        x11=x10.unsqueeze(0)
        if x11 is None:
            print('x11: {}'.format(x11))
        elif isinstance(x11, torch.Tensor):
            print('x11: {}'.format(x11.shape))
        elif isinstance(x11, tuple):
            tuple_shapes = '('
            for item in x11:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x11: {}'.format(tuple_shapes))
        else:
            print('x11: {}'.format(x11))
        x16=torchvision.models.swin_transformer.shifted_window_attention(x4, self.weight0, self.weight1, x11, [7, 7], 3,shift_size=[0, 0], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias0, proj_bias=self.bias1)
        if x16 is None:
            print('x16: {}'.format(x16))
        elif isinstance(x16, torch.Tensor):
            print('x16: {}'.format(x16.shape))
        elif isinstance(x16, tuple):
            tuple_shapes = '('
            for item in x16:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x16: {}'.format(tuple_shapes))
        else:
            print('x16: {}'.format(x16))
        x17=stochastic_depth(x16, 0.0, 'row', False)
        if x17 is None:
            print('x17: {}'.format(x17))
        elif isinstance(x17, torch.Tensor):
            print('x17: {}'.format(x17.shape))
        elif isinstance(x17, tuple):
            tuple_shapes = '('
            for item in x17:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x17: {}'.format(tuple_shapes))
        else:
            print('x17: {}'.format(x17))
        x18=operator.add(x3, x17)
        if x18 is None:
            print('x18: {}'.format(x18))
        elif isinstance(x18, torch.Tensor):
            print('x18: {}'.format(x18.shape))
        elif isinstance(x18, tuple):
            tuple_shapes = '('
            for item in x18:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x18: {}'.format(tuple_shapes))
        else:
            print('x18: {}'.format(x18))
        x19=self.layernorm2(x18)
        if x19 is None:
            print('x19: {}'.format(x19))
        elif isinstance(x19, torch.Tensor):
            print('x19: {}'.format(x19.shape))
        elif isinstance(x19, tuple):
            tuple_shapes = '('
            for item in x19:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x19: {}'.format(tuple_shapes))
        else:
            print('x19: {}'.format(x19))
        x20=self.linear0(x19)
        if x20 is None:
            print('x20: {}'.format(x20))
        elif isinstance(x20, torch.Tensor):
            print('x20: {}'.format(x20.shape))
        elif isinstance(x20, tuple):
            tuple_shapes = '('
            for item in x20:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x20: {}'.format(tuple_shapes))
        else:
            print('x20: {}'.format(x20))
        x21=self.gelu0(x20)
        if x21 is None:
            print('x21: {}'.format(x21))
        elif isinstance(x21, torch.Tensor):
            print('x21: {}'.format(x21.shape))
        elif isinstance(x21, tuple):
            tuple_shapes = '('
            for item in x21:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x21: {}'.format(tuple_shapes))
        else:
            print('x21: {}'.format(x21))
        x22=self.dropout0(x21)
        if x22 is None:
            print('x22: {}'.format(x22))
        elif isinstance(x22, torch.Tensor):
            print('x22: {}'.format(x22.shape))
        elif isinstance(x22, tuple):
            tuple_shapes = '('
            for item in x22:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x22: {}'.format(tuple_shapes))
        else:
            print('x22: {}'.format(x22))
        x23=self.linear1(x22)
        if x23 is None:
            print('x23: {}'.format(x23))
        elif isinstance(x23, torch.Tensor):
            print('x23: {}'.format(x23.shape))
        elif isinstance(x23, tuple):
            tuple_shapes = '('
            for item in x23:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x23: {}'.format(tuple_shapes))
        else:
            print('x23: {}'.format(x23))
        x24=self.dropout1(x23)
        if x24 is None:
            print('x24: {}'.format(x24))
        elif isinstance(x24, torch.Tensor):
            print('x24: {}'.format(x24.shape))
        elif isinstance(x24, tuple):
            tuple_shapes = '('
            for item in x24:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x24: {}'.format(tuple_shapes))
        else:
            print('x24: {}'.format(x24))
        x25=stochastic_depth(x24, 0.0, 'row', False)
        if x25 is None:
            print('x25: {}'.format(x25))
        elif isinstance(x25, torch.Tensor):
            print('x25: {}'.format(x25.shape))
        elif isinstance(x25, tuple):
            tuple_shapes = '('
            for item in x25:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x25: {}'.format(tuple_shapes))
        else:
            print('x25: {}'.format(x25))
        x26=operator.add(x18, x25)
        if x26 is None:
            print('x26: {}'.format(x26))
        elif isinstance(x26, torch.Tensor):
            print('x26: {}'.format(x26.shape))
        elif isinstance(x26, tuple):
            tuple_shapes = '('
            for item in x26:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x26: {}'.format(tuple_shapes))
        else:
            print('x26: {}'.format(x26))
        x27=self.layernorm3(x26)
        if x27 is None:
            print('x27: {}'.format(x27))
        elif isinstance(x27, torch.Tensor):
            print('x27: {}'.format(x27.shape))
        elif isinstance(x27, tuple):
            tuple_shapes = '('
            for item in x27:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x27: {}'.format(tuple_shapes))
        else:
            print('x27: {}'.format(x27))
        x30=operator.getitem(self.relative_position_bias_table1, self.relative_position_index1)
        if x30 is None:
            print('x30: {}'.format(x30))
        elif isinstance(x30, torch.Tensor):
            print('x30: {}'.format(x30.shape))
        elif isinstance(x30, tuple):
            tuple_shapes = '('
            for item in x30:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x30: {}'.format(tuple_shapes))
        else:
            print('x30: {}'.format(x30))
        x31=x30.view(49, 49, -1)
        if x31 is None:
            print('x31: {}'.format(x31))
        elif isinstance(x31, torch.Tensor):
            print('x31: {}'.format(x31.shape))
        elif isinstance(x31, tuple):
            tuple_shapes = '('
            for item in x31:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x31: {}'.format(tuple_shapes))
        else:
            print('x31: {}'.format(x31))
        x32=x31.permute(2, 0, 1)
        if x32 is None:
            print('x32: {}'.format(x32))
        elif isinstance(x32, torch.Tensor):
            print('x32: {}'.format(x32.shape))
        elif isinstance(x32, tuple):
            tuple_shapes = '('
            for item in x32:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x32: {}'.format(tuple_shapes))
        else:
            print('x32: {}'.format(x32))
        x33=x32.contiguous()
        if x33 is None:
            print('x33: {}'.format(x33))
        elif isinstance(x33, torch.Tensor):
            print('x33: {}'.format(x33.shape))
        elif isinstance(x33, tuple):
            tuple_shapes = '('
            for item in x33:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x33: {}'.format(tuple_shapes))
        else:
            print('x33: {}'.format(x33))
        x34=x33.unsqueeze(0)
        if x34 is None:
            print('x34: {}'.format(x34))
        elif isinstance(x34, torch.Tensor):
            print('x34: {}'.format(x34.shape))
        elif isinstance(x34, tuple):
            tuple_shapes = '('
            for item in x34:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x34: {}'.format(tuple_shapes))
        else:
            print('x34: {}'.format(x34))
        x39=torchvision.models.swin_transformer.shifted_window_attention(x27, self.weight2, self.weight3, x34, [7, 7], 3,shift_size=[3, 3], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias2, proj_bias=self.bias3)
        if x39 is None:
            print('x39: {}'.format(x39))
        elif isinstance(x39, torch.Tensor):
            print('x39: {}'.format(x39.shape))
        elif isinstance(x39, tuple):
            tuple_shapes = '('
            for item in x39:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x39: {}'.format(tuple_shapes))
        else:
            print('x39: {}'.format(x39))
        x40=stochastic_depth(x39, 0.018181818181818184, 'row', False)
        if x40 is None:
            print('x40: {}'.format(x40))
        elif isinstance(x40, torch.Tensor):
            print('x40: {}'.format(x40.shape))
        elif isinstance(x40, tuple):
            tuple_shapes = '('
            for item in x40:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x40: {}'.format(tuple_shapes))
        else:
            print('x40: {}'.format(x40))
        x41=operator.add(x26, x40)
        if x41 is None:
            print('x41: {}'.format(x41))
        elif isinstance(x41, torch.Tensor):
            print('x41: {}'.format(x41.shape))
        elif isinstance(x41, tuple):
            tuple_shapes = '('
            for item in x41:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x41: {}'.format(tuple_shapes))
        else:
            print('x41: {}'.format(x41))
        x42=self.layernorm4(x41)
        if x42 is None:
            print('x42: {}'.format(x42))
        elif isinstance(x42, torch.Tensor):
            print('x42: {}'.format(x42.shape))
        elif isinstance(x42, tuple):
            tuple_shapes = '('
            for item in x42:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x42: {}'.format(tuple_shapes))
        else:
            print('x42: {}'.format(x42))
        x43=self.linear2(x42)
        if x43 is None:
            print('x43: {}'.format(x43))
        elif isinstance(x43, torch.Tensor):
            print('x43: {}'.format(x43.shape))
        elif isinstance(x43, tuple):
            tuple_shapes = '('
            for item in x43:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x43: {}'.format(tuple_shapes))
        else:
            print('x43: {}'.format(x43))
        x44=self.gelu1(x43)
        if x44 is None:
            print('x44: {}'.format(x44))
        elif isinstance(x44, torch.Tensor):
            print('x44: {}'.format(x44.shape))
        elif isinstance(x44, tuple):
            tuple_shapes = '('
            for item in x44:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x44: {}'.format(tuple_shapes))
        else:
            print('x44: {}'.format(x44))
        x45=self.dropout2(x44)
        if x45 is None:
            print('x45: {}'.format(x45))
        elif isinstance(x45, torch.Tensor):
            print('x45: {}'.format(x45.shape))
        elif isinstance(x45, tuple):
            tuple_shapes = '('
            for item in x45:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x45: {}'.format(tuple_shapes))
        else:
            print('x45: {}'.format(x45))
        x46=self.linear3(x45)
        if x46 is None:
            print('x46: {}'.format(x46))
        elif isinstance(x46, torch.Tensor):
            print('x46: {}'.format(x46.shape))
        elif isinstance(x46, tuple):
            tuple_shapes = '('
            for item in x46:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x46: {}'.format(tuple_shapes))
        else:
            print('x46: {}'.format(x46))
        x47=self.dropout3(x46)
        if x47 is None:
            print('x47: {}'.format(x47))
        elif isinstance(x47, torch.Tensor):
            print('x47: {}'.format(x47.shape))
        elif isinstance(x47, tuple):
            tuple_shapes = '('
            for item in x47:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x47: {}'.format(tuple_shapes))
        else:
            print('x47: {}'.format(x47))
        x48=stochastic_depth(x47, 0.018181818181818184, 'row', False)
        if x48 is None:
            print('x48: {}'.format(x48))
        elif isinstance(x48, torch.Tensor):
            print('x48: {}'.format(x48.shape))
        elif isinstance(x48, tuple):
            tuple_shapes = '('
            for item in x48:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x48: {}'.format(tuple_shapes))
        else:
            print('x48: {}'.format(x48))
        x49=operator.add(x41, x48)
        if x49 is None:
            print('x49: {}'.format(x49))
        elif isinstance(x49, torch.Tensor):
            print('x49: {}'.format(x49.shape))
        elif isinstance(x49, tuple):
            tuple_shapes = '('
            for item in x49:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x49: {}'.format(tuple_shapes))
        else:
            print('x49: {}'.format(x49))
        x50=builtins.getattr(x49, 'shape')
        if x50 is None:
            print('x50: {}'.format(x50))
        elif isinstance(x50, torch.Tensor):
            print('x50: {}'.format(x50.shape))
        elif isinstance(x50, tuple):
            tuple_shapes = '('
            for item in x50:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x50: {}'.format(tuple_shapes))
        else:
            print('x50: {}'.format(x50))
        x51=operator.getitem(x50, slice(-3, None, None))
        if x51 is None:
            print('x51: {}'.format(x51))
        elif isinstance(x51, torch.Tensor):
            print('x51: {}'.format(x51.shape))
        elif isinstance(x51, tuple):
            tuple_shapes = '('
            for item in x51:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x51: {}'.format(tuple_shapes))
        else:
            print('x51: {}'.format(x51))
        x52=operator.getitem(x51, 0)
        if x52 is None:
            print('x52: {}'.format(x52))
        elif isinstance(x52, torch.Tensor):
            print('x52: {}'.format(x52.shape))
        elif isinstance(x52, tuple):
            tuple_shapes = '('
            for item in x52:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x52: {}'.format(tuple_shapes))
        else:
            print('x52: {}'.format(x52))
        x53=operator.getitem(x51, 1)
        if x53 is None:
            print('x53: {}'.format(x53))
        elif isinstance(x53, torch.Tensor):
            print('x53: {}'.format(x53.shape))
        elif isinstance(x53, tuple):
            tuple_shapes = '('
            for item in x53:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x53: {}'.format(tuple_shapes))
        else:
            print('x53: {}'.format(x53))
        x54=operator.getitem(x51, 2)
        if x54 is None:
            print('x54: {}'.format(x54))
        elif isinstance(x54, torch.Tensor):
            print('x54: {}'.format(x54.shape))
        elif isinstance(x54, tuple):
            tuple_shapes = '('
            for item in x54:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x54: {}'.format(tuple_shapes))
        else:
            print('x54: {}'.format(x54))
        x55=operator.mod(x53, 2)
        if x55 is None:
            print('x55: {}'.format(x55))
        elif isinstance(x55, torch.Tensor):
            print('x55: {}'.format(x55.shape))
        elif isinstance(x55, tuple):
            tuple_shapes = '('
            for item in x55:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x55: {}'.format(tuple_shapes))
        else:
            print('x55: {}'.format(x55))
        x56=operator.mod(x52, 2)
        if x56 is None:
            print('x56: {}'.format(x56))
        elif isinstance(x56, torch.Tensor):
            print('x56: {}'.format(x56.shape))
        elif isinstance(x56, tuple):
            tuple_shapes = '('
            for item in x56:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x56: {}'.format(tuple_shapes))
        else:
            print('x56: {}'.format(x56))
        x57=torch.nn.functional.pad(x49, (0, 0, 0, x55, 0, x56))
        if x57 is None:
            print('x57: {}'.format(x57))
        elif isinstance(x57, torch.Tensor):
            print('x57: {}'.format(x57.shape))
        elif isinstance(x57, tuple):
            tuple_shapes = '('
            for item in x57:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x57: {}'.format(tuple_shapes))
        else:
            print('x57: {}'.format(x57))
        x58=operator.getitem(x57, (Ellipsis, slice(0, None, 2), slice(0, None, 2), slice(None, None, None)))
        if x58 is None:
            print('x58: {}'.format(x58))
        elif isinstance(x58, torch.Tensor):
            print('x58: {}'.format(x58.shape))
        elif isinstance(x58, tuple):
            tuple_shapes = '('
            for item in x58:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x58: {}'.format(tuple_shapes))
        else:
            print('x58: {}'.format(x58))
        x59=operator.getitem(x57, (Ellipsis, slice(1, None, 2), slice(0, None, 2), slice(None, None, None)))
        if x59 is None:
            print('x59: {}'.format(x59))
        elif isinstance(x59, torch.Tensor):
            print('x59: {}'.format(x59.shape))
        elif isinstance(x59, tuple):
            tuple_shapes = '('
            for item in x59:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x59: {}'.format(tuple_shapes))
        else:
            print('x59: {}'.format(x59))
        x60=operator.getitem(x57, (Ellipsis, slice(0, None, 2), slice(1, None, 2), slice(None, None, None)))
        if x60 is None:
            print('x60: {}'.format(x60))
        elif isinstance(x60, torch.Tensor):
            print('x60: {}'.format(x60.shape))
        elif isinstance(x60, tuple):
            tuple_shapes = '('
            for item in x60:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x60: {}'.format(tuple_shapes))
        else:
            print('x60: {}'.format(x60))
        x61=operator.getitem(x57, (Ellipsis, slice(1, None, 2), slice(1, None, 2), slice(None, None, None)))
        if x61 is None:
            print('x61: {}'.format(x61))
        elif isinstance(x61, torch.Tensor):
            print('x61: {}'.format(x61.shape))
        elif isinstance(x61, tuple):
            tuple_shapes = '('
            for item in x61:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x61: {}'.format(tuple_shapes))
        else:
            print('x61: {}'.format(x61))
        x62=torch.cat([x58, x59, x60, x61], -1)
        if x62 is None:
            print('x62: {}'.format(x62))
        elif isinstance(x62, torch.Tensor):
            print('x62: {}'.format(x62.shape))
        elif isinstance(x62, tuple):
            tuple_shapes = '('
            for item in x62:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x62: {}'.format(tuple_shapes))
        else:
            print('x62: {}'.format(x62))
        x63=self.layernorm5(x62)
        if x63 is None:
            print('x63: {}'.format(x63))
        elif isinstance(x63, torch.Tensor):
            print('x63: {}'.format(x63.shape))
        elif isinstance(x63, tuple):
            tuple_shapes = '('
            for item in x63:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x63: {}'.format(tuple_shapes))
        else:
            print('x63: {}'.format(x63))
        x64=self.linear4(x63)
        if x64 is None:
            print('x64: {}'.format(x64))
        elif isinstance(x64, torch.Tensor):
            print('x64: {}'.format(x64.shape))
        elif isinstance(x64, tuple):
            tuple_shapes = '('
            for item in x64:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x64: {}'.format(tuple_shapes))
        else:
            print('x64: {}'.format(x64))
        x65=self.layernorm6(x64)
        if x65 is None:
            print('x65: {}'.format(x65))
        elif isinstance(x65, torch.Tensor):
            print('x65: {}'.format(x65.shape))
        elif isinstance(x65, tuple):
            tuple_shapes = '('
            for item in x65:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x65: {}'.format(tuple_shapes))
        else:
            print('x65: {}'.format(x65))
        x68=operator.getitem(self.relative_position_bias_table2, self.relative_position_index2)
        if x68 is None:
            print('x68: {}'.format(x68))
        elif isinstance(x68, torch.Tensor):
            print('x68: {}'.format(x68.shape))
        elif isinstance(x68, tuple):
            tuple_shapes = '('
            for item in x68:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x68: {}'.format(tuple_shapes))
        else:
            print('x68: {}'.format(x68))
        x69=x68.view(49, 49, -1)
        if x69 is None:
            print('x69: {}'.format(x69))
        elif isinstance(x69, torch.Tensor):
            print('x69: {}'.format(x69.shape))
        elif isinstance(x69, tuple):
            tuple_shapes = '('
            for item in x69:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x69: {}'.format(tuple_shapes))
        else:
            print('x69: {}'.format(x69))
        x70=x69.permute(2, 0, 1)
        if x70 is None:
            print('x70: {}'.format(x70))
        elif isinstance(x70, torch.Tensor):
            print('x70: {}'.format(x70.shape))
        elif isinstance(x70, tuple):
            tuple_shapes = '('
            for item in x70:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x70: {}'.format(tuple_shapes))
        else:
            print('x70: {}'.format(x70))
        x71=x70.contiguous()
        if x71 is None:
            print('x71: {}'.format(x71))
        elif isinstance(x71, torch.Tensor):
            print('x71: {}'.format(x71.shape))
        elif isinstance(x71, tuple):
            tuple_shapes = '('
            for item in x71:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x71: {}'.format(tuple_shapes))
        else:
            print('x71: {}'.format(x71))
        x72=x71.unsqueeze(0)
        if x72 is None:
            print('x72: {}'.format(x72))
        elif isinstance(x72, torch.Tensor):
            print('x72: {}'.format(x72.shape))
        elif isinstance(x72, tuple):
            tuple_shapes = '('
            for item in x72:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x72: {}'.format(tuple_shapes))
        else:
            print('x72: {}'.format(x72))
        x77=torchvision.models.swin_transformer.shifted_window_attention(x65, self.weight4, self.weight5, x72, [7, 7], 6,shift_size=[0, 0], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias4, proj_bias=self.bias5)
        if x77 is None:
            print('x77: {}'.format(x77))
        elif isinstance(x77, torch.Tensor):
            print('x77: {}'.format(x77.shape))
        elif isinstance(x77, tuple):
            tuple_shapes = '('
            for item in x77:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x77: {}'.format(tuple_shapes))
        else:
            print('x77: {}'.format(x77))
        x78=stochastic_depth(x77, 0.03636363636363637, 'row', False)
        if x78 is None:
            print('x78: {}'.format(x78))
        elif isinstance(x78, torch.Tensor):
            print('x78: {}'.format(x78.shape))
        elif isinstance(x78, tuple):
            tuple_shapes = '('
            for item in x78:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x78: {}'.format(tuple_shapes))
        else:
            print('x78: {}'.format(x78))
        x79=operator.add(x64, x78)
        if x79 is None:
            print('x79: {}'.format(x79))
        elif isinstance(x79, torch.Tensor):
            print('x79: {}'.format(x79.shape))
        elif isinstance(x79, tuple):
            tuple_shapes = '('
            for item in x79:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x79: {}'.format(tuple_shapes))
        else:
            print('x79: {}'.format(x79))
        x80=self.layernorm7(x79)
        if x80 is None:
            print('x80: {}'.format(x80))
        elif isinstance(x80, torch.Tensor):
            print('x80: {}'.format(x80.shape))
        elif isinstance(x80, tuple):
            tuple_shapes = '('
            for item in x80:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x80: {}'.format(tuple_shapes))
        else:
            print('x80: {}'.format(x80))
        x81=self.linear5(x80)
        if x81 is None:
            print('x81: {}'.format(x81))
        elif isinstance(x81, torch.Tensor):
            print('x81: {}'.format(x81.shape))
        elif isinstance(x81, tuple):
            tuple_shapes = '('
            for item in x81:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x81: {}'.format(tuple_shapes))
        else:
            print('x81: {}'.format(x81))
        x82=self.gelu2(x81)
        if x82 is None:
            print('x82: {}'.format(x82))
        elif isinstance(x82, torch.Tensor):
            print('x82: {}'.format(x82.shape))
        elif isinstance(x82, tuple):
            tuple_shapes = '('
            for item in x82:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x82: {}'.format(tuple_shapes))
        else:
            print('x82: {}'.format(x82))
        x83=self.dropout4(x82)
        if x83 is None:
            print('x83: {}'.format(x83))
        elif isinstance(x83, torch.Tensor):
            print('x83: {}'.format(x83.shape))
        elif isinstance(x83, tuple):
            tuple_shapes = '('
            for item in x83:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x83: {}'.format(tuple_shapes))
        else:
            print('x83: {}'.format(x83))
        x84=self.linear6(x83)
        if x84 is None:
            print('x84: {}'.format(x84))
        elif isinstance(x84, torch.Tensor):
            print('x84: {}'.format(x84.shape))
        elif isinstance(x84, tuple):
            tuple_shapes = '('
            for item in x84:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x84: {}'.format(tuple_shapes))
        else:
            print('x84: {}'.format(x84))
        x85=self.dropout5(x84)
        if x85 is None:
            print('x85: {}'.format(x85))
        elif isinstance(x85, torch.Tensor):
            print('x85: {}'.format(x85.shape))
        elif isinstance(x85, tuple):
            tuple_shapes = '('
            for item in x85:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x85: {}'.format(tuple_shapes))
        else:
            print('x85: {}'.format(x85))
        x86=stochastic_depth(x85, 0.03636363636363637, 'row', False)
        if x86 is None:
            print('x86: {}'.format(x86))
        elif isinstance(x86, torch.Tensor):
            print('x86: {}'.format(x86.shape))
        elif isinstance(x86, tuple):
            tuple_shapes = '('
            for item in x86:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x86: {}'.format(tuple_shapes))
        else:
            print('x86: {}'.format(x86))
        x87=operator.add(x79, x86)
        if x87 is None:
            print('x87: {}'.format(x87))
        elif isinstance(x87, torch.Tensor):
            print('x87: {}'.format(x87.shape))
        elif isinstance(x87, tuple):
            tuple_shapes = '('
            for item in x87:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x87: {}'.format(tuple_shapes))
        else:
            print('x87: {}'.format(x87))
        x88=self.layernorm8(x87)
        if x88 is None:
            print('x88: {}'.format(x88))
        elif isinstance(x88, torch.Tensor):
            print('x88: {}'.format(x88.shape))
        elif isinstance(x88, tuple):
            tuple_shapes = '('
            for item in x88:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x88: {}'.format(tuple_shapes))
        else:
            print('x88: {}'.format(x88))
        x91=operator.getitem(self.relative_position_bias_table3, self.relative_position_index3)
        if x91 is None:
            print('x91: {}'.format(x91))
        elif isinstance(x91, torch.Tensor):
            print('x91: {}'.format(x91.shape))
        elif isinstance(x91, tuple):
            tuple_shapes = '('
            for item in x91:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x91: {}'.format(tuple_shapes))
        else:
            print('x91: {}'.format(x91))
        x92=x91.view(49, 49, -1)
        if x92 is None:
            print('x92: {}'.format(x92))
        elif isinstance(x92, torch.Tensor):
            print('x92: {}'.format(x92.shape))
        elif isinstance(x92, tuple):
            tuple_shapes = '('
            for item in x92:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x92: {}'.format(tuple_shapes))
        else:
            print('x92: {}'.format(x92))
        x93=x92.permute(2, 0, 1)
        if x93 is None:
            print('x93: {}'.format(x93))
        elif isinstance(x93, torch.Tensor):
            print('x93: {}'.format(x93.shape))
        elif isinstance(x93, tuple):
            tuple_shapes = '('
            for item in x93:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x93: {}'.format(tuple_shapes))
        else:
            print('x93: {}'.format(x93))
        x94=x93.contiguous()
        if x94 is None:
            print('x94: {}'.format(x94))
        elif isinstance(x94, torch.Tensor):
            print('x94: {}'.format(x94.shape))
        elif isinstance(x94, tuple):
            tuple_shapes = '('
            for item in x94:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x94: {}'.format(tuple_shapes))
        else:
            print('x94: {}'.format(x94))
        x95=x94.unsqueeze(0)
        if x95 is None:
            print('x95: {}'.format(x95))
        elif isinstance(x95, torch.Tensor):
            print('x95: {}'.format(x95.shape))
        elif isinstance(x95, tuple):
            tuple_shapes = '('
            for item in x95:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x95: {}'.format(tuple_shapes))
        else:
            print('x95: {}'.format(x95))
        x100=torchvision.models.swin_transformer.shifted_window_attention(x88, self.weight6, self.weight7, x95, [7, 7], 6,shift_size=[3, 3], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias6, proj_bias=self.bias7)
        if x100 is None:
            print('x100: {}'.format(x100))
        elif isinstance(x100, torch.Tensor):
            print('x100: {}'.format(x100.shape))
        elif isinstance(x100, tuple):
            tuple_shapes = '('
            for item in x100:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x100: {}'.format(tuple_shapes))
        else:
            print('x100: {}'.format(x100))
        x101=stochastic_depth(x100, 0.05454545454545456, 'row', False)
        if x101 is None:
            print('x101: {}'.format(x101))
        elif isinstance(x101, torch.Tensor):
            print('x101: {}'.format(x101.shape))
        elif isinstance(x101, tuple):
            tuple_shapes = '('
            for item in x101:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x101: {}'.format(tuple_shapes))
        else:
            print('x101: {}'.format(x101))
        x102=operator.add(x87, x101)
        if x102 is None:
            print('x102: {}'.format(x102))
        elif isinstance(x102, torch.Tensor):
            print('x102: {}'.format(x102.shape))
        elif isinstance(x102, tuple):
            tuple_shapes = '('
            for item in x102:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x102: {}'.format(tuple_shapes))
        else:
            print('x102: {}'.format(x102))
        x103=self.layernorm9(x102)
        if x103 is None:
            print('x103: {}'.format(x103))
        elif isinstance(x103, torch.Tensor):
            print('x103: {}'.format(x103.shape))
        elif isinstance(x103, tuple):
            tuple_shapes = '('
            for item in x103:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x103: {}'.format(tuple_shapes))
        else:
            print('x103: {}'.format(x103))
        x104=self.linear7(x103)
        if x104 is None:
            print('x104: {}'.format(x104))
        elif isinstance(x104, torch.Tensor):
            print('x104: {}'.format(x104.shape))
        elif isinstance(x104, tuple):
            tuple_shapes = '('
            for item in x104:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x104: {}'.format(tuple_shapes))
        else:
            print('x104: {}'.format(x104))
        x105=self.gelu3(x104)
        if x105 is None:
            print('x105: {}'.format(x105))
        elif isinstance(x105, torch.Tensor):
            print('x105: {}'.format(x105.shape))
        elif isinstance(x105, tuple):
            tuple_shapes = '('
            for item in x105:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x105: {}'.format(tuple_shapes))
        else:
            print('x105: {}'.format(x105))
        x106=self.dropout6(x105)
        if x106 is None:
            print('x106: {}'.format(x106))
        elif isinstance(x106, torch.Tensor):
            print('x106: {}'.format(x106.shape))
        elif isinstance(x106, tuple):
            tuple_shapes = '('
            for item in x106:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x106: {}'.format(tuple_shapes))
        else:
            print('x106: {}'.format(x106))
        x107=self.linear8(x106)
        if x107 is None:
            print('x107: {}'.format(x107))
        elif isinstance(x107, torch.Tensor):
            print('x107: {}'.format(x107.shape))
        elif isinstance(x107, tuple):
            tuple_shapes = '('
            for item in x107:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x107: {}'.format(tuple_shapes))
        else:
            print('x107: {}'.format(x107))
        x108=self.dropout7(x107)
        if x108 is None:
            print('x108: {}'.format(x108))
        elif isinstance(x108, torch.Tensor):
            print('x108: {}'.format(x108.shape))
        elif isinstance(x108, tuple):
            tuple_shapes = '('
            for item in x108:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x108: {}'.format(tuple_shapes))
        else:
            print('x108: {}'.format(x108))
        x109=stochastic_depth(x108, 0.05454545454545456, 'row', False)
        if x109 is None:
            print('x109: {}'.format(x109))
        elif isinstance(x109, torch.Tensor):
            print('x109: {}'.format(x109.shape))
        elif isinstance(x109, tuple):
            tuple_shapes = '('
            for item in x109:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x109: {}'.format(tuple_shapes))
        else:
            print('x109: {}'.format(x109))
        x110=operator.add(x102, x109)
        if x110 is None:
            print('x110: {}'.format(x110))
        elif isinstance(x110, torch.Tensor):
            print('x110: {}'.format(x110.shape))
        elif isinstance(x110, tuple):
            tuple_shapes = '('
            for item in x110:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x110: {}'.format(tuple_shapes))
        else:
            print('x110: {}'.format(x110))
        x111=builtins.getattr(x110, 'shape')
        if x111 is None:
            print('x111: {}'.format(x111))
        elif isinstance(x111, torch.Tensor):
            print('x111: {}'.format(x111.shape))
        elif isinstance(x111, tuple):
            tuple_shapes = '('
            for item in x111:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x111: {}'.format(tuple_shapes))
        else:
            print('x111: {}'.format(x111))
        x112=operator.getitem(x111, slice(-3, None, None))
        if x112 is None:
            print('x112: {}'.format(x112))
        elif isinstance(x112, torch.Tensor):
            print('x112: {}'.format(x112.shape))
        elif isinstance(x112, tuple):
            tuple_shapes = '('
            for item in x112:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x112: {}'.format(tuple_shapes))
        else:
            print('x112: {}'.format(x112))
        x113=operator.getitem(x112, 0)
        if x113 is None:
            print('x113: {}'.format(x113))
        elif isinstance(x113, torch.Tensor):
            print('x113: {}'.format(x113.shape))
        elif isinstance(x113, tuple):
            tuple_shapes = '('
            for item in x113:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x113: {}'.format(tuple_shapes))
        else:
            print('x113: {}'.format(x113))
        x114=operator.getitem(x112, 1)
        if x114 is None:
            print('x114: {}'.format(x114))
        elif isinstance(x114, torch.Tensor):
            print('x114: {}'.format(x114.shape))
        elif isinstance(x114, tuple):
            tuple_shapes = '('
            for item in x114:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x114: {}'.format(tuple_shapes))
        else:
            print('x114: {}'.format(x114))
        x115=operator.getitem(x112, 2)
        if x115 is None:
            print('x115: {}'.format(x115))
        elif isinstance(x115, torch.Tensor):
            print('x115: {}'.format(x115.shape))
        elif isinstance(x115, tuple):
            tuple_shapes = '('
            for item in x115:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x115: {}'.format(tuple_shapes))
        else:
            print('x115: {}'.format(x115))
        x116=operator.mod(x114, 2)
        if x116 is None:
            print('x116: {}'.format(x116))
        elif isinstance(x116, torch.Tensor):
            print('x116: {}'.format(x116.shape))
        elif isinstance(x116, tuple):
            tuple_shapes = '('
            for item in x116:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x116: {}'.format(tuple_shapes))
        else:
            print('x116: {}'.format(x116))
        x117=operator.mod(x113, 2)
        if x117 is None:
            print('x117: {}'.format(x117))
        elif isinstance(x117, torch.Tensor):
            print('x117: {}'.format(x117.shape))
        elif isinstance(x117, tuple):
            tuple_shapes = '('
            for item in x117:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x117: {}'.format(tuple_shapes))
        else:
            print('x117: {}'.format(x117))
        x118=torch.nn.functional.pad(x110, (0, 0, 0, x116, 0, x117))
        if x118 is None:
            print('x118: {}'.format(x118))
        elif isinstance(x118, torch.Tensor):
            print('x118: {}'.format(x118.shape))
        elif isinstance(x118, tuple):
            tuple_shapes = '('
            for item in x118:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x118: {}'.format(tuple_shapes))
        else:
            print('x118: {}'.format(x118))
        x119=operator.getitem(x118, (Ellipsis, slice(0, None, 2), slice(0, None, 2), slice(None, None, None)))
        if x119 is None:
            print('x119: {}'.format(x119))
        elif isinstance(x119, torch.Tensor):
            print('x119: {}'.format(x119.shape))
        elif isinstance(x119, tuple):
            tuple_shapes = '('
            for item in x119:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x119: {}'.format(tuple_shapes))
        else:
            print('x119: {}'.format(x119))
        x120=operator.getitem(x118, (Ellipsis, slice(1, None, 2), slice(0, None, 2), slice(None, None, None)))
        if x120 is None:
            print('x120: {}'.format(x120))
        elif isinstance(x120, torch.Tensor):
            print('x120: {}'.format(x120.shape))
        elif isinstance(x120, tuple):
            tuple_shapes = '('
            for item in x120:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x120: {}'.format(tuple_shapes))
        else:
            print('x120: {}'.format(x120))
        x121=operator.getitem(x118, (Ellipsis, slice(0, None, 2), slice(1, None, 2), slice(None, None, None)))
        if x121 is None:
            print('x121: {}'.format(x121))
        elif isinstance(x121, torch.Tensor):
            print('x121: {}'.format(x121.shape))
        elif isinstance(x121, tuple):
            tuple_shapes = '('
            for item in x121:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x121: {}'.format(tuple_shapes))
        else:
            print('x121: {}'.format(x121))
        x122=operator.getitem(x118, (Ellipsis, slice(1, None, 2), slice(1, None, 2), slice(None, None, None)))
        if x122 is None:
            print('x122: {}'.format(x122))
        elif isinstance(x122, torch.Tensor):
            print('x122: {}'.format(x122.shape))
        elif isinstance(x122, tuple):
            tuple_shapes = '('
            for item in x122:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x122: {}'.format(tuple_shapes))
        else:
            print('x122: {}'.format(x122))
        x123=torch.cat([x119, x120, x121, x122], -1)
        if x123 is None:
            print('x123: {}'.format(x123))
        elif isinstance(x123, torch.Tensor):
            print('x123: {}'.format(x123.shape))
        elif isinstance(x123, tuple):
            tuple_shapes = '('
            for item in x123:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x123: {}'.format(tuple_shapes))
        else:
            print('x123: {}'.format(x123))
        x124=self.layernorm10(x123)
        if x124 is None:
            print('x124: {}'.format(x124))
        elif isinstance(x124, torch.Tensor):
            print('x124: {}'.format(x124.shape))
        elif isinstance(x124, tuple):
            tuple_shapes = '('
            for item in x124:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x124: {}'.format(tuple_shapes))
        else:
            print('x124: {}'.format(x124))
        x125=self.linear9(x124)
        if x125 is None:
            print('x125: {}'.format(x125))
        elif isinstance(x125, torch.Tensor):
            print('x125: {}'.format(x125.shape))
        elif isinstance(x125, tuple):
            tuple_shapes = '('
            for item in x125:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x125: {}'.format(tuple_shapes))
        else:
            print('x125: {}'.format(x125))
        x126=self.layernorm11(x125)
        if x126 is None:
            print('x126: {}'.format(x126))
        elif isinstance(x126, torch.Tensor):
            print('x126: {}'.format(x126.shape))
        elif isinstance(x126, tuple):
            tuple_shapes = '('
            for item in x126:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x126: {}'.format(tuple_shapes))
        else:
            print('x126: {}'.format(x126))
        x129=operator.getitem(self.relative_position_bias_table4, self.relative_position_index4)
        if x129 is None:
            print('x129: {}'.format(x129))
        elif isinstance(x129, torch.Tensor):
            print('x129: {}'.format(x129.shape))
        elif isinstance(x129, tuple):
            tuple_shapes = '('
            for item in x129:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x129: {}'.format(tuple_shapes))
        else:
            print('x129: {}'.format(x129))
        x130=x129.view(49, 49, -1)
        if x130 is None:
            print('x130: {}'.format(x130))
        elif isinstance(x130, torch.Tensor):
            print('x130: {}'.format(x130.shape))
        elif isinstance(x130, tuple):
            tuple_shapes = '('
            for item in x130:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x130: {}'.format(tuple_shapes))
        else:
            print('x130: {}'.format(x130))
        x131=x130.permute(2, 0, 1)
        if x131 is None:
            print('x131: {}'.format(x131))
        elif isinstance(x131, torch.Tensor):
            print('x131: {}'.format(x131.shape))
        elif isinstance(x131, tuple):
            tuple_shapes = '('
            for item in x131:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x131: {}'.format(tuple_shapes))
        else:
            print('x131: {}'.format(x131))
        x132=x131.contiguous()
        if x132 is None:
            print('x132: {}'.format(x132))
        elif isinstance(x132, torch.Tensor):
            print('x132: {}'.format(x132.shape))
        elif isinstance(x132, tuple):
            tuple_shapes = '('
            for item in x132:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x132: {}'.format(tuple_shapes))
        else:
            print('x132: {}'.format(x132))
        x133=x132.unsqueeze(0)
        if x133 is None:
            print('x133: {}'.format(x133))
        elif isinstance(x133, torch.Tensor):
            print('x133: {}'.format(x133.shape))
        elif isinstance(x133, tuple):
            tuple_shapes = '('
            for item in x133:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x133: {}'.format(tuple_shapes))
        else:
            print('x133: {}'.format(x133))
        x138=torchvision.models.swin_transformer.shifted_window_attention(x126, self.weight8, self.weight9, x133, [7, 7], 12,shift_size=[0, 0], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias8, proj_bias=self.bias9)
        if x138 is None:
            print('x138: {}'.format(x138))
        elif isinstance(x138, torch.Tensor):
            print('x138: {}'.format(x138.shape))
        elif isinstance(x138, tuple):
            tuple_shapes = '('
            for item in x138:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x138: {}'.format(tuple_shapes))
        else:
            print('x138: {}'.format(x138))
        x139=stochastic_depth(x138, 0.07272727272727274, 'row', False)
        if x139 is None:
            print('x139: {}'.format(x139))
        elif isinstance(x139, torch.Tensor):
            print('x139: {}'.format(x139.shape))
        elif isinstance(x139, tuple):
            tuple_shapes = '('
            for item in x139:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x139: {}'.format(tuple_shapes))
        else:
            print('x139: {}'.format(x139))
        x140=operator.add(x125, x139)
        if x140 is None:
            print('x140: {}'.format(x140))
        elif isinstance(x140, torch.Tensor):
            print('x140: {}'.format(x140.shape))
        elif isinstance(x140, tuple):
            tuple_shapes = '('
            for item in x140:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x140: {}'.format(tuple_shapes))
        else:
            print('x140: {}'.format(x140))
        x141=self.layernorm12(x140)
        if x141 is None:
            print('x141: {}'.format(x141))
        elif isinstance(x141, torch.Tensor):
            print('x141: {}'.format(x141.shape))
        elif isinstance(x141, tuple):
            tuple_shapes = '('
            for item in x141:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x141: {}'.format(tuple_shapes))
        else:
            print('x141: {}'.format(x141))
        x142=self.linear10(x141)
        if x142 is None:
            print('x142: {}'.format(x142))
        elif isinstance(x142, torch.Tensor):
            print('x142: {}'.format(x142.shape))
        elif isinstance(x142, tuple):
            tuple_shapes = '('
            for item in x142:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x142: {}'.format(tuple_shapes))
        else:
            print('x142: {}'.format(x142))
        x143=self.gelu4(x142)
        if x143 is None:
            print('x143: {}'.format(x143))
        elif isinstance(x143, torch.Tensor):
            print('x143: {}'.format(x143.shape))
        elif isinstance(x143, tuple):
            tuple_shapes = '('
            for item in x143:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x143: {}'.format(tuple_shapes))
        else:
            print('x143: {}'.format(x143))
        x144=self.dropout8(x143)
        if x144 is None:
            print('x144: {}'.format(x144))
        elif isinstance(x144, torch.Tensor):
            print('x144: {}'.format(x144.shape))
        elif isinstance(x144, tuple):
            tuple_shapes = '('
            for item in x144:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x144: {}'.format(tuple_shapes))
        else:
            print('x144: {}'.format(x144))
        x145=self.linear11(x144)
        if x145 is None:
            print('x145: {}'.format(x145))
        elif isinstance(x145, torch.Tensor):
            print('x145: {}'.format(x145.shape))
        elif isinstance(x145, tuple):
            tuple_shapes = '('
            for item in x145:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x145: {}'.format(tuple_shapes))
        else:
            print('x145: {}'.format(x145))
        x146=self.dropout9(x145)
        if x146 is None:
            print('x146: {}'.format(x146))
        elif isinstance(x146, torch.Tensor):
            print('x146: {}'.format(x146.shape))
        elif isinstance(x146, tuple):
            tuple_shapes = '('
            for item in x146:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x146: {}'.format(tuple_shapes))
        else:
            print('x146: {}'.format(x146))
        x147=stochastic_depth(x146, 0.07272727272727274, 'row', False)
        if x147 is None:
            print('x147: {}'.format(x147))
        elif isinstance(x147, torch.Tensor):
            print('x147: {}'.format(x147.shape))
        elif isinstance(x147, tuple):
            tuple_shapes = '('
            for item in x147:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x147: {}'.format(tuple_shapes))
        else:
            print('x147: {}'.format(x147))
        x148=operator.add(x140, x147)
        if x148 is None:
            print('x148: {}'.format(x148))
        elif isinstance(x148, torch.Tensor):
            print('x148: {}'.format(x148.shape))
        elif isinstance(x148, tuple):
            tuple_shapes = '('
            for item in x148:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x148: {}'.format(tuple_shapes))
        else:
            print('x148: {}'.format(x148))
        x149=self.layernorm13(x148)
        if x149 is None:
            print('x149: {}'.format(x149))
        elif isinstance(x149, torch.Tensor):
            print('x149: {}'.format(x149.shape))
        elif isinstance(x149, tuple):
            tuple_shapes = '('
            for item in x149:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x149: {}'.format(tuple_shapes))
        else:
            print('x149: {}'.format(x149))
        x152=operator.getitem(self.relative_position_bias_table5, self.relative_position_index5)
        if x152 is None:
            print('x152: {}'.format(x152))
        elif isinstance(x152, torch.Tensor):
            print('x152: {}'.format(x152.shape))
        elif isinstance(x152, tuple):
            tuple_shapes = '('
            for item in x152:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x152: {}'.format(tuple_shapes))
        else:
            print('x152: {}'.format(x152))
        x153=x152.view(49, 49, -1)
        if x153 is None:
            print('x153: {}'.format(x153))
        elif isinstance(x153, torch.Tensor):
            print('x153: {}'.format(x153.shape))
        elif isinstance(x153, tuple):
            tuple_shapes = '('
            for item in x153:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x153: {}'.format(tuple_shapes))
        else:
            print('x153: {}'.format(x153))
        x154=x153.permute(2, 0, 1)
        if x154 is None:
            print('x154: {}'.format(x154))
        elif isinstance(x154, torch.Tensor):
            print('x154: {}'.format(x154.shape))
        elif isinstance(x154, tuple):
            tuple_shapes = '('
            for item in x154:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x154: {}'.format(tuple_shapes))
        else:
            print('x154: {}'.format(x154))
        x155=x154.contiguous()
        if x155 is None:
            print('x155: {}'.format(x155))
        elif isinstance(x155, torch.Tensor):
            print('x155: {}'.format(x155.shape))
        elif isinstance(x155, tuple):
            tuple_shapes = '('
            for item in x155:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x155: {}'.format(tuple_shapes))
        else:
            print('x155: {}'.format(x155))
        x156=x155.unsqueeze(0)
        if x156 is None:
            print('x156: {}'.format(x156))
        elif isinstance(x156, torch.Tensor):
            print('x156: {}'.format(x156.shape))
        elif isinstance(x156, tuple):
            tuple_shapes = '('
            for item in x156:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x156: {}'.format(tuple_shapes))
        else:
            print('x156: {}'.format(x156))
        x161=torchvision.models.swin_transformer.shifted_window_attention(x149, self.weight10, self.weight11, x156, [7, 7], 12,shift_size=[3, 3], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias10, proj_bias=self.bias11)
        if x161 is None:
            print('x161: {}'.format(x161))
        elif isinstance(x161, torch.Tensor):
            print('x161: {}'.format(x161.shape))
        elif isinstance(x161, tuple):
            tuple_shapes = '('
            for item in x161:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x161: {}'.format(tuple_shapes))
        else:
            print('x161: {}'.format(x161))
        x162=stochastic_depth(x161, 0.09090909090909091, 'row', False)
        if x162 is None:
            print('x162: {}'.format(x162))
        elif isinstance(x162, torch.Tensor):
            print('x162: {}'.format(x162.shape))
        elif isinstance(x162, tuple):
            tuple_shapes = '('
            for item in x162:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x162: {}'.format(tuple_shapes))
        else:
            print('x162: {}'.format(x162))
        x163=operator.add(x148, x162)
        if x163 is None:
            print('x163: {}'.format(x163))
        elif isinstance(x163, torch.Tensor):
            print('x163: {}'.format(x163.shape))
        elif isinstance(x163, tuple):
            tuple_shapes = '('
            for item in x163:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x163: {}'.format(tuple_shapes))
        else:
            print('x163: {}'.format(x163))
        x164=self.layernorm14(x163)
        if x164 is None:
            print('x164: {}'.format(x164))
        elif isinstance(x164, torch.Tensor):
            print('x164: {}'.format(x164.shape))
        elif isinstance(x164, tuple):
            tuple_shapes = '('
            for item in x164:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x164: {}'.format(tuple_shapes))
        else:
            print('x164: {}'.format(x164))
        x165=self.linear12(x164)
        if x165 is None:
            print('x165: {}'.format(x165))
        elif isinstance(x165, torch.Tensor):
            print('x165: {}'.format(x165.shape))
        elif isinstance(x165, tuple):
            tuple_shapes = '('
            for item in x165:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x165: {}'.format(tuple_shapes))
        else:
            print('x165: {}'.format(x165))
        x166=self.gelu5(x165)
        if x166 is None:
            print('x166: {}'.format(x166))
        elif isinstance(x166, torch.Tensor):
            print('x166: {}'.format(x166.shape))
        elif isinstance(x166, tuple):
            tuple_shapes = '('
            for item in x166:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x166: {}'.format(tuple_shapes))
        else:
            print('x166: {}'.format(x166))
        x167=self.dropout10(x166)
        if x167 is None:
            print('x167: {}'.format(x167))
        elif isinstance(x167, torch.Tensor):
            print('x167: {}'.format(x167.shape))
        elif isinstance(x167, tuple):
            tuple_shapes = '('
            for item in x167:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x167: {}'.format(tuple_shapes))
        else:
            print('x167: {}'.format(x167))
        x168=self.linear13(x167)
        if x168 is None:
            print('x168: {}'.format(x168))
        elif isinstance(x168, torch.Tensor):
            print('x168: {}'.format(x168.shape))
        elif isinstance(x168, tuple):
            tuple_shapes = '('
            for item in x168:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x168: {}'.format(tuple_shapes))
        else:
            print('x168: {}'.format(x168))
        x169=self.dropout11(x168)
        if x169 is None:
            print('x169: {}'.format(x169))
        elif isinstance(x169, torch.Tensor):
            print('x169: {}'.format(x169.shape))
        elif isinstance(x169, tuple):
            tuple_shapes = '('
            for item in x169:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x169: {}'.format(tuple_shapes))
        else:
            print('x169: {}'.format(x169))
        x170=stochastic_depth(x169, 0.09090909090909091, 'row', False)
        if x170 is None:
            print('x170: {}'.format(x170))
        elif isinstance(x170, torch.Tensor):
            print('x170: {}'.format(x170.shape))
        elif isinstance(x170, tuple):
            tuple_shapes = '('
            for item in x170:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x170: {}'.format(tuple_shapes))
        else:
            print('x170: {}'.format(x170))
        x171=operator.add(x163, x170)
        if x171 is None:
            print('x171: {}'.format(x171))
        elif isinstance(x171, torch.Tensor):
            print('x171: {}'.format(x171.shape))
        elif isinstance(x171, tuple):
            tuple_shapes = '('
            for item in x171:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x171: {}'.format(tuple_shapes))
        else:
            print('x171: {}'.format(x171))
        x172=self.layernorm15(x171)
        if x172 is None:
            print('x172: {}'.format(x172))
        elif isinstance(x172, torch.Tensor):
            print('x172: {}'.format(x172.shape))
        elif isinstance(x172, tuple):
            tuple_shapes = '('
            for item in x172:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x172: {}'.format(tuple_shapes))
        else:
            print('x172: {}'.format(x172))
        x175=operator.getitem(self.relative_position_bias_table6, self.relative_position_index6)
        if x175 is None:
            print('x175: {}'.format(x175))
        elif isinstance(x175, torch.Tensor):
            print('x175: {}'.format(x175.shape))
        elif isinstance(x175, tuple):
            tuple_shapes = '('
            for item in x175:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x175: {}'.format(tuple_shapes))
        else:
            print('x175: {}'.format(x175))
        x176=x175.view(49, 49, -1)
        if x176 is None:
            print('x176: {}'.format(x176))
        elif isinstance(x176, torch.Tensor):
            print('x176: {}'.format(x176.shape))
        elif isinstance(x176, tuple):
            tuple_shapes = '('
            for item in x176:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x176: {}'.format(tuple_shapes))
        else:
            print('x176: {}'.format(x176))
        x177=x176.permute(2, 0, 1)
        if x177 is None:
            print('x177: {}'.format(x177))
        elif isinstance(x177, torch.Tensor):
            print('x177: {}'.format(x177.shape))
        elif isinstance(x177, tuple):
            tuple_shapes = '('
            for item in x177:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x177: {}'.format(tuple_shapes))
        else:
            print('x177: {}'.format(x177))
        x178=x177.contiguous()
        if x178 is None:
            print('x178: {}'.format(x178))
        elif isinstance(x178, torch.Tensor):
            print('x178: {}'.format(x178.shape))
        elif isinstance(x178, tuple):
            tuple_shapes = '('
            for item in x178:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x178: {}'.format(tuple_shapes))
        else:
            print('x178: {}'.format(x178))
        x179=x178.unsqueeze(0)
        if x179 is None:
            print('x179: {}'.format(x179))
        elif isinstance(x179, torch.Tensor):
            print('x179: {}'.format(x179.shape))
        elif isinstance(x179, tuple):
            tuple_shapes = '('
            for item in x179:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x179: {}'.format(tuple_shapes))
        else:
            print('x179: {}'.format(x179))
        x184=torchvision.models.swin_transformer.shifted_window_attention(x172, self.weight12, self.weight13, x179, [7, 7], 12,shift_size=[0, 0], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias12, proj_bias=self.bias13)
        if x184 is None:
            print('x184: {}'.format(x184))
        elif isinstance(x184, torch.Tensor):
            print('x184: {}'.format(x184.shape))
        elif isinstance(x184, tuple):
            tuple_shapes = '('
            for item in x184:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x184: {}'.format(tuple_shapes))
        else:
            print('x184: {}'.format(x184))
        x185=stochastic_depth(x184, 0.10909090909090911, 'row', False)
        if x185 is None:
            print('x185: {}'.format(x185))
        elif isinstance(x185, torch.Tensor):
            print('x185: {}'.format(x185.shape))
        elif isinstance(x185, tuple):
            tuple_shapes = '('
            for item in x185:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x185: {}'.format(tuple_shapes))
        else:
            print('x185: {}'.format(x185))
        x186=operator.add(x171, x185)
        if x186 is None:
            print('x186: {}'.format(x186))
        elif isinstance(x186, torch.Tensor):
            print('x186: {}'.format(x186.shape))
        elif isinstance(x186, tuple):
            tuple_shapes = '('
            for item in x186:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x186: {}'.format(tuple_shapes))
        else:
            print('x186: {}'.format(x186))
        x187=self.layernorm16(x186)
        if x187 is None:
            print('x187: {}'.format(x187))
        elif isinstance(x187, torch.Tensor):
            print('x187: {}'.format(x187.shape))
        elif isinstance(x187, tuple):
            tuple_shapes = '('
            for item in x187:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x187: {}'.format(tuple_shapes))
        else:
            print('x187: {}'.format(x187))
        x188=self.linear14(x187)
        if x188 is None:
            print('x188: {}'.format(x188))
        elif isinstance(x188, torch.Tensor):
            print('x188: {}'.format(x188.shape))
        elif isinstance(x188, tuple):
            tuple_shapes = '('
            for item in x188:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x188: {}'.format(tuple_shapes))
        else:
            print('x188: {}'.format(x188))
        x189=self.gelu6(x188)
        if x189 is None:
            print('x189: {}'.format(x189))
        elif isinstance(x189, torch.Tensor):
            print('x189: {}'.format(x189.shape))
        elif isinstance(x189, tuple):
            tuple_shapes = '('
            for item in x189:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x189: {}'.format(tuple_shapes))
        else:
            print('x189: {}'.format(x189))
        x190=self.dropout12(x189)
        if x190 is None:
            print('x190: {}'.format(x190))
        elif isinstance(x190, torch.Tensor):
            print('x190: {}'.format(x190.shape))
        elif isinstance(x190, tuple):
            tuple_shapes = '('
            for item in x190:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x190: {}'.format(tuple_shapes))
        else:
            print('x190: {}'.format(x190))
        x191=self.linear15(x190)
        if x191 is None:
            print('x191: {}'.format(x191))
        elif isinstance(x191, torch.Tensor):
            print('x191: {}'.format(x191.shape))
        elif isinstance(x191, tuple):
            tuple_shapes = '('
            for item in x191:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x191: {}'.format(tuple_shapes))
        else:
            print('x191: {}'.format(x191))
        x192=self.dropout13(x191)
        if x192 is None:
            print('x192: {}'.format(x192))
        elif isinstance(x192, torch.Tensor):
            print('x192: {}'.format(x192.shape))
        elif isinstance(x192, tuple):
            tuple_shapes = '('
            for item in x192:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x192: {}'.format(tuple_shapes))
        else:
            print('x192: {}'.format(x192))
        x193=stochastic_depth(x192, 0.10909090909090911, 'row', False)
        if x193 is None:
            print('x193: {}'.format(x193))
        elif isinstance(x193, torch.Tensor):
            print('x193: {}'.format(x193.shape))
        elif isinstance(x193, tuple):
            tuple_shapes = '('
            for item in x193:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x193: {}'.format(tuple_shapes))
        else:
            print('x193: {}'.format(x193))
        x194=operator.add(x186, x193)
        if x194 is None:
            print('x194: {}'.format(x194))
        elif isinstance(x194, torch.Tensor):
            print('x194: {}'.format(x194.shape))
        elif isinstance(x194, tuple):
            tuple_shapes = '('
            for item in x194:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x194: {}'.format(tuple_shapes))
        else:
            print('x194: {}'.format(x194))
        x195=self.layernorm17(x194)
        if x195 is None:
            print('x195: {}'.format(x195))
        elif isinstance(x195, torch.Tensor):
            print('x195: {}'.format(x195.shape))
        elif isinstance(x195, tuple):
            tuple_shapes = '('
            for item in x195:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x195: {}'.format(tuple_shapes))
        else:
            print('x195: {}'.format(x195))
        x198=operator.getitem(self.relative_position_bias_table7, self.relative_position_index7)
        if x198 is None:
            print('x198: {}'.format(x198))
        elif isinstance(x198, torch.Tensor):
            print('x198: {}'.format(x198.shape))
        elif isinstance(x198, tuple):
            tuple_shapes = '('
            for item in x198:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x198: {}'.format(tuple_shapes))
        else:
            print('x198: {}'.format(x198))
        x199=x198.view(49, 49, -1)
        if x199 is None:
            print('x199: {}'.format(x199))
        elif isinstance(x199, torch.Tensor):
            print('x199: {}'.format(x199.shape))
        elif isinstance(x199, tuple):
            tuple_shapes = '('
            for item in x199:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x199: {}'.format(tuple_shapes))
        else:
            print('x199: {}'.format(x199))
        x200=x199.permute(2, 0, 1)
        if x200 is None:
            print('x200: {}'.format(x200))
        elif isinstance(x200, torch.Tensor):
            print('x200: {}'.format(x200.shape))
        elif isinstance(x200, tuple):
            tuple_shapes = '('
            for item in x200:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x200: {}'.format(tuple_shapes))
        else:
            print('x200: {}'.format(x200))
        x201=x200.contiguous()
        if x201 is None:
            print('x201: {}'.format(x201))
        elif isinstance(x201, torch.Tensor):
            print('x201: {}'.format(x201.shape))
        elif isinstance(x201, tuple):
            tuple_shapes = '('
            for item in x201:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x201: {}'.format(tuple_shapes))
        else:
            print('x201: {}'.format(x201))
        x202=x201.unsqueeze(0)
        if x202 is None:
            print('x202: {}'.format(x202))
        elif isinstance(x202, torch.Tensor):
            print('x202: {}'.format(x202.shape))
        elif isinstance(x202, tuple):
            tuple_shapes = '('
            for item in x202:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x202: {}'.format(tuple_shapes))
        else:
            print('x202: {}'.format(x202))
        x207=torchvision.models.swin_transformer.shifted_window_attention(x195, self.weight14, self.weight15, x202, [7, 7], 12,shift_size=[3, 3], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias14, proj_bias=self.bias15)
        if x207 is None:
            print('x207: {}'.format(x207))
        elif isinstance(x207, torch.Tensor):
            print('x207: {}'.format(x207.shape))
        elif isinstance(x207, tuple):
            tuple_shapes = '('
            for item in x207:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x207: {}'.format(tuple_shapes))
        else:
            print('x207: {}'.format(x207))
        x208=stochastic_depth(x207, 0.1272727272727273, 'row', False)
        if x208 is None:
            print('x208: {}'.format(x208))
        elif isinstance(x208, torch.Tensor):
            print('x208: {}'.format(x208.shape))
        elif isinstance(x208, tuple):
            tuple_shapes = '('
            for item in x208:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x208: {}'.format(tuple_shapes))
        else:
            print('x208: {}'.format(x208))
        x209=operator.add(x194, x208)
        if x209 is None:
            print('x209: {}'.format(x209))
        elif isinstance(x209, torch.Tensor):
            print('x209: {}'.format(x209.shape))
        elif isinstance(x209, tuple):
            tuple_shapes = '('
            for item in x209:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x209: {}'.format(tuple_shapes))
        else:
            print('x209: {}'.format(x209))
        x210=self.layernorm18(x209)
        if x210 is None:
            print('x210: {}'.format(x210))
        elif isinstance(x210, torch.Tensor):
            print('x210: {}'.format(x210.shape))
        elif isinstance(x210, tuple):
            tuple_shapes = '('
            for item in x210:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x210: {}'.format(tuple_shapes))
        else:
            print('x210: {}'.format(x210))
        x211=self.linear16(x210)
        if x211 is None:
            print('x211: {}'.format(x211))
        elif isinstance(x211, torch.Tensor):
            print('x211: {}'.format(x211.shape))
        elif isinstance(x211, tuple):
            tuple_shapes = '('
            for item in x211:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x211: {}'.format(tuple_shapes))
        else:
            print('x211: {}'.format(x211))
        x212=self.gelu7(x211)
        if x212 is None:
            print('x212: {}'.format(x212))
        elif isinstance(x212, torch.Tensor):
            print('x212: {}'.format(x212.shape))
        elif isinstance(x212, tuple):
            tuple_shapes = '('
            for item in x212:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x212: {}'.format(tuple_shapes))
        else:
            print('x212: {}'.format(x212))
        x213=self.dropout14(x212)
        if x213 is None:
            print('x213: {}'.format(x213))
        elif isinstance(x213, torch.Tensor):
            print('x213: {}'.format(x213.shape))
        elif isinstance(x213, tuple):
            tuple_shapes = '('
            for item in x213:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x213: {}'.format(tuple_shapes))
        else:
            print('x213: {}'.format(x213))
        x214=self.linear17(x213)
        if x214 is None:
            print('x214: {}'.format(x214))
        elif isinstance(x214, torch.Tensor):
            print('x214: {}'.format(x214.shape))
        elif isinstance(x214, tuple):
            tuple_shapes = '('
            for item in x214:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x214: {}'.format(tuple_shapes))
        else:
            print('x214: {}'.format(x214))
        x215=self.dropout15(x214)
        if x215 is None:
            print('x215: {}'.format(x215))
        elif isinstance(x215, torch.Tensor):
            print('x215: {}'.format(x215.shape))
        elif isinstance(x215, tuple):
            tuple_shapes = '('
            for item in x215:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x215: {}'.format(tuple_shapes))
        else:
            print('x215: {}'.format(x215))
        x216=stochastic_depth(x215, 0.1272727272727273, 'row', False)
        if x216 is None:
            print('x216: {}'.format(x216))
        elif isinstance(x216, torch.Tensor):
            print('x216: {}'.format(x216.shape))
        elif isinstance(x216, tuple):
            tuple_shapes = '('
            for item in x216:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x216: {}'.format(tuple_shapes))
        else:
            print('x216: {}'.format(x216))
        x217=operator.add(x209, x216)
        if x217 is None:
            print('x217: {}'.format(x217))
        elif isinstance(x217, torch.Tensor):
            print('x217: {}'.format(x217.shape))
        elif isinstance(x217, tuple):
            tuple_shapes = '('
            for item in x217:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x217: {}'.format(tuple_shapes))
        else:
            print('x217: {}'.format(x217))
        x218=self.layernorm19(x217)
        if x218 is None:
            print('x218: {}'.format(x218))
        elif isinstance(x218, torch.Tensor):
            print('x218: {}'.format(x218.shape))
        elif isinstance(x218, tuple):
            tuple_shapes = '('
            for item in x218:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x218: {}'.format(tuple_shapes))
        else:
            print('x218: {}'.format(x218))
        x221=operator.getitem(self.relative_position_bias_table8, self.relative_position_index8)
        if x221 is None:
            print('x221: {}'.format(x221))
        elif isinstance(x221, torch.Tensor):
            print('x221: {}'.format(x221.shape))
        elif isinstance(x221, tuple):
            tuple_shapes = '('
            for item in x221:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x221: {}'.format(tuple_shapes))
        else:
            print('x221: {}'.format(x221))
        x222=x221.view(49, 49, -1)
        if x222 is None:
            print('x222: {}'.format(x222))
        elif isinstance(x222, torch.Tensor):
            print('x222: {}'.format(x222.shape))
        elif isinstance(x222, tuple):
            tuple_shapes = '('
            for item in x222:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x222: {}'.format(tuple_shapes))
        else:
            print('x222: {}'.format(x222))
        x223=x222.permute(2, 0, 1)
        if x223 is None:
            print('x223: {}'.format(x223))
        elif isinstance(x223, torch.Tensor):
            print('x223: {}'.format(x223.shape))
        elif isinstance(x223, tuple):
            tuple_shapes = '('
            for item in x223:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x223: {}'.format(tuple_shapes))
        else:
            print('x223: {}'.format(x223))
        x224=x223.contiguous()
        if x224 is None:
            print('x224: {}'.format(x224))
        elif isinstance(x224, torch.Tensor):
            print('x224: {}'.format(x224.shape))
        elif isinstance(x224, tuple):
            tuple_shapes = '('
            for item in x224:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x224: {}'.format(tuple_shapes))
        else:
            print('x224: {}'.format(x224))
        x225=x224.unsqueeze(0)
        if x225 is None:
            print('x225: {}'.format(x225))
        elif isinstance(x225, torch.Tensor):
            print('x225: {}'.format(x225.shape))
        elif isinstance(x225, tuple):
            tuple_shapes = '('
            for item in x225:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x225: {}'.format(tuple_shapes))
        else:
            print('x225: {}'.format(x225))
        x230=torchvision.models.swin_transformer.shifted_window_attention(x218, self.weight16, self.weight17, x225, [7, 7], 12,shift_size=[0, 0], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias16, proj_bias=self.bias17)
        if x230 is None:
            print('x230: {}'.format(x230))
        elif isinstance(x230, torch.Tensor):
            print('x230: {}'.format(x230.shape))
        elif isinstance(x230, tuple):
            tuple_shapes = '('
            for item in x230:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x230: {}'.format(tuple_shapes))
        else:
            print('x230: {}'.format(x230))
        x231=stochastic_depth(x230, 0.14545454545454548, 'row', False)
        if x231 is None:
            print('x231: {}'.format(x231))
        elif isinstance(x231, torch.Tensor):
            print('x231: {}'.format(x231.shape))
        elif isinstance(x231, tuple):
            tuple_shapes = '('
            for item in x231:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x231: {}'.format(tuple_shapes))
        else:
            print('x231: {}'.format(x231))
        x232=operator.add(x217, x231)
        if x232 is None:
            print('x232: {}'.format(x232))
        elif isinstance(x232, torch.Tensor):
            print('x232: {}'.format(x232.shape))
        elif isinstance(x232, tuple):
            tuple_shapes = '('
            for item in x232:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x232: {}'.format(tuple_shapes))
        else:
            print('x232: {}'.format(x232))
        x233=self.layernorm20(x232)
        if x233 is None:
            print('x233: {}'.format(x233))
        elif isinstance(x233, torch.Tensor):
            print('x233: {}'.format(x233.shape))
        elif isinstance(x233, tuple):
            tuple_shapes = '('
            for item in x233:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x233: {}'.format(tuple_shapes))
        else:
            print('x233: {}'.format(x233))
        x234=self.linear18(x233)
        if x234 is None:
            print('x234: {}'.format(x234))
        elif isinstance(x234, torch.Tensor):
            print('x234: {}'.format(x234.shape))
        elif isinstance(x234, tuple):
            tuple_shapes = '('
            for item in x234:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x234: {}'.format(tuple_shapes))
        else:
            print('x234: {}'.format(x234))
        x235=self.gelu8(x234)
        if x235 is None:
            print('x235: {}'.format(x235))
        elif isinstance(x235, torch.Tensor):
            print('x235: {}'.format(x235.shape))
        elif isinstance(x235, tuple):
            tuple_shapes = '('
            for item in x235:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x235: {}'.format(tuple_shapes))
        else:
            print('x235: {}'.format(x235))
        x236=self.dropout16(x235)
        if x236 is None:
            print('x236: {}'.format(x236))
        elif isinstance(x236, torch.Tensor):
            print('x236: {}'.format(x236.shape))
        elif isinstance(x236, tuple):
            tuple_shapes = '('
            for item in x236:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x236: {}'.format(tuple_shapes))
        else:
            print('x236: {}'.format(x236))
        x237=self.linear19(x236)
        if x237 is None:
            print('x237: {}'.format(x237))
        elif isinstance(x237, torch.Tensor):
            print('x237: {}'.format(x237.shape))
        elif isinstance(x237, tuple):
            tuple_shapes = '('
            for item in x237:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x237: {}'.format(tuple_shapes))
        else:
            print('x237: {}'.format(x237))
        x238=self.dropout17(x237)
        if x238 is None:
            print('x238: {}'.format(x238))
        elif isinstance(x238, torch.Tensor):
            print('x238: {}'.format(x238.shape))
        elif isinstance(x238, tuple):
            tuple_shapes = '('
            for item in x238:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x238: {}'.format(tuple_shapes))
        else:
            print('x238: {}'.format(x238))
        x239=stochastic_depth(x238, 0.14545454545454548, 'row', False)
        if x239 is None:
            print('x239: {}'.format(x239))
        elif isinstance(x239, torch.Tensor):
            print('x239: {}'.format(x239.shape))
        elif isinstance(x239, tuple):
            tuple_shapes = '('
            for item in x239:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x239: {}'.format(tuple_shapes))
        else:
            print('x239: {}'.format(x239))
        x240=operator.add(x232, x239)
        if x240 is None:
            print('x240: {}'.format(x240))
        elif isinstance(x240, torch.Tensor):
            print('x240: {}'.format(x240.shape))
        elif isinstance(x240, tuple):
            tuple_shapes = '('
            for item in x240:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x240: {}'.format(tuple_shapes))
        else:
            print('x240: {}'.format(x240))
        x241=self.layernorm21(x240)
        if x241 is None:
            print('x241: {}'.format(x241))
        elif isinstance(x241, torch.Tensor):
            print('x241: {}'.format(x241.shape))
        elif isinstance(x241, tuple):
            tuple_shapes = '('
            for item in x241:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x241: {}'.format(tuple_shapes))
        else:
            print('x241: {}'.format(x241))
        x244=operator.getitem(self.relative_position_bias_table9, self.relative_position_index9)
        if x244 is None:
            print('x244: {}'.format(x244))
        elif isinstance(x244, torch.Tensor):
            print('x244: {}'.format(x244.shape))
        elif isinstance(x244, tuple):
            tuple_shapes = '('
            for item in x244:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x244: {}'.format(tuple_shapes))
        else:
            print('x244: {}'.format(x244))
        x245=x244.view(49, 49, -1)
        if x245 is None:
            print('x245: {}'.format(x245))
        elif isinstance(x245, torch.Tensor):
            print('x245: {}'.format(x245.shape))
        elif isinstance(x245, tuple):
            tuple_shapes = '('
            for item in x245:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x245: {}'.format(tuple_shapes))
        else:
            print('x245: {}'.format(x245))
        x246=x245.permute(2, 0, 1)
        if x246 is None:
            print('x246: {}'.format(x246))
        elif isinstance(x246, torch.Tensor):
            print('x246: {}'.format(x246.shape))
        elif isinstance(x246, tuple):
            tuple_shapes = '('
            for item in x246:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x246: {}'.format(tuple_shapes))
        else:
            print('x246: {}'.format(x246))
        x247=x246.contiguous()
        if x247 is None:
            print('x247: {}'.format(x247))
        elif isinstance(x247, torch.Tensor):
            print('x247: {}'.format(x247.shape))
        elif isinstance(x247, tuple):
            tuple_shapes = '('
            for item in x247:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x247: {}'.format(tuple_shapes))
        else:
            print('x247: {}'.format(x247))
        x248=x247.unsqueeze(0)
        if x248 is None:
            print('x248: {}'.format(x248))
        elif isinstance(x248, torch.Tensor):
            print('x248: {}'.format(x248.shape))
        elif isinstance(x248, tuple):
            tuple_shapes = '('
            for item in x248:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x248: {}'.format(tuple_shapes))
        else:
            print('x248: {}'.format(x248))
        x253=torchvision.models.swin_transformer.shifted_window_attention(x241, self.weight18, self.weight19, x248, [7, 7], 12,shift_size=[3, 3], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias18, proj_bias=self.bias19)
        if x253 is None:
            print('x253: {}'.format(x253))
        elif isinstance(x253, torch.Tensor):
            print('x253: {}'.format(x253.shape))
        elif isinstance(x253, tuple):
            tuple_shapes = '('
            for item in x253:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x253: {}'.format(tuple_shapes))
        else:
            print('x253: {}'.format(x253))
        x254=stochastic_depth(x253, 0.16363636363636364, 'row', False)
        if x254 is None:
            print('x254: {}'.format(x254))
        elif isinstance(x254, torch.Tensor):
            print('x254: {}'.format(x254.shape))
        elif isinstance(x254, tuple):
            tuple_shapes = '('
            for item in x254:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x254: {}'.format(tuple_shapes))
        else:
            print('x254: {}'.format(x254))
        x255=operator.add(x240, x254)
        if x255 is None:
            print('x255: {}'.format(x255))
        elif isinstance(x255, torch.Tensor):
            print('x255: {}'.format(x255.shape))
        elif isinstance(x255, tuple):
            tuple_shapes = '('
            for item in x255:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x255: {}'.format(tuple_shapes))
        else:
            print('x255: {}'.format(x255))
        x256=self.layernorm22(x255)
        if x256 is None:
            print('x256: {}'.format(x256))
        elif isinstance(x256, torch.Tensor):
            print('x256: {}'.format(x256.shape))
        elif isinstance(x256, tuple):
            tuple_shapes = '('
            for item in x256:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x256: {}'.format(tuple_shapes))
        else:
            print('x256: {}'.format(x256))
        x257=self.linear20(x256)
        if x257 is None:
            print('x257: {}'.format(x257))
        elif isinstance(x257, torch.Tensor):
            print('x257: {}'.format(x257.shape))
        elif isinstance(x257, tuple):
            tuple_shapes = '('
            for item in x257:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x257: {}'.format(tuple_shapes))
        else:
            print('x257: {}'.format(x257))
        x258=self.gelu9(x257)
        if x258 is None:
            print('x258: {}'.format(x258))
        elif isinstance(x258, torch.Tensor):
            print('x258: {}'.format(x258.shape))
        elif isinstance(x258, tuple):
            tuple_shapes = '('
            for item in x258:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x258: {}'.format(tuple_shapes))
        else:
            print('x258: {}'.format(x258))
        x259=self.dropout18(x258)
        if x259 is None:
            print('x259: {}'.format(x259))
        elif isinstance(x259, torch.Tensor):
            print('x259: {}'.format(x259.shape))
        elif isinstance(x259, tuple):
            tuple_shapes = '('
            for item in x259:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x259: {}'.format(tuple_shapes))
        else:
            print('x259: {}'.format(x259))
        x260=self.linear21(x259)
        if x260 is None:
            print('x260: {}'.format(x260))
        elif isinstance(x260, torch.Tensor):
            print('x260: {}'.format(x260.shape))
        elif isinstance(x260, tuple):
            tuple_shapes = '('
            for item in x260:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x260: {}'.format(tuple_shapes))
        else:
            print('x260: {}'.format(x260))
        x261=self.dropout19(x260)
        if x261 is None:
            print('x261: {}'.format(x261))
        elif isinstance(x261, torch.Tensor):
            print('x261: {}'.format(x261.shape))
        elif isinstance(x261, tuple):
            tuple_shapes = '('
            for item in x261:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x261: {}'.format(tuple_shapes))
        else:
            print('x261: {}'.format(x261))
        x262=stochastic_depth(x261, 0.16363636363636364, 'row', False)
        if x262 is None:
            print('x262: {}'.format(x262))
        elif isinstance(x262, torch.Tensor):
            print('x262: {}'.format(x262.shape))
        elif isinstance(x262, tuple):
            tuple_shapes = '('
            for item in x262:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x262: {}'.format(tuple_shapes))
        else:
            print('x262: {}'.format(x262))
        x263=operator.add(x255, x262)
        if x263 is None:
            print('x263: {}'.format(x263))
        elif isinstance(x263, torch.Tensor):
            print('x263: {}'.format(x263.shape))
        elif isinstance(x263, tuple):
            tuple_shapes = '('
            for item in x263:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x263: {}'.format(tuple_shapes))
        else:
            print('x263: {}'.format(x263))
        x264=builtins.getattr(x263, 'shape')
        if x264 is None:
            print('x264: {}'.format(x264))
        elif isinstance(x264, torch.Tensor):
            print('x264: {}'.format(x264.shape))
        elif isinstance(x264, tuple):
            tuple_shapes = '('
            for item in x264:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x264: {}'.format(tuple_shapes))
        else:
            print('x264: {}'.format(x264))
        x265=operator.getitem(x264, slice(-3, None, None))
        if x265 is None:
            print('x265: {}'.format(x265))
        elif isinstance(x265, torch.Tensor):
            print('x265: {}'.format(x265.shape))
        elif isinstance(x265, tuple):
            tuple_shapes = '('
            for item in x265:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x265: {}'.format(tuple_shapes))
        else:
            print('x265: {}'.format(x265))
        x266=operator.getitem(x265, 0)
        if x266 is None:
            print('x266: {}'.format(x266))
        elif isinstance(x266, torch.Tensor):
            print('x266: {}'.format(x266.shape))
        elif isinstance(x266, tuple):
            tuple_shapes = '('
            for item in x266:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x266: {}'.format(tuple_shapes))
        else:
            print('x266: {}'.format(x266))
        x267=operator.getitem(x265, 1)
        if x267 is None:
            print('x267: {}'.format(x267))
        elif isinstance(x267, torch.Tensor):
            print('x267: {}'.format(x267.shape))
        elif isinstance(x267, tuple):
            tuple_shapes = '('
            for item in x267:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x267: {}'.format(tuple_shapes))
        else:
            print('x267: {}'.format(x267))
        x268=operator.getitem(x265, 2)
        if x268 is None:
            print('x268: {}'.format(x268))
        elif isinstance(x268, torch.Tensor):
            print('x268: {}'.format(x268.shape))
        elif isinstance(x268, tuple):
            tuple_shapes = '('
            for item in x268:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x268: {}'.format(tuple_shapes))
        else:
            print('x268: {}'.format(x268))
        x269=operator.mod(x267, 2)
        if x269 is None:
            print('x269: {}'.format(x269))
        elif isinstance(x269, torch.Tensor):
            print('x269: {}'.format(x269.shape))
        elif isinstance(x269, tuple):
            tuple_shapes = '('
            for item in x269:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x269: {}'.format(tuple_shapes))
        else:
            print('x269: {}'.format(x269))
        x270=operator.mod(x266, 2)
        if x270 is None:
            print('x270: {}'.format(x270))
        elif isinstance(x270, torch.Tensor):
            print('x270: {}'.format(x270.shape))
        elif isinstance(x270, tuple):
            tuple_shapes = '('
            for item in x270:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x270: {}'.format(tuple_shapes))
        else:
            print('x270: {}'.format(x270))
        x271=torch.nn.functional.pad(x263, (0, 0, 0, x269, 0, x270))
        if x271 is None:
            print('x271: {}'.format(x271))
        elif isinstance(x271, torch.Tensor):
            print('x271: {}'.format(x271.shape))
        elif isinstance(x271, tuple):
            tuple_shapes = '('
            for item in x271:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x271: {}'.format(tuple_shapes))
        else:
            print('x271: {}'.format(x271))
        x272=operator.getitem(x271, (Ellipsis, slice(0, None, 2), slice(0, None, 2), slice(None, None, None)))
        if x272 is None:
            print('x272: {}'.format(x272))
        elif isinstance(x272, torch.Tensor):
            print('x272: {}'.format(x272.shape))
        elif isinstance(x272, tuple):
            tuple_shapes = '('
            for item in x272:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x272: {}'.format(tuple_shapes))
        else:
            print('x272: {}'.format(x272))
        x273=operator.getitem(x271, (Ellipsis, slice(1, None, 2), slice(0, None, 2), slice(None, None, None)))
        if x273 is None:
            print('x273: {}'.format(x273))
        elif isinstance(x273, torch.Tensor):
            print('x273: {}'.format(x273.shape))
        elif isinstance(x273, tuple):
            tuple_shapes = '('
            for item in x273:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x273: {}'.format(tuple_shapes))
        else:
            print('x273: {}'.format(x273))
        x274=operator.getitem(x271, (Ellipsis, slice(0, None, 2), slice(1, None, 2), slice(None, None, None)))
        if x274 is None:
            print('x274: {}'.format(x274))
        elif isinstance(x274, torch.Tensor):
            print('x274: {}'.format(x274.shape))
        elif isinstance(x274, tuple):
            tuple_shapes = '('
            for item in x274:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x274: {}'.format(tuple_shapes))
        else:
            print('x274: {}'.format(x274))
        x275=operator.getitem(x271, (Ellipsis, slice(1, None, 2), slice(1, None, 2), slice(None, None, None)))
        if x275 is None:
            print('x275: {}'.format(x275))
        elif isinstance(x275, torch.Tensor):
            print('x275: {}'.format(x275.shape))
        elif isinstance(x275, tuple):
            tuple_shapes = '('
            for item in x275:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x275: {}'.format(tuple_shapes))
        else:
            print('x275: {}'.format(x275))
        x276=torch.cat([x272, x273, x274, x275], -1)
        if x276 is None:
            print('x276: {}'.format(x276))
        elif isinstance(x276, torch.Tensor):
            print('x276: {}'.format(x276.shape))
        elif isinstance(x276, tuple):
            tuple_shapes = '('
            for item in x276:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x276: {}'.format(tuple_shapes))
        else:
            print('x276: {}'.format(x276))
        x277=self.layernorm23(x276)
        if x277 is None:
            print('x277: {}'.format(x277))
        elif isinstance(x277, torch.Tensor):
            print('x277: {}'.format(x277.shape))
        elif isinstance(x277, tuple):
            tuple_shapes = '('
            for item in x277:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x277: {}'.format(tuple_shapes))
        else:
            print('x277: {}'.format(x277))
        x278=self.linear22(x277)
        if x278 is None:
            print('x278: {}'.format(x278))
        elif isinstance(x278, torch.Tensor):
            print('x278: {}'.format(x278.shape))
        elif isinstance(x278, tuple):
            tuple_shapes = '('
            for item in x278:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x278: {}'.format(tuple_shapes))
        else:
            print('x278: {}'.format(x278))
        x279=self.layernorm24(x278)
        if x279 is None:
            print('x279: {}'.format(x279))
        elif isinstance(x279, torch.Tensor):
            print('x279: {}'.format(x279.shape))
        elif isinstance(x279, tuple):
            tuple_shapes = '('
            for item in x279:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x279: {}'.format(tuple_shapes))
        else:
            print('x279: {}'.format(x279))
        x282=operator.getitem(self.relative_position_bias_table10, self.relative_position_index10)
        if x282 is None:
            print('x282: {}'.format(x282))
        elif isinstance(x282, torch.Tensor):
            print('x282: {}'.format(x282.shape))
        elif isinstance(x282, tuple):
            tuple_shapes = '('
            for item in x282:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x282: {}'.format(tuple_shapes))
        else:
            print('x282: {}'.format(x282))
        x283=x282.view(49, 49, -1)
        if x283 is None:
            print('x283: {}'.format(x283))
        elif isinstance(x283, torch.Tensor):
            print('x283: {}'.format(x283.shape))
        elif isinstance(x283, tuple):
            tuple_shapes = '('
            for item in x283:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x283: {}'.format(tuple_shapes))
        else:
            print('x283: {}'.format(x283))
        x284=x283.permute(2, 0, 1)
        if x284 is None:
            print('x284: {}'.format(x284))
        elif isinstance(x284, torch.Tensor):
            print('x284: {}'.format(x284.shape))
        elif isinstance(x284, tuple):
            tuple_shapes = '('
            for item in x284:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x284: {}'.format(tuple_shapes))
        else:
            print('x284: {}'.format(x284))
        x285=x284.contiguous()
        if x285 is None:
            print('x285: {}'.format(x285))
        elif isinstance(x285, torch.Tensor):
            print('x285: {}'.format(x285.shape))
        elif isinstance(x285, tuple):
            tuple_shapes = '('
            for item in x285:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x285: {}'.format(tuple_shapes))
        else:
            print('x285: {}'.format(x285))
        x286=x285.unsqueeze(0)
        if x286 is None:
            print('x286: {}'.format(x286))
        elif isinstance(x286, torch.Tensor):
            print('x286: {}'.format(x286.shape))
        elif isinstance(x286, tuple):
            tuple_shapes = '('
            for item in x286:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x286: {}'.format(tuple_shapes))
        else:
            print('x286: {}'.format(x286))
        x291=torchvision.models.swin_transformer.shifted_window_attention(x279, self.weight20, self.weight21, x286, [7, 7], 24,shift_size=[0, 0], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias20, proj_bias=self.bias21)
        if x291 is None:
            print('x291: {}'.format(x291))
        elif isinstance(x291, torch.Tensor):
            print('x291: {}'.format(x291.shape))
        elif isinstance(x291, tuple):
            tuple_shapes = '('
            for item in x291:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x291: {}'.format(tuple_shapes))
        else:
            print('x291: {}'.format(x291))
        x292=stochastic_depth(x291, 0.18181818181818182, 'row', False)
        if x292 is None:
            print('x292: {}'.format(x292))
        elif isinstance(x292, torch.Tensor):
            print('x292: {}'.format(x292.shape))
        elif isinstance(x292, tuple):
            tuple_shapes = '('
            for item in x292:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x292: {}'.format(tuple_shapes))
        else:
            print('x292: {}'.format(x292))
        x293=operator.add(x278, x292)
        if x293 is None:
            print('x293: {}'.format(x293))
        elif isinstance(x293, torch.Tensor):
            print('x293: {}'.format(x293.shape))
        elif isinstance(x293, tuple):
            tuple_shapes = '('
            for item in x293:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x293: {}'.format(tuple_shapes))
        else:
            print('x293: {}'.format(x293))
        x294=self.layernorm25(x293)
        if x294 is None:
            print('x294: {}'.format(x294))
        elif isinstance(x294, torch.Tensor):
            print('x294: {}'.format(x294.shape))
        elif isinstance(x294, tuple):
            tuple_shapes = '('
            for item in x294:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x294: {}'.format(tuple_shapes))
        else:
            print('x294: {}'.format(x294))
        x295=self.linear23(x294)
        if x295 is None:
            print('x295: {}'.format(x295))
        elif isinstance(x295, torch.Tensor):
            print('x295: {}'.format(x295.shape))
        elif isinstance(x295, tuple):
            tuple_shapes = '('
            for item in x295:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x295: {}'.format(tuple_shapes))
        else:
            print('x295: {}'.format(x295))
        x296=self.gelu10(x295)
        if x296 is None:
            print('x296: {}'.format(x296))
        elif isinstance(x296, torch.Tensor):
            print('x296: {}'.format(x296.shape))
        elif isinstance(x296, tuple):
            tuple_shapes = '('
            for item in x296:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x296: {}'.format(tuple_shapes))
        else:
            print('x296: {}'.format(x296))
        x297=self.dropout20(x296)
        if x297 is None:
            print('x297: {}'.format(x297))
        elif isinstance(x297, torch.Tensor):
            print('x297: {}'.format(x297.shape))
        elif isinstance(x297, tuple):
            tuple_shapes = '('
            for item in x297:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x297: {}'.format(tuple_shapes))
        else:
            print('x297: {}'.format(x297))
        x298=self.linear24(x297)
        if x298 is None:
            print('x298: {}'.format(x298))
        elif isinstance(x298, torch.Tensor):
            print('x298: {}'.format(x298.shape))
        elif isinstance(x298, tuple):
            tuple_shapes = '('
            for item in x298:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x298: {}'.format(tuple_shapes))
        else:
            print('x298: {}'.format(x298))
        x299=self.dropout21(x298)
        if x299 is None:
            print('x299: {}'.format(x299))
        elif isinstance(x299, torch.Tensor):
            print('x299: {}'.format(x299.shape))
        elif isinstance(x299, tuple):
            tuple_shapes = '('
            for item in x299:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x299: {}'.format(tuple_shapes))
        else:
            print('x299: {}'.format(x299))
        x300=stochastic_depth(x299, 0.18181818181818182, 'row', False)
        if x300 is None:
            print('x300: {}'.format(x300))
        elif isinstance(x300, torch.Tensor):
            print('x300: {}'.format(x300.shape))
        elif isinstance(x300, tuple):
            tuple_shapes = '('
            for item in x300:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x300: {}'.format(tuple_shapes))
        else:
            print('x300: {}'.format(x300))
        x301=operator.add(x293, x300)
        if x301 is None:
            print('x301: {}'.format(x301))
        elif isinstance(x301, torch.Tensor):
            print('x301: {}'.format(x301.shape))
        elif isinstance(x301, tuple):
            tuple_shapes = '('
            for item in x301:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x301: {}'.format(tuple_shapes))
        else:
            print('x301: {}'.format(x301))
        x302=self.layernorm26(x301)
        if x302 is None:
            print('x302: {}'.format(x302))
        elif isinstance(x302, torch.Tensor):
            print('x302: {}'.format(x302.shape))
        elif isinstance(x302, tuple):
            tuple_shapes = '('
            for item in x302:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x302: {}'.format(tuple_shapes))
        else:
            print('x302: {}'.format(x302))
        x305=operator.getitem(self.relative_position_bias_table11, self.relative_position_index11)
        if x305 is None:
            print('x305: {}'.format(x305))
        elif isinstance(x305, torch.Tensor):
            print('x305: {}'.format(x305.shape))
        elif isinstance(x305, tuple):
            tuple_shapes = '('
            for item in x305:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x305: {}'.format(tuple_shapes))
        else:
            print('x305: {}'.format(x305))
        x306=x305.view(49, 49, -1)
        if x306 is None:
            print('x306: {}'.format(x306))
        elif isinstance(x306, torch.Tensor):
            print('x306: {}'.format(x306.shape))
        elif isinstance(x306, tuple):
            tuple_shapes = '('
            for item in x306:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x306: {}'.format(tuple_shapes))
        else:
            print('x306: {}'.format(x306))
        x307=x306.permute(2, 0, 1)
        if x307 is None:
            print('x307: {}'.format(x307))
        elif isinstance(x307, torch.Tensor):
            print('x307: {}'.format(x307.shape))
        elif isinstance(x307, tuple):
            tuple_shapes = '('
            for item in x307:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x307: {}'.format(tuple_shapes))
        else:
            print('x307: {}'.format(x307))
        x308=x307.contiguous()
        if x308 is None:
            print('x308: {}'.format(x308))
        elif isinstance(x308, torch.Tensor):
            print('x308: {}'.format(x308.shape))
        elif isinstance(x308, tuple):
            tuple_shapes = '('
            for item in x308:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x308: {}'.format(tuple_shapes))
        else:
            print('x308: {}'.format(x308))
        x309=x308.unsqueeze(0)
        if x309 is None:
            print('x309: {}'.format(x309))
        elif isinstance(x309, torch.Tensor):
            print('x309: {}'.format(x309.shape))
        elif isinstance(x309, tuple):
            tuple_shapes = '('
            for item in x309:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x309: {}'.format(tuple_shapes))
        else:
            print('x309: {}'.format(x309))
        x314=torchvision.models.swin_transformer.shifted_window_attention(x302, self.weight22, self.weight23, x309, [7, 7], 24,shift_size=[3, 3], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias22, proj_bias=self.bias23)
        if x314 is None:
            print('x314: {}'.format(x314))
        elif isinstance(x314, torch.Tensor):
            print('x314: {}'.format(x314.shape))
        elif isinstance(x314, tuple):
            tuple_shapes = '('
            for item in x314:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x314: {}'.format(tuple_shapes))
        else:
            print('x314: {}'.format(x314))
        x315=stochastic_depth(x314, 0.2, 'row', False)
        if x315 is None:
            print('x315: {}'.format(x315))
        elif isinstance(x315, torch.Tensor):
            print('x315: {}'.format(x315.shape))
        elif isinstance(x315, tuple):
            tuple_shapes = '('
            for item in x315:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x315: {}'.format(tuple_shapes))
        else:
            print('x315: {}'.format(x315))
        x316=operator.add(x301, x315)
        if x316 is None:
            print('x316: {}'.format(x316))
        elif isinstance(x316, torch.Tensor):
            print('x316: {}'.format(x316.shape))
        elif isinstance(x316, tuple):
            tuple_shapes = '('
            for item in x316:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x316: {}'.format(tuple_shapes))
        else:
            print('x316: {}'.format(x316))
        x317=self.layernorm27(x316)
        if x317 is None:
            print('x317: {}'.format(x317))
        elif isinstance(x317, torch.Tensor):
            print('x317: {}'.format(x317.shape))
        elif isinstance(x317, tuple):
            tuple_shapes = '('
            for item in x317:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x317: {}'.format(tuple_shapes))
        else:
            print('x317: {}'.format(x317))
        x318=self.linear25(x317)
        if x318 is None:
            print('x318: {}'.format(x318))
        elif isinstance(x318, torch.Tensor):
            print('x318: {}'.format(x318.shape))
        elif isinstance(x318, tuple):
            tuple_shapes = '('
            for item in x318:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x318: {}'.format(tuple_shapes))
        else:
            print('x318: {}'.format(x318))
        x319=self.gelu11(x318)
        if x319 is None:
            print('x319: {}'.format(x319))
        elif isinstance(x319, torch.Tensor):
            print('x319: {}'.format(x319.shape))
        elif isinstance(x319, tuple):
            tuple_shapes = '('
            for item in x319:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x319: {}'.format(tuple_shapes))
        else:
            print('x319: {}'.format(x319))
        x320=self.dropout22(x319)
        if x320 is None:
            print('x320: {}'.format(x320))
        elif isinstance(x320, torch.Tensor):
            print('x320: {}'.format(x320.shape))
        elif isinstance(x320, tuple):
            tuple_shapes = '('
            for item in x320:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x320: {}'.format(tuple_shapes))
        else:
            print('x320: {}'.format(x320))
        x321=self.linear26(x320)
        if x321 is None:
            print('x321: {}'.format(x321))
        elif isinstance(x321, torch.Tensor):
            print('x321: {}'.format(x321.shape))
        elif isinstance(x321, tuple):
            tuple_shapes = '('
            for item in x321:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x321: {}'.format(tuple_shapes))
        else:
            print('x321: {}'.format(x321))
        x322=self.dropout23(x321)
        if x322 is None:
            print('x322: {}'.format(x322))
        elif isinstance(x322, torch.Tensor):
            print('x322: {}'.format(x322.shape))
        elif isinstance(x322, tuple):
            tuple_shapes = '('
            for item in x322:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x322: {}'.format(tuple_shapes))
        else:
            print('x322: {}'.format(x322))
        x323=stochastic_depth(x322, 0.2, 'row', False)
        if x323 is None:
            print('x323: {}'.format(x323))
        elif isinstance(x323, torch.Tensor):
            print('x323: {}'.format(x323.shape))
        elif isinstance(x323, tuple):
            tuple_shapes = '('
            for item in x323:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x323: {}'.format(tuple_shapes))
        else:
            print('x323: {}'.format(x323))
        x324=operator.add(x316, x323)
        if x324 is None:
            print('x324: {}'.format(x324))
        elif isinstance(x324, torch.Tensor):
            print('x324: {}'.format(x324.shape))
        elif isinstance(x324, tuple):
            tuple_shapes = '('
            for item in x324:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x324: {}'.format(tuple_shapes))
        else:
            print('x324: {}'.format(x324))
        x325=self.layernorm28(x324)
        if x325 is None:
            print('x325: {}'.format(x325))
        elif isinstance(x325, torch.Tensor):
            print('x325: {}'.format(x325.shape))
        elif isinstance(x325, tuple):
            tuple_shapes = '('
            for item in x325:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x325: {}'.format(tuple_shapes))
        else:
            print('x325: {}'.format(x325))
        x326=x325.permute(0, 3, 1, 2)
        if x326 is None:
            print('x326: {}'.format(x326))
        elif isinstance(x326, torch.Tensor):
            print('x326: {}'.format(x326.shape))
        elif isinstance(x326, tuple):
            tuple_shapes = '('
            for item in x326:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x326: {}'.format(tuple_shapes))
        else:
            print('x326: {}'.format(x326))
        x327=self.adaptiveavgpool2d0(x326)
        if x327 is None:
            print('x327: {}'.format(x327))
        elif isinstance(x327, torch.Tensor):
            print('x327: {}'.format(x327.shape))
        elif isinstance(x327, tuple):
            tuple_shapes = '('
            for item in x327:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x327: {}'.format(tuple_shapes))
        else:
            print('x327: {}'.format(x327))
        x328=torch.flatten(x327, 1)
        if x328 is None:
            print('x328: {}'.format(x328))
        elif isinstance(x328, torch.Tensor):
            print('x328: {}'.format(x328.shape))
        elif isinstance(x328, tuple):
            tuple_shapes = '('
            for item in x328:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x328: {}'.format(tuple_shapes))
        else:
            print('x328: {}'.format(x328))
        x329=self.linear27(x328)
        if x329 is None:
            print('x329: {}'.format(x329))
        elif isinstance(x329, torch.Tensor):
            print('x329: {}'.format(x329.shape))
        elif isinstance(x329, tuple):
            tuple_shapes = '('
            for item in x329:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x329: {}'.format(tuple_shapes))
        else:
            print('x329: {}'.format(x329))

m = M().eval()
CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x = torch.rand(1, 3, 224, 224)
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
