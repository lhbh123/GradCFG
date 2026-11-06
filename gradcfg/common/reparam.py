import os
import sys
import shutil
import time
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from contextlib import contextmanager


term_width = shutil.get_terminal_size(fallback=(100, 24)).columns

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

# ReparamModule is based on:
# https://github.com/GeorgeCazenavette/mtt-distillation                             

class ReparamModule(nn.Module):
    def _get_module_from_name(self, mn):
        if mn == '':
            return self
        m = self
        for p in mn.split('.'):
            m = getattr(m, p)
        return m

    def __init__(self, module):
        super(ReparamModule, self).__init__()
        self.module = module

        param_infos = []
        shared_param_memo = {}
        shared_param_infos = []
        params = []
        param_numels = []
        param_shapes = []
        for mn, m in self.named_modules():
            for n, p in m.named_parameters(recurse=False):
                if p is not None:
                    if p in shared_param_memo:
                        shared_mn, shared_n = shared_param_memo[p]
                        shared_param_infos.append((mn, n, shared_mn, shared_n))
                    else:
                        shared_param_memo[p] = (mn, n)
                        param_infos.append((mn, n))
                        params.append(p.detach())
                        param_numels.append(p.numel())
                        param_shapes.append(p.size())

        assert len(set(p.dtype for p in params)) <= 1, \
            "expects all parameters in module to have same dtype"

        self._param_infos = tuple(param_infos)
        self._shared_param_infos = tuple(shared_param_infos)
        self._param_numels = tuple(param_numels)
        self._param_shapes = tuple(param_shapes)

        flat_param = nn.Parameter(torch.cat([p.reshape(-1) for p in params], 0))
        self.register_parameter('flat_param', flat_param)
        self.param_numel = flat_param.numel()
        del params
        del shared_param_memo

        for mn, n in self._param_infos:
            delattr(self._get_module_from_name(mn), n)
        for mn, n, _, _ in self._shared_param_infos:
            delattr(self._get_module_from_name(mn), n)

        self._unflatten_param(self.flat_param)

        buffer_infos = []
        for mn, m in self.named_modules():
            for n, b in m.named_buffers(recurse=False):
                if b is not None:
                    buffer_infos.append((mn, n, b))

        self._buffer_infos = tuple(buffer_infos)
        self._traced_self = None

    def trace(self, example_input, **trace_kwargs):
        assert self._traced_self is None, 'This ReparamModule is already traced'

        if isinstance(example_input, torch.Tensor):
            example_input = (example_input,)
        example_input = tuple(example_input)
        example_param = (self.flat_param.detach().clone(),)
        example_buffers = (tuple(b.detach().clone() for _, _, b in self._buffer_infos),)

        self._traced_self = torch.jit.trace_module(
            self,
            inputs=dict(
                _forward_with_param=example_param + example_input,
                _forward_with_param_and_buffers=example_param + example_buffers + example_input,
            ),
            **trace_kwargs,
        )

        self._forward_with_param = self._traced_self._forward_with_param
        self._forward_with_param_and_buffers = self._traced_self._forward_with_param_and_buffers
        return self

    def clear_views(self):
        for mn, n in self._param_infos:
            setattr(self._get_module_from_name(mn), n, None)  # This will set as plain attr

    def _apply(self, *args, **kwargs):
        if self._traced_self is not None:
            self._traced_self._apply(*args, **kwargs)
            return self
        return super(ReparamModule, self)._apply(*args, **kwargs)

    def _unflatten_param(self, flat_param):
        ps = (t.view(s) for (t, s) in zip(flat_param.split(self._param_numels), self._param_shapes))
        for (mn, n), p in zip(self._param_infos, ps):
            setattr(self._get_module_from_name(mn), n, p)  # This will set as plain attr
        for (mn, n, shared_mn, shared_n) in self._shared_param_infos:
            setattr(self._get_module_from_name(mn), n, getattr(self._get_module_from_name(shared_mn), shared_n))

    @contextmanager
    def unflattened_param(self, flat_param):
        saved_views = [getattr(self._get_module_from_name(mn), n) for mn, n in self._param_infos]
        self._unflatten_param(flat_param)
        yield
        for (mn, n), p in zip(self._param_infos, saved_views):
            setattr(self._get_module_from_name(mn), n, p)
        for (mn, n, shared_mn, shared_n) in self._shared_param_infos:
            setattr(self._get_module_from_name(mn), n, getattr(self._get_module_from_name(shared_mn), shared_n))

    @contextmanager
    def replaced_buffers(self, buffers):
        for (mn, n, _), new_b in zip(self._buffer_infos, buffers):
            setattr(self._get_module_from_name(mn), n, new_b)
        yield
        for mn, n, old_b in self._buffer_infos:
            setattr(self._get_module_from_name(mn), n, old_b)

    def _forward_with_param_and_buffers(self, flat_param, buffers, *inputs, **kwinputs):
        with self.unflattened_param(flat_param):
            with self.replaced_buffers(buffers):
                return self.module(*inputs, **kwinputs)

    def _forward_with_param(self, flat_param, *inputs, **kwinputs):
        with self.unflattened_param(flat_param):
            return self.module(*inputs, **kwinputs)

    def forward(self, *inputs, flat_param=None, buffers=None, **kwinputs):
        if flat_param is None:
            flat_param = self.flat_param
        else:
            flat_param = flat_param.reshape(-1)
        if buffers is None:
            return self._forward_with_param(flat_param, *inputs, **kwinputs)
        else:
            return self._forward_with_param_and_buffers(flat_param, tuple(buffers), *inputs, **kwinputs)


def manual_scaled_dot_product_attention(
        query, key, value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None
):
    # 确保输入维度兼容
    assert query.dim() == 4, "Query 形状应为 [batch, heads, seq_len, head_dim]"
    batch_size, num_heads, seq_len, head_dim = query.shape

    # 计算缩放因子（默认使用 head_dim 的平方根）
    scale_factor = scale if scale is not None else head_dim ** -0.5

    # Step 1: 计算 Q 和 K 的点积，并缩放
    attn = torch.matmul(query, key.transpose(-2, -1)) * scale_factor

    # Step 2: 处理因果掩码（如果启用）
    if is_causal:
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=query.device),
            diagonal=1
        )
        attn = attn.masked_fill(causal_mask[None, None, :, :], float('-inf'))

    # Step 3: 处理自定义 attn_mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn = attn.masked_fill(~attn_mask, float('-inf'))  # 将 False 位置设为 -inf
        else:
            attn = attn + attn_mask  # 直接相加（如预先添加的 -inf 掩码）

    # Step 4: Softmax 归一化
    attn_weights = F.softmax(attn, dim=-1)

    # Step 5: 应用 Dropout（训练时）
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # Step 6: 加权求和（乘以 V）
    output = torch.matmul(attn_weights, value)

    return output
