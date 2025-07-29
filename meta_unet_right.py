import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import namedtuple


from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config


class MetaModule(nn.Module):
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                grad = src
                if first_order:
                    grad = self.to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:
            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = self.to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = self.to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)

    def to_var(self, x, requires_grad=True):
        if torch.cuda.is_available():
            x = x.cuda()
        return torch.autograd.Variable(x, requires_grad=requires_grad)



class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv2d(*args, **kwargs)

        self.in_channels = ignore.in_channels
        self.out_channels = ignore.out_channels
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        self.kernel_size = ignore.kernel_size

        self.register_buffer('weight', self.to_var(ignore.weight.data, requires_grad=True))


        if ignore.bias is not None:
            self.register_buffer('bias', self.to_var(ignore.bias.data, requires_grad=True))
        else:
            self.bias = None

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def named_leaves(self):
        leaves = [('weight', self.weight)]
        if self.bias is not None:
            leaves.append(('bias', self.bias))
        return leaves


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', self.to_var(ignore.weight.data, requires_grad=True))
        

        if ignore.bias is not None:
            self.register_buffer('bias', self.to_var(ignore.bias.data, requires_grad=True))
        else:
            self.bias = None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        leaves = [('weight', self.weight)]
        if self.bias is not None:
            leaves.append(('bias', self.bias))
        return leaves


class MetaGroupNorm(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.GroupNorm(*args, **kwargs)

        self.num_groups = ignore.num_groups
        self.num_channels = ignore.num_channels
        self.eps = ignore.eps
        self.affine = ignore.affine

        if self.affine:
            self.register_buffer('weight', self.to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', self.to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('weight', None)
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)

    def named_leaves(self):
        if self.affine:
            return [('weight', self.weight), ('bias', self.bias)]
        return []


class MetaLayerNorm(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.LayerNorm(*args, **kwargs)

        self.normalized_shape = ignore.normalized_shape
        self.eps = ignore.eps
        self.elementwise_affine = ignore.elementwise_affine

        if self.elementwise_affine:
            self.register_buffer('weight', self.to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', self.to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('weight', None)
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

    def named_leaves(self):
        if self.elementwise_affine:
            return [('weight', self.weight), ('bias', self.bias)]
        return []


class MetaDropout(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(*args, **kwargs)

    def forward(self, x):
        return self.dropout(x)

    def named_leaves(self):
        return []



class UNet2DConditionConfig(ConfigMixin):
    model_type = "unet_2d_condition"
    config_name = "unet_2d_condition_config.json" 
    
    @register_to_config
    def __init__(
        self,
        in_channels=4,
        addition_time_embed_dim=256,
        sample_size=128,
        **kwargs
    ):
        super().__init__(**kwargs)






class Timesteps(nn.Module):
    def __init__(self, num_channels: int = 320):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps):
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(
            half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim - 0.0)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        emb = torch.cat([cos_emb, sin_emb], dim=-1)

        return emb


class TimestepEmbedding(MetaModule):
    def __init__(self, in_features, out_features):
        super(TimestepEmbedding, self).__init__()
        self.linear_1 = MetaLinear(in_features, out_features, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = MetaLinear(out_features, out_features, bias=True)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class ResnetBlock2D(MetaModule):
    def __init__(self, in_channels, out_channels, conv_shortcut=True):
        super(ResnetBlock2D, self).__init__()
        self.norm1 = MetaGroupNorm(32, in_channels, eps=1e-05, affine=True)
        self.conv1 = MetaConv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.time_emb_proj = MetaLinear(1280, out_channels, bias=True)
        self.norm2 = MetaGroupNorm(32, out_channels, eps=1e-05, affine=True)
        self.dropout = MetaDropout(p=0.0, inplace=False)
        self.conv2 = MetaConv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.nonlinearity = nn.SiLU()
        self.conv_shortcut = None
        if conv_shortcut:
            self.conv_shortcut = MetaConv2d(
                in_channels, out_channels, kernel_size=1, stride=1
            )

    def forward(self, input_tensor, temb):
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        temb = self.nonlinearity(temb)
        temb = self.time_emb_proj(temb)[:, :, None, None]
        hidden_states = hidden_states + temb
        hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + hidden_states
        return output_tensor


class Attention(MetaModule):
    def __init__(
        self, inner_dim, cross_attention_dim=None, num_heads=None, dropout=0.0
    ):
        super(Attention, self).__init__()
        if num_heads is None:
            self.head_dim = 64
            self.num_heads = inner_dim // self.head_dim
        else:
            self.num_heads = num_heads
            self.head_dim = inner_dim // num_heads

        self.scale = self.head_dim**-0.5
        if cross_attention_dim is None:
            cross_attention_dim = inner_dim
        self.to_q = MetaLinear(inner_dim, inner_dim, bias=False)
        self.to_k = MetaLinear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = MetaLinear(cross_attention_dim, inner_dim, bias=False)

        self.to_out = nn.ModuleList(
            [MetaLinear(inner_dim, inner_dim), MetaDropout(dropout, inplace=False)]
        )

    def forward(self, hidden_states, encoder_hidden_states=None):
        q = self.to_q(hidden_states)
        k = (
            self.to_k(encoder_hidden_states)
            if encoder_hidden_states is not None
            else self.to_k(hidden_states)
        )
        v = (
            self.to_v(encoder_hidden_states)
            if encoder_hidden_states is not None
            else self.to_v(hidden_states)
        )
        b, t, c = q.size()

        q = q.view(q.size(0), q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, t, c)

        for layer in self.to_out:
            attn_output = layer(attn_output)

        return attn_output


class GEGLU(MetaModule):
    def __init__(self, in_features, out_features):
        super(GEGLU, self).__init__()
        self.proj = MetaLinear(in_features, out_features * 2, bias=True)

    def forward(self, x):
        x_proj = self.proj(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        return x1 * torch.nn.functional.gelu(x2)


class FeedForward(MetaModule):
    def __init__(self, in_features, out_features):
        super(FeedForward, self).__init__()

        self.net = nn.ModuleList(
            [
                GEGLU(in_features, out_features * 4),
                MetaDropout(p=0.0, inplace=False),
                MetaLinear(out_features * 4, out_features, bias=True),
            ]
        )

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x


class BasicTransformerBlock(MetaModule):
    def __init__(self, hidden_size):
        super(BasicTransformerBlock, self).__init__()
        self.norm1 = MetaLayerNorm(hidden_size, eps=1e-05, elementwise_affine=True)
        self.attn1 = Attention(hidden_size)
        self.norm2 = MetaLayerNorm(hidden_size, eps=1e-05, elementwise_affine=True)
        self.attn2 = Attention(hidden_size, 2048)
        self.norm3 = MetaLayerNorm(hidden_size, eps=1e-05, elementwise_affine=True)
        self.ff = FeedForward(hidden_size, hidden_size)

    def forward(self, x, encoder_hidden_states=None):
        residual = x

        x = self.norm1(x)
        x = self.attn1(x)
        x = x + residual

        residual = x

        x = self.norm2(x)
        if encoder_hidden_states is not None:
            x = self.attn2(x, encoder_hidden_states)
        else:
            x = self.attn2(x)
        x = x + residual

        residual = x

        x = self.norm3(x)
        x = self.ff(x)
        x = x + residual
        return x


class Transformer2DModel(MetaModule):
    def __init__(self, in_channels, out_channels, n_layers):
        super(Transformer2DModel, self).__init__()
        self.norm = MetaGroupNorm(32, in_channels, eps=1e-06, affine=True)
        self.proj_in = MetaLinear(in_channels, out_channels, bias=True)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(out_channels) for _ in range(n_layers)]
        )
        self.proj_out = MetaLinear(out_channels, out_channels, bias=True)

    def forward(self, hidden_states, encoder_hidden_states=None):
        batch, _, height, width = hidden_states.shape
        res = hidden_states
        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
            batch, height * width, inner_dim
        )
        hidden_states = self.proj_in(hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states)

        hidden_states = self.proj_out(hidden_states)
        hidden_states = (
            hidden_states.reshape(batch, height, width, inner_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return hidden_states + res


class Downsample2D(MetaModule):
    def __init__(self, in_channels, out_channels):
        super(Downsample2D, self).__init__()
        self.conv = MetaConv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x):
        return self.conv(x)


class Upsample2D(MetaModule):
    def __init__(self, in_channels, out_channels):
        super(Upsample2D, self).__init__()
        self.conv = MetaConv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class DownBlock2D(MetaModule):
    def __init__(self, in_channels, out_channels):
        super(DownBlock2D, self).__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(in_channels, out_channels, conv_shortcut=False),
                ResnetBlock2D(out_channels, out_channels, conv_shortcut=False),
            ]
        )
        self.downsamplers = nn.ModuleList([Downsample2D(out_channels, out_channels)])

    def forward(self, hidden_states, temb):
        output_states = []
        for module in self.resnets:
            hidden_states = module(hidden_states, temb)
            output_states.append(hidden_states)

        hidden_states = self.downsamplers[0](hidden_states)
        output_states.append(hidden_states)

        return hidden_states, output_states


class CrossAttnDownBlock2D(MetaModule):
    def __init__(self, in_channels, out_channels, n_layers, has_downsamplers=True):
        super(CrossAttnDownBlock2D, self).__init__()
        self.attentions = nn.ModuleList(
            [
                Transformer2DModel(out_channels, out_channels, n_layers),
                Transformer2DModel(out_channels, out_channels, n_layers),
            ]
        )
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(in_channels, out_channels),
                ResnetBlock2D(out_channels, out_channels, conv_shortcut=False),
            ]
        )
        self.downsamplers = None
        if has_downsamplers:
            self.downsamplers = nn.ModuleList(
                [Downsample2D(out_channels, out_channels)]
            )

    def forward(self, hidden_states, temb, encoder_hidden_states):
        output_states = []
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
            output_states.append(hidden_states)

        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](hidden_states)
            output_states.append(hidden_states)

        return hidden_states, output_states


class CrossAttnUpBlock2D(MetaModule):
    def __init__(self, in_channels, out_channels, prev_output_channel, n_layers):
        super(CrossAttnUpBlock2D, self).__init__()
        self.attentions = nn.ModuleList(
            [
                Transformer2DModel(out_channels, out_channels, n_layers),
                Transformer2DModel(out_channels, out_channels, n_layers),
                Transformer2DModel(out_channels, out_channels, n_layers),
            ]
        )
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(prev_output_channel + out_channels, out_channels),
                ResnetBlock2D(out_channels + out_channels, out_channels),
                ResnetBlock2D(out_channels + in_channels, out_channels),
            ]
        )
        self.upsamplers = nn.ModuleList([Upsample2D(out_channels, out_channels)])

    def forward(
        self, hidden_states, temb, res_hidden_states_tuple, encoder_hidden_states
    ):
        output_states = []
        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
            output_states.append(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class UpBlock2D(MetaModule):
    def __init__(self, in_channels, out_channels, prev_output_channel):
        super(UpBlock2D, self).__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(out_channels + prev_output_channel, out_channels),
                ResnetBlock2D(out_channels + out_channels, out_channels),
                ResnetBlock2D(out_channels + in_channels, out_channels),
            ]
        )
        self.upsamplers = nn.ModuleList([])

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None):
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
        

        return hidden_states


class UNetMidBlock2DCrossAttn(MetaModule):
    def __init__(self, in_features):
        super(UNetMidBlock2DCrossAttn, self).__init__()
        self.attentions = nn.ModuleList(
            [Transformer2DModel(in_features, in_features, n_layers=10)]
        )
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(in_features, in_features, conv_shortcut=False),
                ResnetBlock2D(in_features, in_features, conv_shortcut=False),
            ]
        )

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None):
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class UNet2DConditionModel(MetaModule,ModelMixin, ConfigMixin):
    config_class = UNet2DConditionConfig
    @register_to_config
    def __init__(self, config=None):
        if config is None:
            config = self.config_class()
        super().__init__()


        self.conv_in = MetaConv2d(config.in_channels, 320, kernel_size=3, stride=1, padding=1)
        self.time_proj = Timesteps()
        self.time_embedding = TimestepEmbedding(in_features=320, out_features=1280)
        self.add_time_proj = Timesteps(256)
        self.add_embedding = TimestepEmbedding(in_features=2816, out_features=1280)
        self.down_blocks = nn.ModuleList(
            [
                DownBlock2D(in_channels=320, out_channels=320),
                CrossAttnDownBlock2D(in_channels=320, out_channels=640, n_layers=2),
                CrossAttnDownBlock2D(
                    in_channels=640,
                    out_channels=1280,
                    n_layers=10,
                    has_downsamplers=False,
                ),
            ]
        )
        self.up_blocks = nn.ModuleList(
            [
                CrossAttnUpBlock2D(
                    in_channels=640,
                    out_channels=1280,
                    prev_output_channel=1280,
                    n_layers=10,
                ),
                CrossAttnUpBlock2D(
                    in_channels=320,
                    out_channels=640,
                    prev_output_channel=1280,
                    n_layers=2,
                ),
                UpBlock2D(in_channels=320, out_channels=320, prev_output_channel=640),
            ]
        )
        self.mid_block = UNetMidBlock2DCrossAttn(1280)
        self.conv_norm_out = MetaGroupNorm(32, 320, eps=1e-05, affine=True)
        self.conv_act = nn.SiLU()
        self.conv_out = MetaConv2d(320, 4, kernel_size=3, stride=1, padding=1)




    def forward(
        self, sample, timesteps, encoder_hidden_states, added_cond_kwargs, **kwargs
    ):
        # 实现前向传播
        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb)

        text_embeds = added_cond_kwargs.get("text_embeds")
        time_ids = added_cond_kwargs.get("time_ids")

        time_embeds = self.add_time_proj(time_ids.flatten())
        time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

        add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
        add_embeds = add_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(add_embeds)

        emb = emb + aug_emb

        sample = self.conv_in(sample)


        s0 = sample
        sample, [s1, s2, s3] = self.down_blocks[0](
            sample,
            temb=emb,
        )

        sample, [s4, s5, s6] = self.down_blocks[1](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )

        sample, [s7, s8] = self.down_blocks[2](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )


        sample = self.mid_block(
            sample, emb, encoder_hidden_states=encoder_hidden_states
        )


        sample = self.up_blocks[0](
            hidden_states=sample,
            temb=emb,
            res_hidden_states_tuple=[s6, s7, s8],
            encoder_hidden_states=encoder_hidden_states,
        )

        sample = self.up_blocks[1](
            hidden_states=sample,
            temb=emb,
            res_hidden_states_tuple=[s3, s4, s5],
            encoder_hidden_states=encoder_hidden_states,
        )

        sample = self.up_blocks[2](
            hidden_states=sample,
            temb=emb,
            res_hidden_states_tuple=[s0, s1, s2],
        )


        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return [sample]
    