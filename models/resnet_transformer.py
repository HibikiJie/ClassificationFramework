import math
import torch
from torch import nn, einsum
from einops import rearrange
from torchvision.models import resnet50


def pair(x):
    return (x, x) if not isinstance(x, tuple) else x


def expand_dim(t, dim, k):
    t = t.unsqueeze(dim=dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def rel_to_abs(x):
    b, h, l, _, device, dtype = *x.shape, x.device, x.dtype
    dd = {'device': device, 'dtype': dtype}
    col_pad = torch.zeros((b, h, l, 1), **dd)
    x = torch.cat((x, col_pad), dim=3)
    flat_x = rearrange(x, 'b h l c -> b h (l c)')
    flat_pad = torch.zeros((b, h, l - 1), **dd)
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
    final_x = flat_x_padded.reshape(b, h, l + 1, 2 * l - 1)
    final_x = final_x[:, :, :l, (l - 1):]
    return final_x


def relative_logits_1d(q, rel_k):
    b, heads, h, w, dim = q.shape
    logits = einsum('b h x y d, r d -> b h x y r', q, rel_k)
    logits = rearrange(logits, 'b h x y r -> b (h x) y r')
    logits = rel_to_abs(logits)
    logits = logits.reshape(b, heads, h, w, w)
    logits = expand_dim(logits, dim=3, k=h)
    return logits


class AbsPosEmb(nn.Module):
    """
    绝对位置编码
    参数：
        fmap_size:特征图的尺寸
        dim_head：注意力每一头的通道数
    """
    def __init__(
            self,
            fmap_size,
            dim_head
    ):
        super().__init__()
        height, width = pair(fmap_size)
        scale = dim_head ** -0.5
        self.height = nn.Parameter(torch.randn(height, dim_head) * scale)
        self.width = nn.Parameter(torch.randn(width, dim_head) * scale)

    def forward(self, q):
        emb = rearrange(self.height, 'h d -> h () d') + rearrange(self.width, 'w d -> () w d')
        emb = rearrange(emb, ' h w d -> (h w) d')
        logits = einsum('b h i d, j d -> b h i j', q, emb)
        return logits


class RelPosEmb(nn.Module):
    """
    相对位置编码
    参数：
        fmap_size:特征图的尺寸
        dim_head：注意力每一头的通道数
    """
    def __init__(
            self,
            fmap_size,
            dim_head
    ):
        super().__init__()
        height, width = pair(fmap_size)
        scale = dim_head ** -0.5
        self.fmap_size = pair(fmap_size)
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        h, w = self.fmap_size

        q = rearrange(q, 'b h (x y) d -> b h x y d', x=h, y=w)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b h x i y j-> b h (x y) (i j)')

        q = rearrange(q, 'b h x y d -> b h y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b h x i y j -> b h (y x) (j i)')
        return rel_logits_w + rel_logits_h


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, channels, feature_map_size, num_head=4, head_channels=128, is_abs=True):
        """

        :param channels: 通道数
        :param feature_map_size: 输入特诊图的尺寸,最大不超过64x64,否则会内存溢出
        :param num_head: 多头注意力的头数
        :param head_channels: 每个头的通道数
        :param is_abs: 是否使用绝对位置编码
        """
        super(MultiHeadSelfAttention, self).__init__()
        assert feature_map_size[0] <= 64, '特诊图的尺寸应该不超过64x64'

        self.channels = channels

        self.num_head = num_head
        self.scale = head_channels ** -0.5
        self.to_qkv = nn.Conv2d(self.channels, num_head * head_channels * 3, 1, bias=False)
        if is_abs:
            self.position = AbsPosEmb(feature_map_size, head_channels)
        else:
            self.position = RelPosEmb(feature_map_size, head_channels)
        self.out_linear = nn.Conv2d(num_head * head_channels, self.channels, 1, bias=False)

    def forward(self, x):
        '''分出q、k、v'''
        n, _, height, weight = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        n, c, h, w = q.shape
        '''n,head v ,x,y  ---> n,head,s,v'''
        """批次，头数，特征数，高度，宽度===>批次，头数，序列长度，特诊数"""
        q = q.reshape(n, self.num_head, c // self.num_head, h, w).permute(0, 1, 3, 4, 2).reshape(n, self.num_head,
                                                                                                 h * w,
                                                                                                 c // self.num_head)
        k = k.reshape(n, self.num_head, c // self.num_head, h, w).permute(0, 1, 3, 4, 2).reshape(n, self.num_head,
                                                                                                 h * w,
                                                                                                 c // self.num_head)
        v = v.reshape(n, self.num_head, c // self.num_head, h, w).permute(0, 1, 3, 4, 2).reshape(n, self.num_head,
                                                                                                 h * w,
                                                                                                 c // self.num_head)
        """计算QK^T,并维度归一化"""
        qk = torch.matmul(q, k.transpose(-1, -2))  # n,h,s,v ===> n,h,s,s
        qk = qk * self.scale
        """获取qr"""
        qr = self.position(q)
        """计算注意力"""
        attention = torch.matmul(torch.softmax(qk + qr, dim=-1), v)
        n, h, s, v = attention.shape
        """变化形状后由输出层输出"""
        attention = attention.permute(0, 1, 3, 2).reshape(n, h * v, height, weight)
        return self.out_linear(attention)


class BottleBlock(nn.Module):
    def __init__(
            self,
            *,
            dim,
            fmap_size,
            dim_out,
            downsample,
            heads=4,
            head_channels=128,
            is_abs=False,
            activation=nn.ReLU()
    ):
        super().__init__()

        # 残差

        if dim != dim_out or downsample:
            kernel_size, stride, padding = (3, 2, 1) if downsample else (1, 1, 0)

            self.shortcut = nn.Sequential(
                nn.Conv2d(dim, dim_out, kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(dim_out),
                activation
            )
        else:
            self.shortcut = nn.Identity()

        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            activation,
            MultiHeadSelfAttention(
                channels=dim,
                feature_map_size=fmap_size,
                num_head=heads,
                head_channels=head_channels,
                is_abs=is_abs),
            nn.AvgPool2d((2, 2)) if downsample else nn.Identity(),
            nn.BatchNorm2d(dim),
            activation,
            nn.Conv2d(dim, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out)
        )

        nn.init.zeros_(self.net[-1].weight)
        self.activation = activation

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.net(x)
        x += shortcut
        return self.activation(x)


class BottleStack(nn.Module):
    def __init__(
            self,
            *,
            dim,
            fmap_size,
            dim_out=2048,
            num_layers=3,
            heads=4,
            dim_head=128,
            downsample=True,
            is_abs=False,
            activation=nn.ReLU()
    ):
        super().__init__()
        fmap_size = pair(fmap_size)

        self.dim = dim
        self.fmap_size = fmap_size

        layers = []

        for i in range(num_layers):
            is_first = i == 0
            dim = (dim if is_first else dim_out)
            layer_downsample = is_first and downsample

            fmap_divisor = (2 if downsample and not is_first else 1)
            layer_fmap_size = tuple(map(lambda t: t // fmap_divisor, fmap_size))

            layers.append(BottleBlock(
                dim=dim,
                fmap_size=layer_fmap_size,
                dim_out=dim_out,
                heads=heads,
                head_channels=dim_head,
                downsample=layer_downsample,
                is_abs=is_abs,
                activation=activation
            ))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        _, c, h, w = x.shape
        assert c == self.dim, f'特诊图的通道 {c} 必须匹配初始化给定的通道数 {self.dim}'
        assert h == self.fmap_size[0] and w == self.fmap_size[
            1], f'特征图的高宽 ({h} {w})  必须匹配初始化网络给定的尺寸 {self.fmap_size}'
        return self.net(x)


class BotNet(nn.Module):

    def __init__(self, out_features):
        super(BotNet, self).__init__()
        layer = BottleStack(
            dim=256,
            fmap_size=28,  # 图片尺寸为 112 x 112时，该位置参数应为28，
            dim_out=512,
            downsample=True,
            heads=4,
            dim_head=128,
            is_abs=True,
            activation=nn.ReLU()
        )
        resnet = resnet50()
        backbone = list(resnet.children())
        self.net = nn.Sequential(
            *backbone[:5],
            layer,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Linear(512, out_features)
        )

    def forward(self, input_):
        return self.net(input_)


if __name__ == '__main__':
    model = BotNet(3)
    img = torch.randn(2, 3, 112, 112)
    print(model)
    preds = model(img)  # (2, 1000)
    print(preds.shape)
    # print(resnet50())
    # m = MultiHeadSelfAttention(256,56,8,128,False)
    # x = torch.randn(2,256,56,56)
    # print(m(x).shape)
