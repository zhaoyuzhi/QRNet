import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter

# ----------------------------------------
#               Conv2d Block
# ----------------------------------------
class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = False):
        super(Conv2dLayer, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        
        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        
        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace = True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace = True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
    
    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class TransposeConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = True, scale_factor = 2):
        super(TransposeConv2dLayer, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.conv2d = Conv2dLayer(in_channels, out_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor = self.scale_factor, mode = 'nearest')
        x = self.conv2d(x)
        return x
        
class ResConv2dLayer(nn.Module):
    def __init__(self, in_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = False):
        super(ResConv2dLayer, self).__init__()
        # Initialize the conv scheme
        self.conv2d = nn.Sequential(
            Conv2dLayer(in_channels, in_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn),
            Conv2dLayer(in_channels, in_channels, kernel_size, stride, padding, dilation, pad_type, activation = 'none', norm = norm, sn = sn)
        )
        
        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace = True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace = True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)
    
    def forward(self, x):
        residual = x
        out = self.conv2d(x)
        out = 0.1 * out + residual
        return out

# ----------------------------------------
#            Group Conv Block
# ----------------------------------------
class DWBase_Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', groups = 1, sn = False):
        super(DWBase_Conv2dLayer, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        
        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        
        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace = True)
        elif activation == 'relu6':
            self.activation = nn.ReLU6(inplace = True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace = True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation, groups = groups, bias = False))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation, groups = groups, bias = False)
    
    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class DWBase_TransposeConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', groups = 1, sn = False, scale_factor = 2):
        super(DWBase_TransposeConv2dLayer, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.conv2d = DWBase_Conv2dLayer(in_channels, out_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, groups, sn)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor = self.scale_factor, mode = 'nearest')
        x = self.conv2d(x)
        return x

class DWConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', groups = 1, sn = False):
        super(DWConv2dLayer, self).__init__()
        self.conv = nn.Sequential(
            # dw
            DWBase_Conv2dLayer(in_channels, in_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, groups = in_channels, sn = sn),
            # pw
            DWBase_Conv2dLayer(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, dilation = dilation, pad_type = pad_type, activation = activation, norm = norm, groups = 1, sn = sn)
        )

    def forward(self, x):
        return self.conv(x)

class DWTransposeConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', groups = 1, sn = False, scale_factor = 2):
        super(DWTransposeConv2dLayer, self).__init__()
        self.conv = nn.Sequential(
            # pw
            DWBase_Conv2dLayer(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, dilation = dilation, pad_type = pad_type, activation = activation, norm = norm, groups = 1, sn = sn),
            # dw
            DWBase_TransposeConv2dLayer(out_channels, out_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, groups = out_channels, sn = sn, scale_factor = scale_factor),
            # pw-linear
            DWBase_Conv2dLayer(out_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, dilation = dilation, pad_type = pad_type, activation = activation, norm = norm, groups = 1, sn = sn)
        )

    def forward(self, x):
        return self.conv(x)

# ----------------------------------------
#            ConvLSTM2d Block
# ----------------------------------------
class ConvLSTM2d(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size = 3):
        super(ConvLSTM2d, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size = self.kernel_size, stride = 1, padding = self.padding)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.shape[0]
        spatial_size = input_.shape[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size)).cuda(),
                Variable(torch.zeros(state_size)).cuda()
            )

        # prev_state has two components
        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension: split it to 4 samples at dimension 1
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell

# ----------------------------------------
#               Layer Norm
# ----------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps = 1e-8, affine = True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = Parameter(torch.Tensor(num_features).uniform_())
            self.beta = Parameter(torch.zeros(num_features))

    def forward(self, x):
        # layer norm
        shape = [-1] + [1] * (x.dim() - 1)                                  # for 4d input: [-1, 1, 1, 1]
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)
        # if it is learnable
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)                          # for 4d input: [1, -1, 1, 1]
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

# ----------------------------------------
#                  AdaIN
# ----------------------------------------
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, eps = 1e-8):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.eps = eps

    def calc_mean_std(self, features):
        """
        :param features: shape of features -> [batch_size, c, h, w]
        :return: features_mean, feature_s: shape of mean/std ->[batch_size, c, 1, 1]
        """
        batch_size, c = features.size()[:2]
        features_mean = features.reshape(batch_size, c, -1).mean(dim = 2).reshape(batch_size, c, 1, 1)
        features_std = features.reshape(batch_size, c, -1).std(dim = 2).reshape(batch_size, c, 1, 1) + self.eps
        return features_mean, features_std

    def adain(self, content_features, style_features):
        """
        Adaptive Instance Normalization
        :param content_features: shape -> [batch_size, c, h, w]
        :param style_features: shape -> [batch_size, c, h, w]
        :return: normalized_features shape -> [batch_size, c, h, w]
        """
        content_mean, content_std = self.calc_mean_std(content_features)
        style_mean, style_std = self.calc_mean_std(style_features)
        normalized_features = style_std * (content_features - content_mean) / content_std + style_mean
        return normalized_features

    def forward(self, content, style):
        stylized_content = self.adain(content, style)
        return stylized_content

# ----------------------------------------
#              PixelShuffle
# ----------------------------------------
class PixelShuffleAlign(nn.Module):
    def __init__(self, upscale_factor: int = 2, mode: str = 'caffe'):
        """
        :param upscale_factor: upsample scale
        :param mode: caffe, pytorch
        """
        super(PixelShuffleAlign, self).__init__()
        self.upscale_factor = upscale_factor
        self.mode = mode

    def forward(self, x):
        # assert len(x.size()) == 4, "Received input tensor shape is {}".format(
        #     x.size())
        if len(x.size()) != 4:
            raise ValueError("input tensor shape {} is not supported.".format(x.size()))
        N, C, H, W = x.size()
        c = C // (self.upscale_factor ** 2)
        h, w = H * self.upscale_factor, W * self.upscale_factor

        if self.mode == 'caffe':
            # (N, C, H, W) => (N, r, r, c, H, W)
            x = x.reshape(-1, self.upscale_factor,
                          self.upscale_factor, c, H, W)
            x = x.permute(0, 3, 4, 1, 5, 2)
        elif self.mode == 'pytorch':
            # (N, C, H, W) => (N, c, r, r, H, W)
            x = x.reshape(-1, c, self.upscale_factor,
                          self.upscale_factor, H, W)
            x = x.permute(0, 1, 4, 2, 5, 3)
        else:
            raise NotImplementedError(
                "{} mode is not implemented".format(self.mode))

        x = x.reshape(-1, c, h, w)
        return x

class PixelUnShuffleAlign(nn.Module):

    def __init__(self, downscale_factor: int = 2, mode: str = 'caffe'):
        """
        :param downscale_factor: downsample scale
        :param mode: caffe, pytorch
        """
        super(PixelUnShuffleAlign, self).__init__()
        self.dsf = downscale_factor
        self.mode = mode

    def forward(self, x):
        if len(x.size()) != 4:
            raise ValueError("input tensor shape {} is not supported.".format(x.size()))
        N, C, H, W = x.size()
        c = int(C * (self.dsf ** 2))
        h, w = H // self.dsf, W // self.dsf

        x = x.reshape(-1, C, h, self.dsf, w, self.dsf)
        if self.mode == 'caffe':
            x = x.permute(0, 3, 5, 1, 2, 4)
        elif self.mode == 'pytorch':
            x = x.permute(0, 1, 3, 5, 2, 4)
        else:
            raise NotImplementedError(
                "{} mode is not implemented".format(self.mode))

        x = x.reshape(-1, c, h, w)
        return x

# ----------------------------------------
#                DWT / IDWT
# ----------------------------------------
def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def idwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

class IDWT(nn.Module):
    def __init__(self):
        super(IDWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return idwt_init(x)

# ----------------------------------------
#           Spectral Norm Block
# ----------------------------------------
def l2normalize(v, eps = 1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

# ----------------------------------------
#              Self Attention
# ----------------------------------------
class SelfAttnModule(nn.Module):
    """ Self attention Layer for Feature Map dimension"""
    def __init__(self, in_dim, latent_dim = 8):
        super(SelfAttnModule, self).__init__()
        self.channel_in = in_dim
        self.channel_latent = in_dim // latent_dim
        self.query_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim // latent_dim, kernel_size = 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim // latent_dim, kernel_size = 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim, kernel_size = 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim = -1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X H X W)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Height * Width)
        """
        batchsize, C, height, width = x.size()
        # proj_query: reshape to B x N x c, N = H x W
        proj_query = self.query_conv(x).view(batchsize, -1, height * width).permute(0, 2, 1)
        # proj_query: reshape to B x c x N, N = H x W
        proj_key = self.key_conv(x).view(batchsize, -1, height * width)
        # transpose check, energy: B x N x N, N = H x W
        energy = torch.bmm(proj_query, proj_key)
        # attention: B x N x N, N = H x W
        attention = self.softmax(energy)
        # proj_value is normal convolution, B x C x N
        proj_value = self.value_conv(x).view(batchsize, -1, height * width)
        # out: B x C x N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batchsize, C, height, width)
        
        out = self.gamma * out + x

        return out

class AttnModule(nn.Module):
    """ Dual attention Layer"""
    def __init__(self, in_dim, latent_dim = 8, efficient = False, pool = False):
        super(AttnModule, self).__init__()
        # if efficient is True, use adaptive pool
        self.efficient = efficient
        self.pool = pool
        # other settings
        self.query_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim // latent_dim, kernel_size = 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim // latent_dim, kernel_size = 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim, kernel_size = 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, x, y):
        """
            inputs :
                x : input feature maps(B X C X W X H)
                y : input feature maps(B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        _, _, W, H = x.size()
        if self.efficient:
            if W > 64 or H > 64:
                x = F.adaptive_max_pool2d(x, 64)
                y = F.adaptive_max_pool2d(y, 64)

        if self.pool:
            x = F.max_pool2d(x, 3, 2, 1)
            y = F.max_pool2d(y, 3, 2, 1)

        m_batchsize, C, width, height = y.size()
        proj_query = self.query_conv(y).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x

        if self.efficient or self.pool:
            out = F.interpolate(out, size = (W, H), mode = 'bilinear')

        return out, attention

if __name__ == "__main__":

    # Quad-Bayer sensor data:
    # e.g., Rs: Red color with short exposure; 4*4 block
    # Bs Bl Gs Gl Bs Bl Gs Gl
    # Bs Bl Gs Gl Bs Bl Gs Gl
    # Gs Gl Rs Rl Gs Gl Rs Rl
    # Gs Gl Rs Rl Gs Gl Rs Rl
    # Bs Bl Gs Gl Bs Bl Gs Gl
    # Bs Bl Gs Gl Bs Bl Gs Gl
    # Gs Gl Rs Rl Gs Gl Rs Rl
    # Gs Gl Rs Rl Gs Gl Rs Rl
    # PixelUnShuffle:
    # (1): [::2, ::2]
    # Bs Gs Bs Gs
    # Gs Rs Gs Rs
    # Bs Gs Bs Gs
    # Gs Rs Gs Rs
    # (2): [::2, 1::2]
    # Bl Gl Bl Gl
    # Gl Rl Gl Rl
    # Bl Gl Bl Gl
    # Gl Rl Gl Rl
    # (3): [1::2, ::2]
    # Bs Gs Bs Gs
    # Gs Rs Gs Rs
    # Bs Gs Bs Gs
    # Gs Rs Gs Rs
    # (4): [1::2, 1::2]
    # Bl Gl Bl Gl
    # Gl Rl Gl Rl
    # Bl Gl Bl Gl
    # Gl Rl Gl Rl

    net = PixelUnShuffleAlign(2)
    net4 = PixelUnShuffleAlign(4)
    inet = PixelShuffleAlign(2)
    a = torch.randn(1, 1, 8, 8)
    b = net(a)
    b = net(b)
    print(a)
    print(b)
    c = inet(inet(b))
    print(a-c)

    '''
    dwt = DWT()
    idwt = IDWT()
    aa = dwt(dwt(a))
    aaa = idwt(idwt(aa))
    print(a-aaa)
    '''
