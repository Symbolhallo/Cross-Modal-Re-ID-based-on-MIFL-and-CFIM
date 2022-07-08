import math
from itertools import repeat
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from resnet import resnet50, resnet18
from vit_model import vit_base_patch16_224_in21k as create_model

import collections.abc

container_abcs = collections.abc


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class modal_Classifier(nn.Module):
    def __init__(self, embed_dim, modal_class):
        super(modal_Classifier, self).__init__()
        hidden_size = 1024
        self.first_layer = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=hidden_size, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.layers = nn.ModuleList()
        for layer_index in range(7):
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size // 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(hidden_size // 2),
                nn.ReLU(inplace=True)
            )
            hidden_size = hidden_size // 2  # 512-32-8
            self.layers.append(conv_block)
        self.Liner = nn.Linear(hidden_size, modal_class)

    def forward(self, latent):
        latent = latent.unsqueeze(2)
        hidden = self.first_layer(latent)
        for i in range(7):
            hidden = self.layers[i](hidden)
        style_cls_feature = hidden.squeeze(2)
        modal_cls = self.Liner(style_cls_feature)
        if self.training:
            return modal_cls  # [batch,3]


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio // reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                      padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


class PatchEmbed_overlap(nn.Module):
    """ Image to Patch Embedding with overlapping patches
    """

    def __init__(self, img_size=224, patch_size=16, stride_size=20, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        print('using stride: {}, and patch number is num_y{} * num_x{}'.format(stride_size, self.num_y, self.num_x))
        num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=stride_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape

        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)

        x = x.flatten(2).transpose(1, 2)  # [64, 8, 768]
        return x


class visible_module(nn.Module):
    def __init__(self, arch='resnet50', share_net=1):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.share_net = share_net

        if self.share_net == 0:
            pass
        else:
            self.visible = nn.ModuleList()
            self.visible.conv1 = model_v.conv1
            self.visible.bn1 = model_v.bn1
            self.visible.relu = model_v.relu
            self.visible.maxpool = model_v.maxpool
            if self.share_net > 1:
                for i in range(1, self.share_net):
                    setattr(self.visible, 'layer' + str(i), getattr(model_v, 'layer' + str(i)))

    def forward(self, x):
        if self.share_net == 0:
            return x
        else:
            x = self.visible.conv1(x)
            x = self.visible.bn1(x)
            x = self.visible.relu(x)
            x = self.visible.maxpool(x)

            if self.share_net > 1:
                for i in range(1, self.share_net):
                    x = getattr(self.visible, 'layer' + str(i))(x)
            return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50', share_net=1):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.share_net = share_net

        if self.share_net == 0:
            pass
        else:
            self.thermal = nn.ModuleList()
            self.thermal.conv1 = model_t.conv1
            self.thermal.bn1 = model_t.bn1
            self.thermal.relu = model_t.relu
            self.thermal.maxpool = model_t.maxpool
            if self.share_net > 1:
                for i in range(1, self.share_net):
                    setattr(self.thermal, 'layer' + str(i), getattr(model_t, 'layer' + str(i)))

    def forward(self, x):
        if self.share_net == 0:
            return x
        else:
            x = self.thermal.conv1(x)
            x = self.thermal.bn1(x)
            x = self.thermal.relu(x)
            x = self.thermal.maxpool(x)

            if self.share_net > 1:
                for i in range(1, self.share_net):
                    x = getattr(self.thermal, 'layer' + str(i))(x)
            return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50', share_net=1):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.share_net = share_net
        if self.share_net == 0:
            self.base = model_base
        else:
            self.base = nn.ModuleList()

            if self.share_net > 4:
                pass
            else:
                for i in range(self.share_net, 5):
                    setattr(self.base, 'layer' + str(i), getattr(model_base, 'layer' + str(i)))

    def forward(self, x):
        if self.share_net == 0:
            x = self.base.conv1(x)
            x = self.base.bn1(x)
            x = self.base.relu(x)
            x = self.base.maxpool(x)

            x = self.base.layer1(x)
            x = self.base.layer2(x)
            x = self.base.layer3(x)
            x = self.base.layer4(x)
            return x
        elif self.share_net > 4:
            return x
        else:
            for i in range(self.share_net, 5):
                x = getattr(self.base, 'layer' + str(i))(x)
            return x


class embed_net(nn.Module):
    def __init__(self, class_num, no_local='off', gm_pool='on', arch='resnet50', share_net=1, cd='off', m3='off',
                 local_feat_dim=256, num_strips=6):
        super(embed_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch, share_net=share_net)
        self.visible_module = visible_module(arch=arch, share_net=share_net)
        self.base_resnet = base_resnet(arch=arch, share_net=share_net)

        # 模态特征
        self.thermal_modal_module = PatchEmbed_overlap(img_size=(288, 144))
        self.visible_modal_module = PatchEmbed_overlap(img_size=(288, 144))
        self.transformer = create_model(num_classes=768, has_logits=False)

        self.non_local = no_local
        self.cd = cd
        self.m3 = m3
        if self.non_local == 'on':
            pass
        self.pool_dim = 2048
        self.l2norm = Normalize(2)
        self.gm_pool = gm_pool

        if self.cd == 'on':
            self.num_channel = 4
            self.num_stripes = num_strips
            self.local_feat_dim = local_feat_dim
            local_conv_out_channels = local_feat_dim

            # 语义分类器
            self.classifier_4 = nn.Linear(self.num_stripes * self.local_feat_dim, self.num_channel, bias=False)
            self.classifier_4.apply(weights_init_classifier)

            # channel 1
            self.local_conv_list1 = nn.ModuleList()
            for _ in range(self.num_stripes):
                conv = nn.Conv2d(self.pool_dim//self.num_channel, local_conv_out_channels, 1)
                conv.apply(weights_init_kaiming)
                self.local_conv_list1.append(nn.Sequential(
                    conv,
                    nn.BatchNorm2d(local_conv_out_channels),
                    nn.ReLU(inplace=True)
                ))
            self.fc_list1 = nn.ModuleList()
            for _ in range(self.num_stripes):
                fc = nn.Linear(local_conv_out_channels, class_num)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_list1.append(fc)

            # channel 2
            self.local_conv_list2 = nn.ModuleList()
            for _ in range(self.num_stripes):
                conv = nn.Conv2d(self.pool_dim//self.num_channel, local_conv_out_channels, 1)
                conv.apply(weights_init_kaiming)
                self.local_conv_list2.append(nn.Sequential(
                    conv,
                    nn.BatchNorm2d(local_conv_out_channels),
                    nn.ReLU(inplace=True)
                ))
            self.fc_list2 = nn.ModuleList()
            for _ in range(self.num_stripes):
                fc = nn.Linear(local_conv_out_channels, class_num)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_list2.append(fc)

            # channel 3
            self.local_conv_list3 = nn.ModuleList()
            for _ in range(self.num_stripes):
                conv = nn.Conv2d(self.pool_dim//self.num_channel, local_conv_out_channels, 1)
                conv.apply(weights_init_kaiming)
                self.local_conv_list3.append(nn.Sequential(
                    conv,
                    nn.BatchNorm2d(local_conv_out_channels),
                    nn.ReLU(inplace=True)
                ))
            self.fc_list3 = nn.ModuleList()
            for _ in range(self.num_stripes):
                fc = nn.Linear(local_conv_out_channels, class_num)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_list3.append(fc)

            # channel 4
            self.local_conv_list4 = nn.ModuleList()
            for _ in range(self.num_stripes):
                conv = nn.Conv2d(self.pool_dim//self.num_channel, local_conv_out_channels, 1)
                conv.apply(weights_init_kaiming)
                self.local_conv_list4.append(nn.Sequential(
                    conv,
                    nn.BatchNorm2d(local_conv_out_channels),
                    nn.ReLU(inplace=True)
                ))
            self.fc_list4 = nn.ModuleList()
            for _ in range(self.num_stripes):
                fc = nn.Linear(local_conv_out_channels, class_num)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_list4.append(fc)

            # 通道分组的分类损失
            self.fc_channel_list = nn.ModuleList()
            out_channels = 1536
            for _ in range(self.num_stripes):
                fc = nn.Linear(out_channels, class_num)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_channel_list.append(fc)

        if self.m3 == 'on':
            self.modal_dim = 2048
            self.modal_num = 2
            self.modal_bottleneck = nn.BatchNorm1d(self.modal_dim)
            self.modal_bottleneck.bias.requires_grad_(False)  # no shift
            self.modal_bottleneck.apply(weights_init_kaiming)

            self.modal_classifier = nn.Linear(self.modal_dim, self.modal_num, bias=False)  # 模态分类器
            self.modal_classifier.apply(weights_init_classifier)

            self.bottleneck = nn.BatchNorm1d(self.modal_dim)
            self.bottleneck.apply(weights_init_kaiming)
            self.bottleneck.bias.requires_grad_(False)  # no shift




        else:
            self.bottleneck = nn.BatchNorm1d(self.pool_dim)
            self.bottleneck.bias.requires_grad_(False)  # no shift

            self.classifier = nn.Linear(self.pool_dim, class_num, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x1_, x2_, modal=0):
        if modal == 0:
            x1 = self.visible_module(x1_)
            x2 = self.thermal_module(x2_)
            x = torch.cat((x1, x2), 0)

            x1_m = self.visible_modal_module(x1_)
            x2_m = self.thermal_modal_module(x2_)
            modal_x = torch.cat((x1_m, x2_m), 0)  # [B 98 768]
        elif modal == 1:
            x = self.visible_module(x1_)
            x1_m = self.visible_modal_module(x1_)
            modal_x = x1_m  # [B 98 768]
        elif modal == 2:
            x = self.thermal_module(x2_)
            x2_m = self.thermal_modal_module(x2_)
            modal_x = x2_m  # [B 98 768]

        # shared block
        if self.non_local == 'on':
            pass
        else:
            x = self.base_resnet(x)

        b, c, h, w = x.shape
        x_ = x.view(b, c, -1)
        p = 3.0
        x_pool = (torch.mean(x_ ** p, dim=-1) + 1e-12) ** (1 / p)
        # 模态不变性学习模块
        feat_x = self.bottleneck(x_pool)
        modal_token = self.transformer(modal_x)
        modal_feat = self.modal_bottleneck(modal_token)
        modal_score = self.modal_classifier(modal_feat)

        if self.cd == 'on':
            # 通道分组
            channel1 = x[:, 0:512:, :]
            channel2 = x[:, 512:1024, :]
            channel3 = x[:, 1024:1536:, :]
            channel4 = x[:, 1536:, :]

            feat = channel1
            assert feat.size(2) % self.num_stripes == 0
            stripe_h = int(feat.size(2) / self.num_stripes)
            local_feat_list1 = []
            logits_list1 = []
            # 水平分块
            for i in range(self.num_stripes):
                if self.gm_pool == 'on':
                    local_feat = feat[:, :, i * stripe_h: (i + 1) * stripe_h, :]
                    b, c, h, w = local_feat.shape
                    local_feat = local_feat.view(b, c, -1)
                    p = 3.0  # regDB: 10.0    SYSU: 3.0
                    local_feat = (torch.mean(local_feat ** p, dim=-1) + 1e-12) ** (1 / p)
                else:
                    local_feat = F.max_pool2d(feat[:, :, i * stripe_h: (i + 1) * stripe_h, :],
                                              (stripe_h, feat.size(-1)))
                # shape [N, c, 1, 1]
                local_feat = self.local_conv_list1[i](local_feat.view(feat.size(0), feat.size(1), 1, 1))
                # shape [N, c]
                local_feat = local_feat.view(local_feat.size(0), -1)
                local_feat_list1.append(local_feat)

                if hasattr(self, 'fc_list1'):
                    logits_list1.append(self.fc_list1[i](local_feat))

            feat_1 = [lf for lf in local_feat_list1]

            feat = channel2
            assert feat.size(2) % self.num_stripes == 0
            stripe_h = int(feat.size(2) / self.num_stripes)
            local_feat_list2 = []
            logits_list2 = []
            for i in range(self.num_stripes):
                if self.gm_pool == 'on':
                    local_feat = feat[:, :, i * stripe_h: (i + 1) * stripe_h, :]
                    b, c, h, w = local_feat.shape
                    local_feat = local_feat.view(b, c, -1)
                    p = 3.0  # regDB: 10.0    SYSU: 3.0
                    local_feat = (torch.mean(local_feat ** p, dim=-1) + 1e-12) ** (1 / p)
                else:
                    local_feat = F.max_pool2d(feat[:, :, i * stripe_h: (i + 1) * stripe_h, :],
                                              (stripe_h, feat.size(-1)))
                # shape [N, c, 1, 1]
                local_feat = self.local_conv_list2[i](local_feat.view(feat.size(0), feat.size(1), 1, 1))

                # shape [N, c]
                local_feat = local_feat.view(local_feat.size(0), -1)
                local_feat_list2.append(local_feat)

                if hasattr(self, 'fc_list2'):
                    logits_list2.append(self.fc_list2[i](local_feat))
            feat_2 = [lf for lf in local_feat_list2]

            feat = channel3
            assert feat.size(2) % self.num_stripes == 0
            stripe_h = int(feat.size(2) / self.num_stripes)
            local_feat_list3 = []
            logits_list3 = []
            for i in range(self.num_stripes):
                if self.gm_pool == 'on':
                    local_feat = feat[:, :, i * stripe_h: (i + 1) * stripe_h, :]
                    b, c, h, w = local_feat.shape
                    local_feat = local_feat.view(b, c, -1)
                    p = 3.0  # regDB: 10.0    SYSU: 3.0
                    local_feat = (torch.mean(local_feat ** p, dim=-1) + 1e-12) ** (1 / p)
                else:
                    local_feat = F.max_pool2d(feat[:, :, i * stripe_h: (i + 1) * stripe_h, :],
                                              (stripe_h, feat.size(-1)))
                # shape [N, c, 1, 1]
                local_feat = self.local_conv_list3[i](local_feat.view(feat.size(0), feat.size(1), 1, 1))
                # shape [N, c]
                local_feat = local_feat.view(local_feat.size(0), -1)
                local_feat_list3.append(local_feat)
                if hasattr(self, 'fc_list3'):
                    logits_list3.append(self.fc_list3[i](local_feat))

            feat_3 = [lf for lf in local_feat_list3]

            feat = channel4
            assert feat.size(2) % self.num_stripes == 0
            stripe_h = int(feat.size(2) / self.num_stripes)
            local_feat_list4 = []
            logits_list4 = []
            for i in range(self.num_stripes):
                if self.gm_pool == 'on':
                    local_feat = feat[:, :, i * stripe_h: (i + 1) * stripe_h, :]
                    b, c, h, w = local_feat.shape
                    local_feat = local_feat.view(b, c, -1)
                    p = 3.0  # regDB: 10.0    SYSU: 3.0
                    local_feat = (torch.mean(local_feat ** p, dim=-1) + 1e-12) ** (1 / p)
                else:
                    local_feat = F.max_pool2d(feat[:, :, i * stripe_h: (i + 1) * stripe_h, :],
                                              (stripe_h, feat.size(-1)))
                # shape [N, c, 1, 1]
                local_feat = self.local_conv_list4[i](local_feat.view(feat.size(0), feat.size(1), 1, 1))
                # shape [N, c]
                local_feat = local_feat.view(local_feat.size(0), -1)
                local_feat_list4.append(local_feat)

                if hasattr(self, 'fc_list4'):
                    logits_list4.append(self.fc_list4[i](local_feat))
            feat_4 = [lf for lf in local_feat_list4]
            feat_4 = torch.cat(feat_4, dim=1)

            feat_list = [feat_1, feat_2, feat_3, feat_4]
            local_feat_list_all = [local_feat_list1, local_feat_list2, local_feat_list3, local_feat_list4]
            logits_list_all = [logits_list1, logits_list2, logits_list3, logits_list4]
            # 语义分类
            temp = []
            for i in range(len(feat_list)):
                temp.append(feat_list[i].view(b, 1, self.num_stripes * self.local_feat_dim))
            temp = torch.cat(temp, dim=1)  # [b num_channel local_feat_dim]
            masks = temp.view(b, self.num_channel, self.num_stripes * self.local_feat_dim)
            loss_reg0 = torch.bmm(masks, masks.permute(0, 2, 1))
            loss_reg0 = torch.triu(loss_reg0, diagonal=1).sum() / (b * self.num_channel * (self.num_channel - 1) / 2)

            # 语义分类
            feat_cd4 = torch.cat((feat_1, feat_2, feat_3, feat_4), 0)  # [B*number 512]
            score_4 = self.classifier_4(feat_cd4)

            # 不要划块的去相关
            loss_reg = loss_reg0
            # 水平划块拼接起来的身份损失
            channel_scores1 = self.fc_channel_list[0](feat_1)
            channel_scores2 = self.fc_channel_list[1](feat_2)
            channel_scores3 = self.fc_channel_list[2](feat_3)
            channel_scores4 = self.fc_channel_list[3](feat_4)
            channel_scores_list = [channel_scores1, channel_scores2, channel_scores3, channel_scores4]

            feat_all = torch.cat((feat_1, feat_2, feat_3, feat_4), dim=1)
            if self.training:
                return local_feat_list_all, logits_list_all, channel_scores_list, feat_list, feat_all, loss_reg, feat_x, modal_score, modal_feat, score_4
            else:
                return self.l2norm(feat_all)
        else:
            if self.gm_pool == 'on':
                b, c, h, w = x.shape
                x = x.view(b, c, -1)
                p = 3.0
                x_pool = (torch.mean(x ** p, dim=-1) + 1e-12) ** (1 / p)
            else:
                x_pool = self.avgpool(x)
                x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))

            feat = self.bottleneck(x_pool)

            if self.training:
                return x_pool, self.classifier(feat)  # , scores
            else:
                return self.l2norm(x_pool), self.l2norm(feat)
