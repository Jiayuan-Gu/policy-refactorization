import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, RelaxedBernoulli

from space.models.space_v0 import SPACE_v0

from space.utils.model_utils import make_conv1x1_bn
from space.utils.anchor_utils import generate_anchors, grid_anchors
from space.utils.box_utils import decode_boxes, boxes_xyxy2xywh, image_to_glimpse, glimpse_to_image


class SPACE_v1(SPACE_v0):
    """SPACE: Unsupervised Object-Oriented Scene Representation via Spatial Attention and Decomposition.
    (For 128x128x3 images with complex backgrounds)
    """

    def __init__(self,
                 image_shape=(128, 128, 3),
                 cell_scales=(14,),
                 cell_aspect_ratios=(1.0,),
                 glimpse_shape=(16, 16),
                 z_what_size=50,
                 eps=1e-4,
                 clip_to_image=True,
                 clip_delta=2.0,
                 **kwargs,
                 ):
        super(SPACE_v0, self).__init__()

        self.image_shape = image_shape  # (h0, w0, c0)
        self.glimpse_shape = glimpse_shape  # (h2, w2)

        self.fg_encoder = nn.Sequential(
            nn.Conv2d(image_shape[-1], 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (64, 64)
            nn.Conv2d(16, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (32, 32)
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (16, 16)
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.rpn = make_conv1x1_bn(128, (128, 128),
                                   act_builder=lambda: nn.ReLU(inplace=True))

        # Single-level anchors
        # (num_scales * num_ratios, 4)
        self.register_buffer('cell_anchors', generate_anchors(cell_scales, cell_aspect_ratios))
        self.num_anchors_per_cell = len(self.cell_anchors)
        self._eps = eps  # for numerical stability of Bernoulli p
        self.clip_to_image = clip_to_image
        self.clip_delta = math.log(clip_delta)

        # Latent codes
        self.latent_where = nn.Conv2d(128, self.num_anchors_per_cell * 8, 1)
        self.latent_pres = nn.Conv2d(128, self.num_anchors_per_cell, 1)
        self.latent_depth = nn.Conv2d(128, self.num_anchors_per_cell * 2, 1)

        # glimpse encoder
        self.glimpse_encoder = nn.Sequential(
            nn.Conv2d(image_shape[-1], 16, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (8, 8)
            nn.Conv2d(16, 32, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (4, 4)
            nn.Conv2d(32, 64, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (2, 2)
            nn.Conv2d(64, 128, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (1, 1)
        )

        self.z_what_size = z_what_size
        self.latent_what = nn.Linear(128, z_what_size * 2)

        # glimpse decoder
        self.glimpse_decoder = nn.Sequential(
            nn.ConvTranspose2d(z_what_size, 128, 2, stride=2, bias=True), nn.ReLU(inplace=True),  # (2, 2)
            nn.Conv2d(128, 64, 1, bias=True), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 2, stride=2, bias=True), nn.ReLU(inplace=True),  # (4, 4)
            nn.Conv2d(64, 32, 1, bias=True), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 2, stride=2, bias=True), nn.ReLU(inplace=True),  # (8, 8)
            nn.Conv2d(32, 16, 1, bias=True), nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),  # (16, 16)
            nn.Conv2d(16, image_shape[-1] + 1, 1),
        )

        # Background
        self.bg_encoder = nn.Sequential(
            nn.Conv2d(image_shape[-1], 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (64, 64)
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (32, 32)
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (16, 16)
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (8, 8)
        )
        self.bg_decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 2, stride=2, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),  # (16, 16)
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 2, stride=2, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),  # (32, 32)
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 2, stride=2, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),  # (64, 64)
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, image_shape[-1], 1),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self,
                data_batch: dict,
                temperature=1.0,
                depth_scale=10.0,
                fast=False,
                **kwargs):
        pd_dict = dict()
        image = data_batch['image']
        b, c0, h0, w0 = image.size()
        A = self.num_anchors_per_cell

        # ---------------------------------------------------------------------------- #
        # CNN encodes feature maps
        # ---------------------------------------------------------------------------- #
        fg_feature = self.fg_encoder(image)
        fg_feature = self.rpn(fg_feature)
        _, c1, h1, w1 = fg_feature.size()

        # ---------------------------------------------------------------------------- #
        # Relaxed Bernoulli z_pres
        # ---------------------------------------------------------------------------- #
        latent_pres = self.latent_pres(fg_feature)  # (b, A, h1, w1)
        z_pres_p = torch.sigmoid(latent_pres)
        z_pres_p = z_pres_p.reshape(b, -1)  # (b, A * h1 * w1)
        # In order to avoid gradient explosion at 0 and 1, clip
        z_pres_p = z_pres_p.clamp(min=self._eps, max=1.0 - self._eps)
        z_pres_post = RelaxedBernoulli(z_pres_p.new_tensor(temperature), probs=z_pres_p)
        if self.training:
            z_pres = z_pres_post.rsample()
        else:
            z_pres = z_pres_p

        pd_dict['z_pres'] = z_pres  # (b, A * h1 * w1)
        pd_dict['z_pres_p'] = z_pres_p
        pd_dict['z_pres_post'] = z_pres_post

        # ---------------------------------------------------------------------------- #
        # Gaussian z_depth
        # ---------------------------------------------------------------------------- #
        latent_depth = self.latent_depth(fg_feature)  # (b, A * 2, h1, w1)
        z_depth_loc = latent_depth.narrow(1, 0, A)
        z_depth_scale = F.softplus(latent_depth.narrow(1, A, A))
        z_depth_post = Normal(z_depth_loc.reshape(b, -1), z_depth_scale.reshape(b, -1))
        if self.training:
            z_depth = z_depth_post.rsample()  # (b, A * h1 * w1)
        else:
            z_depth = z_depth_loc

        pd_dict['z_depth_post'] = z_depth_post  # (b, A * h1 * w1)
        pd_dict['z_depth'] = z_depth  # (b, A * h1 * w1)

        # ---------------------------------------------------------------------------- #
        # Gaussian z_where
        # (offset_x, offset_y, scale_x, scale_y)
        # ---------------------------------------------------------------------------- #
        latent_where = self.latent_where(fg_feature)  # (b, A * 8, h1, w1)
        latent_where = latent_where.reshape(b, A, 8, h1, w1)  # (b, A, 8, h1, w1)
        latent_where = latent_where.permute(0, 1, 3, 4, 2).contiguous()  # (b, A, h1, w1, 8)
        latent_where = latent_where.reshape(b, A * h1 * w1, 8)
        z_where_loc = latent_where.narrow(-1, 0, 4)
        z_where_scale = F.softplus(latent_where.narrow(-1, 4, 4))
        z_where_post = Normal(z_where_loc, z_where_scale)
        if self.training:
            z_where = z_where_post.rsample()  # (b, A * h1 * w1, 4)
        else:
            z_where = z_where_loc

        pd_dict['z_where_post'] = z_where_post  # (b, A * h1 * w1, 4)

        # ---------------------------------------------------------------------------- #
        # Decode z_where to boxes
        # ---------------------------------------------------------------------------- #
        # (A * h1 * w1, 4)
        anchors = grid_anchors(self.cell_anchors, [h1, w1], [int(h0 / h1), int(w0 / w1)])
        # (b, A * h1 * w1, 4)
        boxes = decode_boxes(anchors, z_where, clip_delta=self.clip_delta,
                             image_shape=(h0, w0) if self.clip_to_image else None)

        pd_dict['boxes'] = boxes
        pd_dict['grid_size'] = (h1, w1)

        # ---------------------------------------------------------------------------- #
        # Normalize boxes
        # Note that spatial transform assumes [-1, 1] for coordinates
        # ---------------------------------------------------------------------------- #
        x_min, y_min, x_max, y_max = torch.split(boxes, 1, dim=-1)
        # (b, A * h1 * w1, 4)
        normalized_boxes = torch.cat([x_min / w0, y_min / h0, x_max / w0, y_max / h0], dim=-1)
        # convert xyxy to xywh
        normalized_boxes = boxes_xyxy2xywh(normalized_boxes)

        if fast:
            pd_dict['normalized_boxes'] = normalized_boxes
            return pd_dict

        # ---------------------------------------------------------------------------- #
        # Gaussian z_what
        # ---------------------------------------------------------------------------- #
        # Crop glimpses, (b * A * h1 * w1, c, h2, w2)
        glimpses = image_to_glimpse(image, normalized_boxes, glimpse_shape=self.glimpse_shape)
        # Gaussian z_what
        glimpses_feature = self.glimpse_encoder(glimpses)
        latent_what = self.latent_what(glimpses_feature.flatten(1))
        z_what_loc = latent_what[:, 0:self.z_what_size]
        z_what_scale = F.softplus(latent_what[:, self.z_what_size:])
        z_what_post = Normal(z_what_loc, z_what_scale)
        if self.training:
            z_what = z_what_post.rsample()
        else:
            z_what = z_what_loc

        pd_dict['z_what_post'] = z_what_post  # (b * A * h1 * w1, z_what_size)
        pd_dict['glimpses'] = glimpses.reshape(b, -1, c0, self.glimpse_shape[0], self.glimpse_shape[1])
        pd_dict['glimpses_feature'] = glimpses_feature.reshape(b, A * h1 * w1, -1)

        # ---------------------------------------------------------------------------- #
        # Decode z_what
        # ---------------------------------------------------------------------------- #
        # (b * A * h1 * w1, (c+1), h2, w2)
        glimpses_recon = self.glimpse_decoder(z_what.unsqueeze(-1).unsqueeze(-1))
        glimpses_recon = torch.sigmoid(glimpses_recon)

        glimpses_recon_reshape = glimpses_recon.reshape(b, -1, c0 + 1, self.glimpse_shape[0], self.glimpse_shape[1])
        pd_dict['glimpse_rgb'] = glimpses_recon_reshape[:, :, :-1]
        pd_dict['glimpse_alpha'] = glimpses_recon_reshape[:, :, -1:]

        # ---------------------------------------------------------------------------- #
        # Foreground
        # ---------------------------------------------------------------------------- #
        # (b * A * h1 * w1, c0 + 1, h0, w0)
        fg_rgba = glimpse_to_image(glimpses_recon, normalized_boxes, image_shape=(h0, w0))
        # (b, A * h1 * w1, c + 1, h, w)
        fg_rgba = fg_rgba.reshape(b, -1, c0 + 1, h0, w0)
        # (b, A * h1 * w1, 1, 1, 1)
        z_pres_reshape = z_pres.reshape(b, -1, 1, 1, 1)

        # Note that first c0 channels are rgb, and the last one is alpha.
        fg_rgb = fg_rgba[:, :, :-1]  # (b, A * h1 * w1, c0, h0, w0)
        fg_alpha = fg_rgba[:, :, -1:]  # (b, A * h1 * w1, 1, h0, w0)
        # Use foreground objects only
        fg_alpha_valid = fg_alpha * z_pres_reshape
        z_depth_reshape = z_depth.reshape(b, -1, 1, 1, 1)
        fg_weight = torch.softmax(fg_alpha_valid * depth_scale * torch.sigmoid(z_depth_reshape), dim=1)
        fg_mask_all = fg_alpha_valid * fg_weight
        fg_recon = (fg_rgb * fg_mask_all).sum(1)
        fg_mask = fg_mask_all.sum(1)

        pd_dict['fg_recon'] = fg_recon
        pd_dict['fg_mask'] = fg_mask

        # ---------------------------------------------------------------------------- #
        # Background
        # ---------------------------------------------------------------------------- #
        bg_feature = self.bg_encoder(image)
        bg_recon = self.bg_decoder(bg_feature)
        bg_recon = torch.sigmoid(bg_recon)
        bg_recon = bg_recon.reshape(b, c0, h0, w0)
        pd_dict['bg_recon'] = bg_recon

        return pd_dict


def test():
    from common.utils.misc import print_dict

    model = SPACE_v1()
    print(model)
    data_batch = {
        'image': torch.rand(4, 3, 64, 64),
        'z_pres_p_prior': 0.1,
        'z_where_loc_prior': torch.tensor([0.0, 0.0, 0.0, 0.0]),
        'z_where_scale_prior': torch.tensor([0.2, 0.2, 0.2, 0.2]),
        'z_what_loc_prior': 0.0,
        'z_what_scale_prior': 1.0,
        'z_depth_loc_prior': 0.0,
        'z_depth_scale_prior': 1.0,
        'fg_recon_scale_prior': 0.15,
        'bg_recon_scale_prior': 0.15,
    }
    pd_dict = model(data_batch)
    print_dict(pd_dict)
    loss_dict = model.compute_losses(pd_dict, data_batch)
    print_dict(loss_dict)
