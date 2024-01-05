import torch
from torch import nn
from utils.config_files_utils import ConfigParams
import pickle
from .module_mlpmixer import MLPMixerPerLayerOut
from .module_transformer import TransformerPerLayerOut


class LAMM(nn.Module):
    """
    """
    default_values = {'scale_dim': 4, 'scale_dim_token': 0.5, 'dropout': 0, 'bottleneck_dim': None,
                      'depth': None, 'encoder_depth': None, 'decoder_depth': None,
                      'Dinput': 3, 'Dlms': 8, 'heads': 8
                      }
    def __init__(self, model_config):
        super().__init__()
        self.model_config = ConfigParams(model_config, self.default_values)
        self.Npatches = self.model_config['Npatches']
        self.dim = self.model_config['dim']
        self.depth = self.model_config['depth']
        self.encoder_depth = self.model_config['encoder_depth']
        self.decoder_depth = self.model_config['decoder_depth']
        if self.encoder_depth is None:
            self.encoder_depth = self.depth
        if self.decoder_depth is None:
            self.decoder_depth = self.depth
        self.heads = self.model_config['heads']
        self.dim_head = self.dim // self.heads
        self.scale_dim = self.model_config['scale_dim']
        self.scale_dim_token = self.model_config['scale_dim_token']
        self.dropout = self.model_config['dropout']
        self.face_part_ids_file = self.model_config['face_part_ids_file']
        self.bottleneck_dim = self.model_config['bottleneck_dim']
        self.Dinput = self.model_config['Dinput']
        self.backbone = self.model_config['backbone']
        self.manipulation = self.model_config['manipulation']
        if self.manipulation:
            self.regions = dict(sorted(self.model_config['regions'].items()))
            self.region_keys = list(self.regions.keys())
            self.region_sizes = [3 * len(i) for i in self.regions.values()]

        with open(self.face_part_ids_file, 'rb') as handle:
            face_parts_ids = pickle.load(handle, encoding='latin1')
            if type(face_parts_ids) is dict:
                self.face_part_ids = [torch.tensor(fpids) for fpids in face_parts_ids.values()]
            else:
                self.face_part_ids = [torch.tensor(fpids) for fpids in face_parts_ids]

        self.faceparts_size = [fp.shape[0] for fp in self.face_part_ids]
        self.num_face_vertices = torch.cat(self.face_part_ids).unique().shape[0]

        self.xyz_to_token = nn.ModuleList([nn.Linear(self.Dinput * patch_dim, self.dim) for patch_dim in self.faceparts_size])

        self.semantic_embedding = nn.Parameter(torch.randn(self.Npatches, self.dim))

        self.id_token = nn.Parameter(torch.randn(1, 1, self.dim))

        if self.bottleneck_dim is not None:
            print('bottleneck dim: ', self.bottleneck_dim)
            self.bottleneck_down = nn.Linear(self.dim, self.bottleneck_dim)
            self.bottleneck_up = nn.Linear(self.bottleneck_dim, self.dim)
        else:
            self.bottleneck_down = nn.Identity()
            self.bottleneck_up = nn.Identity()

        self.encoder = self.get_backbone(self.encoder_depth)

        self.semantic_tokens = nn.Parameter(torch.randn(1, self.Npatches, self.dim))

        self.decoder = self.get_backbone(self.decoder_depth)

        self.token_to_xyz = nn.ModuleList([nn.Linear(self.dim, 3 * patch_dim) for patch_dim in self.faceparts_size])

        if self.manipulation:
            self.delta_control_net = nn.ModuleList([
                nn.Sequential(nn.Linear(dl, 64, bias=False),
                              nn.GELU(),
                              nn.Linear(64, 256, bias=False),
                              nn.GELU(),
                              nn.Linear(256, 512, bias=False))
                for dl in self.region_sizes]
            )

    def get_backbone(self, depth):
        if self.backbone == 'transformer':
            return TransformerPerLayerOut(self.dim, depth, self.heads, self.dim_head, self.dim * self.scale_dim)
        elif self.backbone == 'mlpmixer':
            return MLPMixerPerLayerOut(self.dim, depth, self.Npatches + 1, self.scale_dim, expansion_factor_token=self.scale_dim_token)

    def get_parts(self, x):
        B, N, D = x.shape
        xparts = [x[:, ids].reshape(B, -1) for ids in self.face_part_ids]
        return xparts

    def forward(self, x):
        if self.manipulation:
            x, delta_lms = x
        else:
            delta_lms = None
        id_token, encoder_outputs = self.encode(x)
        decoder_outputs = self.decode(id_token, delta_lms)
        return torch.cat((encoder_outputs, decoder_outputs), dim=0)

    def encode(self, x):
        B, N, D = x.shape
        encoder_tokens = self.tokenize(x)
        encoder_tokens = encoder_tokens + self.semantic_embedding
        encoder_tokens = torch.cat((self.id_token.repeat(B, 1, 1), encoder_tokens), dim=1)
        encoder_tokens = self.encoder(encoder_tokens)
        id_token = encoder_tokens[-1][:, 0].unsqueeze(1)
        id_token = self.bottleneck_down(id_token)
        outputs = self.merge(encoder_tokens)
        return id_token, outputs

    def tokenize(self, x):
        xparts = self.get_parts(x)
        tokens = torch.stack([self.xyz_to_token[i](xparts[i]) for i in range(self.Npatches)], dim=1)
        return tokens

    def decode(self, id_token, delta_lms):
        B = id_token.shape[0]
        id_token = self.bottleneck_up(id_token)
        semantic_tokens = self.semantic_tokens.repeat(B, 1, 1)
        if self.manipulation:
            delta_token = [fc(delta) for fc, delta in zip(self.delta_control_net, delta_lms)]
            for i, idx in enumerate(self.region_keys):
                semantic_tokens[:, idx] += delta_token[i]
        decoder_tokens = torch.cat((id_token, semantic_tokens), dim=1)
        decoder_tokens = self.decoder(decoder_tokens)
        decoder_outputs = self.merge(decoder_tokens)
        return decoder_outputs

    def merge(self, tokens):
        B = tokens[0].shape[0]
        output_tokens = [d[:, 1:] for d in tokens]
        xyz = self.inverse_tokenize(output_tokens)
        outputs = torch.zeros((len(tokens), B, self.num_face_vertices, 3)).to(tokens[0].device)
        for i, layer_output in enumerate(xyz):
            for j, face_part in enumerate(layer_output):
                outputs[i, :, self.face_part_ids[j]] += face_part
        return outputs

    def inverse_tokenize(self, y):
        B = y[0].shape[0]
        xyz = [[self.token_to_xyz[i](out_tokens[:, i]).reshape(B, self.faceparts_size[i], 3) for i in
                range(self.Npatches)]
               for out_tokens in y]
        return xyz
