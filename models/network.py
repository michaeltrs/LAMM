import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..')))
import torch
from torch import nn
from utils.config_utils import ConfigParams
import pickle
from models.module_mlpmixer import MLPMixerPerLayerOut
from models.module_transformer import TransformerPerLayerOut


class LAMM(nn.Module):
    """
    Implements the LAMM (Locally Adaptive Morphable Model) neural network architecture.
    arxiv: https://arxiv.org/pdf/2401.02937.pdf

    This architecture is designed for manipulating 3D mesh geometry through a learning-based approach. It supports
    both transformer and MLPMixer backbones and allows for manipulation of specific mesh regions using control vertices.

    Parameters:
        model_config (dict): Configuration dictionary containing model hyperparameters and settings.

    Attributes:
        Npatches (int): Number of patches (mesh regions) the model operates on.
        dim (int): Dimensionality of the token embeddings.
        depth (int): Overall depth of the encoder and decoder, if not specified separately.
        encoder_depth (int): Depth of the encoder. Defaults to `depth` if not specified.
        decoder_depth (int): Depth of the decoder. Defaults to `depth` if not specified.
        heads (int): Number of attention heads in transformer layers.
        dim_head (int): Dimensionality of each attention head.
        scale_dim (int): Scaling factor for the dimensionality in transformer layers.
        scale_dim_token (float): Token dimension scaling factor in MLPMixer.
        dropout (float): Dropout rate applied in the architecture.
        bottleneck_dim (int): Dimensionality of the bottleneck layer. If None, an identity mapping is used.
        Dinput (int): Dimensionality of the input data.
        backbone (str): Specifies the backbone architecture ('transformer' or 'mlpmixer').
        manipulation (bool): Indicates if model is in manipulation mode, requiring control vertices.
        control_vertices (dict): Dictionary mapping regions to control vertices for manipulation.

    The model supports loading region IDs from a file, tokenizing 3D XYZ coordinates into a learned embedding space,
    encoding and decoding mechanisms, and applying specific manipulations based on control vertices.
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
        self.region_ids_file = self.model_config['region_ids_file']
        self.bottleneck_dim = self.model_config['bottleneck_dim']
        self.Dinput = self.model_config['Dinput']
        self.backbone = self.model_config['backbone']
        self.manipulation = self.model_config['manipulation']
        if self.manipulation:
            self.control_vertices = dict(sorted(self.model_config['control_vertices'].items()))
            self.control_region_keys = list(self.control_vertices.keys())
            self.control_region_sizes = [3 * len(i) for i in self.control_vertices.values()]

        with open(self.region_ids_file, 'rb') as handle:
            region_ids = pickle.load(handle, encoding='latin1')
            if type(region_ids) is dict:
                self.region_ids = [torch.tensor(fpids) for fpids in region_ids.values()]
            else:
                self.region_ids = [torch.tensor(fpids) for fpids in region_ids]

        self.faceparts_size = [fp.shape[0] for fp in self.region_ids]
        self.num_face_vertices = torch.cat(self.region_ids).unique().shape[0]

        # Initialize input region tokenizers (XYZ-to-token)
        self.xyz_to_token = nn.ModuleList(
            [nn.Linear(self.Dinput * patch_dim, self.dim) for patch_dim in self.faceparts_size])

        # Create a learned semantic embedding parameter akin to learned positional embeddings
        self.semantic_embedding = nn.Parameter(torch.randn(self.Npatches, self.dim))

        # Initialize state of latent code at input layer akin to cls token
        self.id_token = nn.Parameter(torch.randn(1, 1, self.dim))

        if self.bottleneck_dim is not None:
            print('bottleneck dim: ', self.bottleneck_dim)
            self.bottleneck_down = nn.Linear(self.dim, self.bottleneck_dim)
            self.bottleneck_up = nn.Linear(self.bottleneck_dim, self.dim)
        else:
            self.bottleneck_down = nn.Identity()
            self.bottleneck_up = nn.Identity()

        self.encoder = self.get_backbone(self.encoder_depth)

        # Initialize state of region tokens at decoder input.
        # These will be decoded to target geometries at decoder output
        self.learned_decoder_tokens = nn.Parameter(torch.randn(1, self.Npatches, self.dim))

        self.decoder = self.get_backbone(self.decoder_depth)

        # Initialize region inverse tokenizers (token-to-XYZ)
        self.token_to_xyz = nn.ModuleList([nn.Linear(self.dim, 3 * patch_dim) for patch_dim in self.faceparts_size])

        # Intialize displacement control networks updating the state of decoder region tokens at decoder input
        if self.manipulation:
            self.delta_control_net = nn.ModuleList([
                nn.Sequential(nn.Linear(dl, 64, bias=False),
                              nn.GELU(),
                              nn.Linear(64, 256, bias=False),
                              nn.GELU(),
                              nn.Linear(256, 512, bias=False))
                for dl in self.control_region_sizes]
            )

    def get_backbone(self, depth):
        """
        Initializes the backbone architecture based on the configuration.

        Parameters:
            depth (int): The depth of the model.

        Returns:
            An instance of TransformerPerLayerOut or MLPMixerPerLayerOut depending on the specified backbone.
        """
        if self.backbone == 'transformer':
            return TransformerPerLayerOut(self.dim, depth, self.heads, self.dim_head, self.dim * self.scale_dim)
        elif self.backbone == 'mlpmixer':
            return MLPMixerPerLayerOut(self.dim, depth, self.Npatches + 1, self.scale_dim, expansion_factor_token=self.scale_dim_token)

    def get_regions(self, x):
        """
        Splits input tensor into parts corresponding to the predefined mesh regions.

        Parameters:
            x (Tensor): The input tensor with shape (B, N, D) where B is batch size, N is number of vertices, and D is
            dimensionality.

        Returns:
            List of tensors, each corresponding to a different facial region.
        """
        B, N, D = x.shape
        regions = [x[:, ids].reshape(B, -1) for ids in self.region_ids]
        return regions

    def forward(self, x):
        """
        Defines the forward pass of the LAMM model.

        Parameters:
            x (Tensor or tuple): Input tensor with shape (B, N, D) or a tuple of (input tensor, delta landmarks) for
            manipulation mode.

        Returns:
            Tensor: The output tensor after processing through the encoder, optional manipulation, and decoder.
        """
        if self.manipulation:
            x, delta_lms = x
        else:
            delta_lms = None
        id_token, encoder_outputs = self.encode(x)
        decoder_outputs = self.decode(id_token, delta_lms)
        return torch.cat((encoder_outputs, decoder_outputs), dim=0)

    def encode(self, x):
        """
        Encodes the input data into a compact latent representation and returns together with feature maps from
        individual layers.

        Parameters:
            x (Tensor): The input tensor with shape (B, N, D).

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the ID token and encoded outputs.
        """
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
        """
        Tokenizes the input regions (XYZ) into embeddings suitable for the model. This method applies a linear
        transformation to each mesh region's data to map it into the model's embedding space.

        Parameters:
            x (Tensor): The input tensor with shape (B, N, D), where B is the batch size, N is the number of vertices,
            and D is the dimensionality of each vertex.

        Returns:
            Tensor: A tensor containing the tokenized embeddings for each patch with shape (B, Npatches, dim), where
            dim is the dimensionality of the embeddings.
        """
        regions = self.get_regions(x)
        tokens = torch.stack([self.xyz_to_token[i](regions[i]) for i in range(self.Npatches)], dim=1)
        return tokens

    def decode(self, id_token, delta_lms=None):
        """
        Decodes the embeddings back to the original data space, with an option to apply manipulations. This method
        processes the encoded data (and optionally manipulations) to produce the final output. It supports adding
        delta manipulations to specific regions based on control vertices.

        Parameters:
            id_token (Tensor): The ID token tensor with shape (B, 1, dim).
            delta_lms (list of Tensors, optional): List of tensors containing delta manipulations for each control region.

        Returns:
            Tensor: The decoded output tensor with modifications applied to specified regions, if any.
        """
        B = id_token.shape[0]
        id_token = self.bottleneck_up(id_token)
        learned_decoder_tokens = self.learned_decoder_tokens.repeat(B, 1, 1)
        # For autoencoding (delta_token=0) skip updating learned_decoder_tokens altogether
        if self.manipulation:
            delta_token = [fc(delta) for fc, delta in zip(self.delta_control_net, delta_lms)]
            for i, idx in enumerate(self.control_region_keys):
                learned_decoder_tokens[:, idx] += delta_token[i]
        decoder_tokens = torch.cat((id_token, learned_decoder_tokens), dim=1)
        decoder_tokens = self.decoder(decoder_tokens)
        decoder_outputs = self.merge(decoder_tokens)
        return decoder_outputs

    def merge(self, tokens):
        """
        Merges output tokens based on region indices into a single tensor representing the final 3D structure.
        This method recombines the separately processed mesh regions into a unified representation of the 3D structure.

        Parameters:
            tokens (list of Tensors): A list of tensors, each corresponding to output from a different layer or
            processing block.

        Returns:
            Tensor: A tensor representing the combined output from all regions, with shape
            (layers, B, num_face_vertices, 3), where layers correspond to the number of tokens, B is the batch size,
            and num_face_vertices is the total number of unique vertices in the face mesh.
        """
        B = tokens[0].shape[0]
        output_tokens = [d[:, 1:] for d in tokens]
        xyz = self.inverse_tokenize(output_tokens)
        outputs = torch.zeros((len(tokens), B, self.num_face_vertices, 3)).to(tokens[0].device)
        for i, layer_output in enumerate(xyz):
            for j, face_part in enumerate(layer_output):
                outputs[i, :, self.region_ids[j]] += face_part
        return outputs

    def inverse_tokenize(self, y):
        """
        Converts the decoded embeddings back into XYZ coordinates for each facial region. This method applies a
        linear transformation to the decoded embeddings to map them back into the 3D space, effectively inverting
        the tokenization process.

        Parameters:
            y (list of Tensors): A list of decoded output tokens for each facial region, with each tensor having shape
            (B, Npatches, dim).

        Returns:
            List of lists of Tensors: A nested list where each sublist corresponds to a facial region, and contains
            tensors of XYZ coordinates for that region.
        """
        B = y[0].shape[0]
        xyz = [
            [self.token_to_xyz[i](out_tokens[:, i]).reshape(B, self.faceparts_size[i], 3) for i in range(self.Npatches)]
            for out_tokens in y
        ]
        return xyz
