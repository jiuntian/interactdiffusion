import torch
import torch.nn as nn
from ldm.modules.attention import BasicTransformerBlock
from ldm.modules.diffusionmodules.util import checkpoint, FourierEmbedder
import torch.nn.functional as F

from ldm.modules.x_transformer import AbsolutePositionalEmbedding, FixedPositionalEmbedding


class HOIPositionNetV5(nn.Module):
    """
    Transform interaction information into interaction condition tokens
    """
    def __init__(self, in_dim, out_dim, fourier_freqs=8, max_interactions=30):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.interaction_embedding = AbsolutePositionalEmbedding(dim=out_dim, max_seq_len=max_interactions)
        self.position_embedding = AbsolutePositionalEmbedding(dim=out_dim, max_seq_len=3)
        self.position_dim = fourier_freqs * 2 * 4  # 2 is sin&cos, 4 is xyxy

        self.linears = nn.Sequential(
            nn.Linear(self.in_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

        self.linear_action = nn.Sequential(
            nn.Linear(self.in_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

        self.null_positive_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        self.null_action_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))

    def get_between_box(self, bbox1, bbox2):
        """ Between Set Operation
        Operation of Box A between Box B from Prof. Jiang idea
        """
        all_x = torch.cat([bbox1[:, :, 0::2], bbox2[:, :, 0::2]],dim=-1)
        all_y = torch.cat([bbox1[:, :, 1::2], bbox2[:, :, 1::2]],dim=-1)
        all_x, _ = all_x.sort()
        all_y, _ = all_y.sort()
        return torch.stack([all_x[:,:,1], all_y[:,:,1], all_x[:,:,2], all_y[:,:,2]],2)

    def forward(self, subject_boxes, object_boxes, masks,
                subject_positive_embeddings, object_positive_embeddings, action_positive_embeddings):
        B, N, _ = subject_boxes.shape
        masks = masks.unsqueeze(-1)

        # embedding position (it may include padding as placeholder)
        action_boxes = self.get_between_box(subject_boxes, object_boxes)
        subject_xyxy_embedding = self.fourier_embedder(subject_boxes)  # B*N*4 --> B*N*C
        object_xyxy_embedding = self.fourier_embedder(object_boxes)  # B*N*4 --> B*N*C
        action_xyxy_embedding = self.fourier_embedder(action_boxes)  # B*N*4 --> B*N*C

        # learnable null embedding
        positive_null = self.null_positive_feature.view(1, 1, -1)
        xyxy_null = self.null_position_feature.view(1, 1, -1)
        action_null = self.null_action_feature.view(1, 1, -1)

        # replace padding with learnable null embedding
        subject_positive_embeddings = subject_positive_embeddings * masks + (1 - masks) * positive_null
        object_positive_embeddings = object_positive_embeddings * masks + (1 - masks) * positive_null

        subject_xyxy_embedding = subject_xyxy_embedding * masks + (1 - masks) * xyxy_null
        object_xyxy_embedding = object_xyxy_embedding * masks + (1 - masks) * xyxy_null
        action_xyxy_embedding = action_xyxy_embedding * masks + (1 - masks) * xyxy_null

        action_positive_embeddings = action_positive_embeddings * masks + (1 - masks) * action_null

        objs_subject = self.linears(torch.cat([subject_positive_embeddings, subject_xyxy_embedding], dim=-1))
        objs_object = self.linears(torch.cat([object_positive_embeddings, object_xyxy_embedding], dim=-1))
        objs_action = self.linear_action(torch.cat([action_positive_embeddings, action_xyxy_embedding], dim=-1))

        objs_subject = objs_subject + self.interaction_embedding(objs_subject)
        objs_object = objs_object + self.interaction_embedding(objs_object)
        objs_action = objs_action + self.interaction_embedding(objs_action)

        objs_subject = objs_subject + self.position_embedding.emb(torch.tensor(0).to(objs_subject.device))
        objs_object = objs_object + self.position_embedding.emb(torch.tensor(1).to(objs_object.device))
        objs_action = objs_action + self.position_embedding.emb(torch.tensor(2).to(objs_action.device))

        objs = torch.cat([objs_subject, objs_action, objs_object], dim=1)

        assert objs.shape == torch.Size([B, N*3, self.out_dim])
        return objs