import torch
from torch import nn

from .group_rating_predictor import GroupRatingPredictor
from .attention_layer import AttentionLayer

class AGREELikeWeighing(GroupRatingPredictor):
    def __init__(self, config, user_ratings, users_by_group, user_embeds, item_embeds):
        super().__init__(config, user_ratings, users_by_group)

        self.user_embeds = user_embeds
        self.item_embeds = item_embeds
        self.bias = torch.nn.Parameter(torch.zeros(len(users_by_group)))
        self.attention  = AttentionLayer(config['user_embed_dim'] +  config['item_embed_dim'], config['hidden_dim'])

    def forward(self, groups, items):
        preds = torch.FloatTensor().to(self.config['device'])

        for group, item in zip(groups, items):
            members = self.users_by_group[group]

            members_embeds = self.user_embeds(torch.LongTensor(members).to(self.config['device']))
            item_embeds = self.item_embeds(torch.LongTensor([item]* len(members)).to(self.config['device']))

            weights = self.attention(torch.cat((members_embeds, item_embeds), dim=1))
            preds = torch.cat((preds, (weights.squeeze() @ self.user_ratings[members, item] + self.bias[group]).view(1)))

        return preds

class AGREELikeWeighingStaticEmbeds(AGREELikeWeighing):
    def __init__(self, config, user_ratings, users_by_group):
        super().__init__(config, user_ratings, users_by_group, 
            nn.Embedding(user_ratings.shape[0], config['user_embed_dim']), 
            nn.Embedding(user_ratings.shape[1], config['item_embed_dim'])
        )

class AGREELikeWighingWithEncoder(AGREELikeWeighing):
    def __init__(self, config, user_ratings, users_by_group, ):
        super().__init__(config, user_ratings, users_by_group, 
            Encoder(user_ratings, config['user_embed_dim']),
            Encoder(user_ratings.T, config['item_embed_dim']),
        )

class Encoder(nn.Module):
    def __init__(self, user_ratings, embed_dim) -> None:
        super().__init__()

        self.user_ratings = user_ratings

        self.layers = nn.Sequential(
            nn.Linear(user_ratings.shape[1], 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(self.user_ratings[x])