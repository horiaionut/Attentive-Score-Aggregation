import torch
from torch import nn

from .group_rating_predictor import GroupRatingPredictor
from .attention_layer import AttentionLayer

class MoSANLikeWeighing(GroupRatingPredictor):
    def __init__(self, config, user_ratings, users_by_group):
        super().__init__(config, user_ratings, users_by_group)

        self.context_embeds = nn.Embedding(user_ratings.shape[0], config['context_embed_dim'])
        self.user_embeds = nn.Embedding(user_ratings.shape[0], config['user_embed_dim'])
        self.item_embeds = nn.Embedding(user_ratings.shape[1], config['item_embed_dim'])
        
        self.bias = torch.nn.Parameter(torch.zeros(len(users_by_group)))

        self.attention  = AttentionLayer(config['context_embed_dim'] +  config['item_embed_dim'] + config['user_embed_dim'], config['hidden_dim'])

        self.init_trainspeedup()

        self.save_hyperparameters(logger=False, ignore=['user_ratings', 'user_by_group', 'training_speedup'])

    def init_trainspeedup(self):
        self.training_speedup = []

        for idx, members in enumerate(self.users_by_group):
            tmp = []
            for idx2 in range(len(members)):
                tmp.append( 
                    members[:idx2] + members[idx2 + 1:]                    
                )

            self.training_speedup.append([
                torch.tile(torch.tensor(members), (len(members) - 1, 1)).T.to(self.config['device']),
                torch.tensor(tmp).to(self.config['device'])
            ])

    def forward(self, groups, items):
        preds = torch.FloatTensor().to(self.config['device'])

        for group, item in zip(groups, items):
            members = self.users_by_group[group]

            context_embeds = self.context_embeds(self.training_speedup[group][0])
            members_embeds = self.user_embeds(self.training_speedup[group][1])

            item_embeds = self.item_embeds(
                torch.ones_like(
                    self.training_speedup[group][0], 
                    dtype=torch.int
                ).mul(item).to(self.config['device'])
            )

            cat = torch.cat((context_embeds, members_embeds, item_embeds), dim=2)

            _weights = self.attention(cat) # .squeeze(dim=0) ?

            weights = torch.zeros(len(members)).to(self.config['device'])

            for idx in range(len(members)):
                weights[:idx] += _weights[idx, :idx]
                weights[idx + 1:] += _weights[idx, idx:]

            preds = torch.cat((preds, (weights / len(members) @ self.user_ratings[members, item] + self.bias[group]).view(1)))

        return preds