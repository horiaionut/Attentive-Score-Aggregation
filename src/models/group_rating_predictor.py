import torch
import pytorch_lightning as pl

class GroupRatingPredictor(pl.LightningModule):
    def __init__(self, config, user_ratings, users_by_group):
        super().__init__()
        self.user_ratings = user_ratings
        self.users_by_group = users_by_group

        self.config = config
        self.save_hyperparameters(logger=False, ignore=['user_ratings', 'user_by_group'])

    def loss(self, batch):
        group, item, rating = batch
        return torch.nn.functional.mse_loss(self(group, item).squeeze(), rating)

    def training_step(self, batch, batch_idx):
        loss = self.loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss(batch)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        self.log('test_loss', self.loss(batch))
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, 
                    factor=self.config['lr_reduction_factor'], 
                    patience=self.config['lr_reduction_patience'], 
                    threshold=self.config['lr_reduction_threshold'],
                    cooldown=self.config['lr_reduction_cooldown']
                ),
                
                "monitor": "train_loss", 
                "interval": "step",
                "frequency": 1
            }
        }