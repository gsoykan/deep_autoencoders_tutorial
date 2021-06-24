import pytorch_lightning as pl
import torch
import torchvision


class GenerateCallback(pl.Callback):
    def __init__(self,
                 input_imgs,
                 every_n_epochs=1):
        super(GenerateCallback, self).__init__()
        self.input_imgs = input_imgs
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct Images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(input_imgs)
                pl_module.train()
            # Plot and add to tensorboard
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, range=(-1, 1))
            trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)