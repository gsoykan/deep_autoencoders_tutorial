import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from deep_autoencoder_tutorial.model.autoencoder import Autoencoder
from deep_autoencoder_tutorial.training.generate_callback import GenerateCallback

CHECKPOINT_PATH = "../saved_models/tutorial9"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def train_cifar(train_loader,
                val_loader,
                test_loader,
                latent_dim,
                input_imgs_for_reconstruction):
    # Create a PyTorch Lightning trainer with the generation callback
    # Very impressive callback usage
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "cifar10_%i" % latent_dim),
                         gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=500,
                         callbacks=[ModelCheckpoint(save_weights_only=True),
                                    GenerateCallback(input_imgs_for_reconstruction, every_n_epochs=10),
                                    LearningRateMonitor("epoch")])
    trainer.logger._log_graph = True  # if True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "cifar10_%i.ckpt" % latent_dim)
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = Autoencoder.load_from_checkpoint(pretrained_filename)
    else:
        model = Autoencoder(base_channel_size=32, latent_dim=latent_dim)
        trainer.fit(model, train_loader, val_loader)

    # Test best model on validation and test set
    val_result = trainer.test(model, test_dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}
    return model, result
