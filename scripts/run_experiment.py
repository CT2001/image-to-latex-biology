import sys
import argparse
from argparse import Namespace
from typing import List, Optional
from pytorch_lightning.loggers import TensorBoardLogger

import torch
torch.cuda.empty_cache()
import gc
gc.collect()
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint



@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    sys.path.append(cfg.project_path.path) # cfg的用法
    from image_to_latex.data import Im2Latex
    from image_to_latex.lit_models import LitResNetTransformer
    datamodule = Im2Latex(**cfg.data)#字典型参数
    datamodule.setup()

    lit_model = LitResNetTransformer(**cfg.lit_model)

    callbacks: List[Callback] = []
    if cfg.callbacks.model_checkpoint:
        callbacks.append(ModelCheckpoint(**cfg.callbacks.model_checkpoint))
    if cfg.callbacks.early_stopping:
        callbacks.append(EarlyStopping(**cfg.callbacks.early_stopping))

    # logger: Optional[WandbLogger] = None
    # if cfg.logger:
    #     logger = WandbLogger(**cfg.logger)
    logger = TensorBoardLogger("tb_logs", name="image-to-latex")

    trainer = Trainer(**cfg.trainer, callbacks=callbacks, logger=logger)

    if trainer.logger:
      trainer.logger.log_hyperparams(Namespace(**cfg))

    trainer.tune(lit_model, datamodule=datamodule)  #to find the batch size
    trainer.fit(lit_model, datamodule=datamodule)
    trainer.test(lit_model, datamodule=datamodule)


if __name__ == "__main__":
    main()
