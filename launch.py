import os
import sys
import argparse
import logging
import contextlib


class ColoredFilter(logging.Filter):
    """
    A logging filter to add color to certain log levels.
    """

    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"

    COLORS = {
        "WARNING": YELLOW,
        "INFO": GREEN,
        "DEBUG": BLUE,
        "CRITICAL": MAGENTA,
        "ERROR": RED,
    }

    RESET = "\x1b[0m"

    def __init__(self):
        super().__init__()

    def filter(self, record):
        if record.levelname in self.COLORS:
            color_start = self.COLORS[record.levelname]
            record.levelname = f"{color_start}[{record.levelname}]"
            record.msg = f"{record.msg}{self.RESET}"
        return True


def main(args, extras) -> None:
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
    from pytorch_lightning.utilities.rank_zero import rank_zero_only

    if args.typecheck:
        from jaxtyping import install_import_hook

        install_import_hook("threestudio", "typeguard.typechecked")

    import threestudio
    from threestudio.systems.base import BaseSystem
    from threestudio.utils.callbacks import (CodeSnapshotCallback, ConfigSnapshotCallback, CustomProgressBar, ProgressCallback)
    from threestudio.utils.config import ExperimentConfig, load_config
    from threestudio.utils.misc import get_rank
    from threestudio.utils.typing import Optional

    logger = logging.getLogger("pytorch_lightning")

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    for handler in logger.handlers:
        if handler.stream == sys.stderr:  # type: ignore
            if not args.gradio:
                handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
                handler.addFilter(ColoredFilter())
            else:
                handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    # Parse YAML config to OmegaConf
    cfg: ExperimentConfig
    cfg = load_config(args.config, cli_args=extras, n_gpus=1)

    # Set a different seed for each device
    pl.seed_everything(cfg.seed + get_rank(), workers=True) # TODO: cancel debug

    # Define datamodule="random-camera-datamodule"
    dm = threestudio.find(cfg.data_type)(cfg.data)

    # Define system
    system: BaseSystem = threestudio.find(cfg.system_type)(cfg.system, resumed=cfg.resume is not None)
    
    # Define save directory
    system.set_save_dir(os.path.join(cfg.trial_dir, "save"))

    # Create log dir for each experiment
    if not os.path.exists(os.path.join('logs', args.cur_time, 'log.txt')):
        with open(os.path.join('logs', args.cur_time, 'log.txt'), 'w') as file:
            file.write(cfg.trial_dir)

    if args.gradio:
        fh = logging.FileHandler(os.path.join(cfg.trial_dir, "logs"))
        fh.setLevel(logging.INFO)
        if args.verbose:
            fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(fh)

    callbacks = []

    if args.train:
        callbacks += [
            ModelCheckpoint(dirpath=os.path.join(cfg.trial_dir, "ckpts"), **cfg.checkpoint),
            LearningRateMonitor(logging_interval="step"),
            CodeSnapshotCallback(os.path.join(cfg.trial_dir, "code"), use_version=False),
            ConfigSnapshotCallback(args.config, cfg, os.path.join(cfg.trial_dir, "configs"), use_version=False),
        ]
        if args.gradio:
            callbacks += [ProgressCallback(save_path=os.path.join(cfg.trial_dir, "progress"))]
        else:
            callbacks += [CustomProgressBar(refresh_rate=1)]

    def write_to_text(file, lines):
        with open(file, "w") as f:
            for line in lines:
                f.write(line + "\n")

    loggers = []
    if args.train:
        # Make tensorboard logging directory to suppress warning
        rank_zero_only(lambda: os.makedirs(os.path.join(cfg.trial_dir, "tb_logs"), exist_ok=True))()
        loggers += [TensorBoardLogger(cfg.trial_dir, name="tb_logs"), CSVLogger(cfg.trial_dir, name="csv_logs")] + system.get_loggers()
        rank_zero_only(lambda: write_to_text(os.path.join(cfg.trial_dir, "cmd.txt"), ["python " + " ".join(sys.argv), str(args)]))()

    # Construct trainer using pl
    trainer = Trainer(
        callbacks=callbacks,
        logger=loggers,
        inference_mode=False,
        accelerator="gpu",
        devices=[0],
        **cfg.trainer,  # Provide other training settings according to the config file, e.g., precision=16-mixed
    )

    # Used for loading the model during testing
    def set_system_status(system: BaseSystem, ckpt_path: Optional[str]):
        if ckpt_path is None:
            return
        ckpt = torch.load(ckpt_path, map_location="cpu")
        system.set_resume_status(ckpt["epoch"], ckpt["global_step"])

    # Training
    if args.train:
        # Pass the instantiated pl model which uses 'threestudio' but conforms to pl's standard class construction
        trainer.fit(system, datamodule=dm, ckpt_path=cfg.resume)
        trainer.test(system, datamodule=dm)
        if args.gradio:
            # Also export assets if in gradio mode
            trainer.predict(system, datamodule=dm)

    # Validation
    elif args.validate:
        # Manually set epoch and global_step as they cannot be automatically resumed
        set_system_status(system, cfg.resume)
        trainer.validate(system, datamodule=dm, ckpt_path=cfg.resume)

    # Testing
    elif args.test:
        # Manually set epoch and global_step as they cannot be automatically resumed
        set_system_status(system, cfg.resume)
        trainer.test(system, datamodule=dm, ckpt_path=cfg.resume)

    elif args.export:
        set_system_status(system, cfg.resume)
        trainer.predict(system, datamodule=dm, ckpt_path=cfg.resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp.yaml", help="Path to config file")
    parser.add_argument("--cur_time", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true")
    group.add_argument("--validate", action="store_true")
    group.add_argument("--test", action="store_true")
    group.add_argument("--export", action="store_true")
    parser.add_argument("--gradio", action="store_true", help="If true, run in gradio mode")
    parser.add_argument("--verbose", action="store_true", help="If true, set logging level to DEBUG")
    parser.add_argument("--typecheck", action="store_true", help="Whether to enable dynamic type checking")

    args, extras = parser.parse_known_args()

    if args.gradio:
        with contextlib.redirect_stdout(sys.stderr):
            main(args, extras)
    else:
        main(args, extras)
