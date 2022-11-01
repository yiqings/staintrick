from trainer import Trainer
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str, default=None)
    args = parser.parse_args()

    if args.config_path is None:
        args.config_path = os.path.join(
            os.path.abspath(os.path.join(__file__, "..")), "config.yaml"
        )
    trainer = Trainer(args.config_path)
    trainer.train()
