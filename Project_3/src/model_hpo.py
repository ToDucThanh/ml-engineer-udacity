import argparse
import os
import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from core import Pipeline_e2e

def main():
    parser = argparse.ArgumentParser(description="Grammar Classification")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 32)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5, help="learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--eps", type=float, default=1e-8, help="epsilon (default: 1e-8)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        metavar="N",
        help="The maximum length to truncate a sentence (default: 512)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        metavar="N",
        help="The number of workers used for dataloader (default: 0)"
    )
    parser.add_argument('-o',"--output-data-dir", type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('-m',"--model-dir", type=str, default=os.environ['SM_MODEL_DIR'])
    
    args = parser.parse_args()
    
    # Model and Data pipeline
    pipeline = Pipeline_e2e(
                            lr=args.lr, 
                            eps=args.eps, 
                            max_length=args.max_length, 
                            batch_size=args.batch_size, 
                            num_workers=args.num_workers
                            )
    
    checkpoint_path = "/opt/ml/checkpoints/"
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, monitor="valid_loss", mode="min")
    
    trainer = pl.Trainer(
        default_root_dir=args.output_data_dir,
        accelerator=("gpu" if torch.cuda.is_available() else "cpu"),
        max_epochs=args.epochs,
        fast_dev_run=False,
        logger=pl.loggers.TensorBoardLogger(args.output_data_dir, name="cola", version=1),
        callbacks=[checkpoint_callback],
    )
    
    # Start training
    trainer.fit(pipeline)
    
    # Save the trainer
    trainer.save_checkpoint(os.path.join(checkpoint_path, "trainer.ckpt"))
    
    # Save the latest model
    with open(os.path.join(args.model_dir, 'latest_model.pth'), 'wb') as f:
        torch.save(pipeline.state_dict(), f)
    
       
if __name__ == "__main__":
    main()