import os
import numpy as np
import random
import contextlib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product
import multiprocessing
from sklearn.preprocessing import LabelEncoder

multiprocessing.set_start_method('spawn', force=True)

from models.model_eval import create_train_and_test_model, get_dataset, generate_epoch_list, worker_function

SEED = 42
PART_SEL = '60_20_20'  # FSER: '60_20_20'; MobileNet: '40_30_30'
MODEL_ID = 'fser' # or 'mobilenet'
MODALITY = 'audio' # or 'image' for MobileNet
stage = 'train'  # Stages: 'coarse', 'fine', 'train', 'test'
allow_early_stopping = (stage == 'coarse' or stage == 'fine')
learning_rate = 0.001

image_size = (64, 64) # or (224, 224) for MobileNet

log_folder_name = 'logs'
models_folder_name = 'models'
figures_folder_name = 'figures'
script_dir = Path(__file__).resolve().parent
data_dir = Path(__file__).resolve().parent.parent / "data"
log_dir = script_dir / log_folder_name
models_dir = script_dir / models_folder_name
figures_dir = script_dir / figures_folder_name
os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)
data_path = data_dir / f"UTeMo_{MODALITY}_{PART_SEL}"


if len(PART_SEL.split("_")) != 3:
    print("Partitioning selection length must be equal to three.")
    exit(1)

MEAN_EPOCHS = 300
COARSE_EPOCH_STEP = 100
COARSE_EPOCH_ELEMENTS = 3

FINE_EPOCH_STEP = 50
FINE_EPOCH_ELEMENTS = 3

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'false'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

random.seed(SEED)
np.random.seed(SEED)


if __name__ == "__main__":
    train_path = data_path / "train"
    val_path = data_path / "val"
    test_path = data_path / "test"
    trainplusval_path = data_path / "train+val"

    num_classes = 7
    label_encoder = LabelEncoder()

    if stage == 'coarse' or stage == 'fine':
        if stage == 'coarse':
            if MODEL_ID == 'fser':
                hyperparams = {
                    'epoch_set': generate_epoch_list(MEAN_EPOCHS, COARSE_EPOCH_STEP, COARSE_EPOCH_ELEMENTS),
                    'batch_set': [8, 16, 32],
                    'optimizer_set': ['Adam', 'SGD'],
                    'initializer_set': ['HeNormal', 'GlorotNormal']
                }
            else: # mobilenet
                hyperparams = {
                    'epoch_set': generate_epoch_list(MEAN_EPOCHS, COARSE_EPOCH_STEP, COARSE_EPOCH_ELEMENTS),
                    'batch_set': [8, 16, 32],
                    'optimizer_set': ['Adam', 'SGD']
                }
        elif stage == 'fine':
            try:
                df = pd.read_csv(
                    log_dir / f"{MODEL_ID}_coarse_{PART_SEL}.csv")
                max_idx = df['Test ACC'].idxmax()
                best = df.loc[max_idx]

                if MODEL_ID == 'fser':
                    hyperparams = {
                        'epoch_set': generate_epoch_list(best[0].item(), FINE_EPOCH_STEP, FINE_EPOCH_ELEMENTS),
                        'batch_set': [best[1].item()],
                        'optimizer_set': [best[2]],
                        'initializer_set': [best[3]]
                    }
                else: # mobilenet
                    hyperparams = {
                    'epoch_set': generate_epoch_list(best[0].item(), FINE_EPOCH_STEP, FINE_EPOCH_ELEMENTS),
                    'batch_set': [best[1].item()],
                    'optimizer_set': [best[2]]
                }
            except Exception as e:
                print("*" * 50)
                print('Exception: {}'.format(e))
                print(
                    "Before running the 'fine' stage, make sure you have run the 'coarse' stage.")
                print("*" * 50)

        if MODEL_ID == 'fser':
            grid = list(product(
                hyperparams['epoch_set'],
                hyperparams['batch_set'],
                hyperparams['optimizer_set'],
                hyperparams['initializer_set']
            ))
        else: # mobilenet
            grid = list(product(
                hyperparams['epoch_set'],
                hyperparams['batch_set'],
                hyperparams['optimizer_set']
            ))

        csv_path = log_dir / f"{MODEL_ID}_{stage}_{PART_SEL}.csv"
        if os.path.exists(csv_path):
            df_existing = pd.read_csv(csv_path)
            first_zero_idx = df_existing[df_existing['Test ACC'] == 0.0].index.min(
            )

            if pd.notna(first_zero_idx):
                df_existing = df_existing.iloc[:first_zero_idx]

            if MODEL_ID == 'fser':
                completed = set(
                    tuple(row) for row in df_existing[['Epochs', 'Batch Size', 'Optimizer', 'Initializer']].values
                )
            else: # mobilenet
                completed = set(
                    tuple(row) for row in df_existing[['Epochs', 'Batch Size', 'Optimizer']].values
                )
            results = df_existing.values.tolist()
        else:
            completed = set()
            results = []

        for combo in grid:
            if MODEL_ID == 'fser':
                epochs, batch_size, optimizer, initializer = combo
            else: # mobilenet
                epochs, batch_size, optimizer = combo
                initializer = None
            if combo in completed:
                print(f"Skipping already done: {combo}")
                continue
            print(combo)

            result_queue = multiprocessing.Queue()
            p = multiprocessing.Process(target=worker_function, args=(
                result_queue, train_path, val_path, None, test_path, image_size, epochs, batch_size, optimizer, initializer, stage, learning_rate, MODEL_ID, False, allow_early_stopping, True, SEED))
            p.start()
            test_acc = result_queue.get()
            p.join()
            p.terminate()

            row = list(combo)
            row.append(test_acc)
            results.append(row)
            resultsDF = pd.DataFrame(results)
            if MODEL_ID == 'fser':
                resultsDF.to_csv(log_dir / f"{MODEL_ID}_{stage}_{PART_SEL}.csv", index=False, header=[
                    'Epochs', 'Batch Size', 'Optimizer', 'Initializer', 'Test ACC']
                )
            else: # mobilenet
                resultsDF.to_csv(log_dir / f"{MODEL_ID}_{stage}_{PART_SEL}.csv", index=False, header=[
                    'Epochs', 'Batch Size', 'Optimizer', 'Test ACC']
                )
    else:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(SEED)
        tf.random.set_seed(SEED)
        tf.config.experimental.enable_op_determinism()

        try:
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                for i in range(len(physical_devices)):
                    tf.config.experimental.set_memory_growth(
                        physical_devices[i], True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(physical_devices), "Physical GPUs,",
                    len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            print(e)

        try:
            df = pd.read_csv(
                log_dir / f"{MODEL_ID}_fine_{PART_SEL}.csv")
            max_idx = df['Test ACC'].idxmax()
            best = df.loc[max_idx]

            if MODEL_ID == 'fser':
                epochs, batch_size, optimizer, initializer = best[0].item(
                ), best[1].item(), best[2], best[3]
            else: # mobilenet
                epochs, batch_size, optimizer, initializer = best[0].item(
                ), best[1].item(), best[2], None
        except Exception as e:
            print("*" * 50)
            print('Exception: {}'.format(e))
            print(
                "Before running the 'train'/'test' stage, make sure you have run the 'fine' stage.")
            print("*" * 50)

        data = get_dataset(train_path, val_path, trainplusval_path, test_path, batch_size, image_size, stage, SEED)
        if stage == 'test':
            test_acc, test_f1, _, model = create_train_and_test_model(data,
                                                                    epochs, batch_size, optimizer, initializer, stage, learning_rate, MODEL_ID, False, allow_early_stopping, False)
            model.save(models_dir / f"{MODEL_ID}_{PART_SEL}.keras")
            
            def final(test_acc, test_f1):
                print("*" * 50)
                print(f"Final test accuracy: {test_acc}")
                print(f"Final test f1-sore: {test_f1}")
                print("*" * 50)

            with open(log_dir / f"{MODEL_ID}_final_{stage}_log_{PART_SEL}.txt", "w", encoding="utf-8") as f:
                with contextlib.redirect_stdout(f):
                    final(test_acc, test_f1)

        elif stage == 'train':
            if os.path.exists(log_dir / f"{MODEL_ID}_training_history_{PART_SEL}.csv"):
                print("Training history file found. Skipping training and generating plots...")
                history_df = pd.read_csv(log_dir / f"{MODEL_ID}_training_history_{PART_SEL}.csv")
            else:
                print("No training history found. Starting training...")
                _, _, history, _ = create_train_and_test_model(data,
                                                            epochs, batch_size, optimizer, initializer, stage, learning_rate, MODEL_ID, False, allow_early_stopping, True)
                history_df = pd.DataFrame(history.history)
                history_df.insert(
                    0,
                    "epoch",
                    range(1, len(history_df) + 1)
                )
                history_df.to_csv(
                    log_dir / f"{MODEL_ID}_training_history_{PART_SEL}.csv",
                    index=False
                )

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            # --- Subplot 1: Accuracy ---
            axes[0].plot(
                history_df["accuracy"],
                label="Training Accuracy"
            )
            axes[0].plot(
                history_df["val_accuracy"],
                label="Validation Accuracy"
            )
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Accuracy")
            axes[0].legend()

            # --- Subplot 2: Loss ---
            axes[1].plot(
                history_df["loss"],
                label="Training Loss"
            )
            axes[1].plot(
                history_df["val_loss"],
                label="Validation Loss"
            )
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Loss")
            axes[1].legend()

            plt.tight_layout()
            plt.savefig(
                figures_dir / f"{MODEL_ID}_training_curves_{PART_SEL}.png",
                dpi=300
            )
            plt.close()
        else:
            print("No such stage.")
            exit(1)
