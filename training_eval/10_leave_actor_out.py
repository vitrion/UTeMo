import os
import pandas as pd
from pathlib import Path
import multiprocessing
from models.model_eval import worker_function

multiprocessing.set_start_method('spawn', force=True)

SEED = 42
PART_SEL = '60_20_20'  # FSER: '60_20_20'; MobileNet: '40_30_30'
MODEL_ID = 'fser' # or 'mobilenet'
MODALITY = 'audio' # or 'image' for MobileNet
allow_early_stopping = True
learning_rate = 0.001

image_size = (64, 64) # or (224, 224) for MobileNet

script_dir = Path(__file__).resolve().parent
data_dir = Path(__file__).resolve().parent.parent / "data"
log_folder_name = 'logs'
log_dir = script_dir / log_folder_name
try:
    df = pd.read_csv(log_dir / f"{MODEL_ID}_fine_{PART_SEL}.csv")
    max_idx = df['Test ACC'].idxmax()
    best = df.loc[max_idx]
    if MODEL_ID == 'fser':
        epochs, batch_size, optimizer, initializer = best[0].item(
        ), best[1].item(), best[2], best[3]
    else: # mobilenet
        epochs, batch_size, optimizer = best[0].item(
        ), best[1].item(), best[2]
except Exception as e:
    print("*" * 50)
    print('Exception: {}'.format(e))
    print(
        "Before running this script, make sure you have run the 'fine' stage.")
    print("*" * 50)
    exit(1)


os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'false'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'


if __name__ == "__main__":
    dataset_path = data_dir / "UTeMo_actor_validation" / f"{MODALITY}"
    unique_actors = [entry.name for entry in dataset_path.iterdir() if entry.is_dir()]
    fold_results = []
    for fold_number, test_actor in enumerate(unique_actors, start=1):
        print(
            f"\n========== "
            f"FOLD {fold_number}"
            f"=========="
        )
        print(f"Test Actor: {test_actor}")
        
        train_path = dataset_path / test_actor / "train"
        test_path = dataset_path / test_actor / "test"
        
        result_queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=worker_function, args=(
            result_queue, None, None, train_path, test_path, image_size, epochs, batch_size, optimizer, initializer, 'test', learning_rate, MODEL_ID, False, allow_early_stopping, True, SEED))
        p.start()
        accuracy = result_queue.get()
        p.join()
        p.terminate()

        print(f"Fold Accuracy: {accuracy:.4f}")
        fold_results.append(
            {
                "fold": fold_number,
                "test_actor": test_actor,
                "accuracy": accuracy
            }
        )
    results_df = pd.DataFrame(fold_results)
    mean_accuracy = results_df["accuracy"].mean()
    std_accuracy = results_df["accuracy"].std()
    print("\n======================")
    print("FINAL RESULTS")
    print("======================")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Std Accuracy: {std_accuracy:.4f}")
    results_df.to_csv(
        log_dir / f"{MODEL_ID}_leave_actor_out_results.csv", index=False)
    print(f"\nResults saved in {MODEL_ID}_leave_actor_out_results.csv")
