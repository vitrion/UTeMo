import numpy as np
import tensorflow as tf
import keras
import random
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
class_names = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

def get_early_stoppers(allow_early_stopping, stage):
    delta = 0.001

    if stage == 'test':
        metric = 'loss'
    else:
        metric = 'val_loss'
    if allow_early_stopping:
        class PrintPatience(tf.keras.callbacks.Callback):
            def __init__(self, early_stopping_callback):
                super().__init__()
                self.early_stopping_callback = early_stopping_callback

            def on_epoch_end(self, epoch, logs=None):
                current_wait = self.early_stopping_callback.wait
                patience = self.early_stopping_callback.patience
                print(f"\nPatience counter: {current_wait}/{patience}")

        early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor=metric,
            min_delta=delta,
            patience=5,
            mode='min',
            verbose=1
        )

        print_early_stopping_callback = PrintPatience(early_stopping_callback)
        return [early_stopping_callback, print_early_stopping_callback]
    else:
        return []

def create_train_and_test_model(data, epochs, batch_size, optimizer, initializer, stage, learning_rate, model_id, load_from_file=False, allow_early_stopping=True, verbose=True):
    try:
        script_dir = Path(__file__).resolve().parent
        cb = get_early_stoppers(allow_early_stopping, stage)
        train_ds, val_ds, trainplusval_ds, test_ds = data
        
        if load_from_file:
            if model_id == 'fser':
                epochs = epochs // 20
                model = tf.keras.models.load_model(script_dir / "fser_60_20_20.keras")
            else:  # mobilenet
                epochs = epochs // 10
                model = tf.keras.models.load_model(script_dir / "mobilenet_40_30_30.keras")
        else:
            if model_id == 'fser':
                init = eval(f'tf.keras.initializers.{initializer}()')
            
                model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(64, 64, 3)),
                    tf.keras.layers.Rescaling(1./255),
                    tf.keras.layers.Conv2D(8, (5, 5), input_shape=(
                        64, 64, 3), activation='relu', kernel_initializer=init, padding='same'),
                    # 2,2 es el tamano de la matriz
                    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                    tf.keras.layers.Dropout(rate=0.2),

                    tf.keras.layers.Conv2D(
                        16, (5, 5), activation='relu', kernel_initializer=init, padding='same'),
                    # 2,2 es el tamano de la matriz
                    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                    tf.keras.layers.Dropout(rate=0.2),

                    tf.keras.layers.Conv2D(
                        100, (5, 5), activation='relu', kernel_initializer=init, padding='same'),
                    # 2,2 es el tamano de la matriz
                    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                    tf.keras.layers.Dropout(rate=0.2),

                    tf.keras.layers.Conv2D(
                        200, (5, 5), activation='relu', kernel_initializer=init, padding='same'),
                    # 2,2 es el tamano de la matriz
                    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                    tf.keras.layers.Dropout(rate=0.2),

                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(
                        units=17424, activation='relu', kernel_initializer=init),
                    tf.keras.layers.Dense(
                        units=1024,  activation='relu', kernel_initializer=init),
                    tf.keras.layers.Dense(
                        units=500, activation='relu', kernel_initializer=init),
                    tf.keras.layers.Dense(7, activation="softmax")
                ])
            else:  # mobilenet
                base_model = tf.keras.applications.MobileNet(
                    input_shape=(224, 224, 3),
                    alpha=1.0,
                    depth_multiplier=1,
                    dropout=0.5,
                    include_top=False,
                    weights="imagenet"
                )
                base_model.trainable = True
                
                model = tf.keras.Sequential([
                    base_model,
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dense(1000, activation="relu"),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(7, activation="softmax")
                ])

        if verbose:
            model.summary()

        loss = "categorical_crossentropy"
        metrics = ["accuracy"]
        optmzr = eval(
                    f'tf.keras.optimizers.{optimizer}(learning_rate={learning_rate})')
        
        model.compile(
            optimizer=optmzr,
            loss=loss,
            metrics=metrics
        )
        
        if stage == 'test':
            history = model.fit(
            trainplusval_ds, epochs=epochs, verbose=1, batch_size=batch_size, callbacks = cb)
        else:
            history = model.fit(
            train_ds, validation_data=val_ds, epochs=epochs, verbose=1, batch_size=batch_size, callbacks = cb)
        
        test_ds_ordered = test_ds.unbatch().batch(batch_size) 
        y_true = []
        y_pred = []
        for images, labels in test_ds_ordered:
            true_indices = np.argmax(labels.numpy(), axis=-1)
            y_true.extend(true_indices)
            preds = model.predict(images, verbose=0)
            pred_indices = np.argmax(preds, axis=-1)
            y_pred.extend(pred_indices)
        y_true = np.array(y_true).astype(int)
        y_pred = np.array(y_pred).astype(int)

        test_acc = accuracy_score(y_true, y_pred)
        test_f1 = f1_score(y_true, y_pred, average='macro')

    except Exception as e:
        print("*" * 50)
        print('Exception: {}'.format(e))
        print("*" * 50)
        test_acc = 0
        test_f1 = 0
        history = None
        model = None

    return test_acc, test_f1, history, model

def get_dataset(train_path, val_path, trainplusval_path, test_path, batch_size, image_size, stage='train', seed=42):
    if stage == 'test':
        print(f"{trainplusval_path}:")
        trainplusval_ds = tf.keras.utils.image_dataset_from_directory(
            trainplusval_path,
            batch_size=batch_size,
            seed=seed,
            labels='inferred',
            class_names=class_names,
            label_mode='categorical',
            image_size=image_size,
            color_mode='rgb')
        print(f"{test_path}:")
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_path,
            shuffle=False,
            batch_size=batch_size,
            labels='inferred',
            class_names=class_names,
            label_mode='categorical',
            image_size=image_size,
            color_mode='rgb')   
        return None, None, trainplusval_ds, test_ds
    else:
        print(f"{train_path}:")
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_path,
            batch_size=batch_size,
            seed=seed,
            labels='inferred',
            class_names=class_names,
            label_mode='categorical',
            image_size=image_size,
            color_mode='rgb')
        print(f"{val_path}:")
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_path,
            shuffle=False,
            batch_size=batch_size,
            labels='inferred',
            class_names=class_names,
            label_mode='categorical',
            image_size=image_size,
            color_mode='rgb')
        print(f"{test_path}:")
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_path,
            shuffle=False,
            batch_size=batch_size,
            labels='inferred',
            class_names=class_names,
            label_mode='categorical',
            image_size=image_size,
            color_mode='rgb')
        return train_ds, val_ds, None, test_ds

def worker_function(result_queue, train_path, val_path, trainplusval_path, test_path, image_size, epochs, batch_size, optimizer, initializer, stage, learning_rate, model_id, load_from_file, allow_early_stopping, verbose, seed):
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)
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

    data = get_dataset(train_path, val_path, trainplusval_path, test_path, batch_size, image_size, stage, seed)
    accuracy, _, _, _ = create_train_and_test_model(data,
                                                    epochs, batch_size, optimizer, initializer, stage, learning_rate, model_id, load_from_file, allow_early_stopping, verbose)
    result_queue.put((accuracy))


def generate_epoch_list(mean_value, separation, num_elements=7):
    if num_elements % 2 == 0:
        num_elements += 1
    radio = num_elements // 2
    lista_epocas = [mean_value + (i * separation)
                    for i in range(-radio, radio + 1)]
    return lista_epocas