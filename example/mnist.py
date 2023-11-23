import argparse
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import time

print("TensorFlow version:", tf.__version__)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    level=logging.DEBUG)




# Parse CLI arguments
parser = argparse.ArgumentParser(description='MXNet Gluon MNIST Example')
parser.add_argument('--batch-size', type=int, default=32,
                    help='batch size for training and testing (default: 32)')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate (default: 0.001)')
parser.add_argument('--num-layers', type=int, default=12,
                    help='mnist number of layers (default: 12)')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='Train optimizer (defalut: adam)')
opt = parser.parse_args()



def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

'''
### mxnet-mnist example log
---
2022-11-09T04:58:20Z INFO     Epoch[9] Batch [0-100]    Speed: 36518.50 samples/sec     accuracy=0.993193
2022-11-09T04:58:20Z INFO     Epoch[9] Batch [100-200]  Speed: 50694.69 samples/sec     accuracy=0.993125
2022-11-09T04:58:20Z INFO     Epoch[9] Batch [200-300]  Speed: 50539.97 samples/sec     accuracy=0.990781
2022-11-09T04:58:20Z INFO     Epoch[9] Batch [300-400]  Speed: 50595.31 samples/sec     accuracy=0.990156
2022-11-09T04:58:20Z INFO     Epoch[9] Batch [400-500]  Speed: 50356.61 samples/sec     accuracy=0.989062
2022-11-09T04:58:21Z INFO     Epoch[9] Batch [500-600]  Speed: 47805.06 samples/sec     accuracy=0.990625
2022-11-09T04:58:21Z INFO     Epoch[9] Batch [600-700]  Speed: 37126.72 samples/sec     accuracy=0.990469
2022-11-09T04:58:21Z INFO     Epoch[9] Batch [700-800]  Speed: 36125.92 samples/sec     accuracy=0.987187
2022-11-09T04:58:21Z INFO     Epoch[9] Batch [800-900]  Speed: 36252.53 samples/sec     accuracy=0.987187
2022-11-09T04:58:21Z INFO     Epoch[9] Train-accuracy=0.990205
2022-11-09T04:58:21Z INFO     Epoch[9] Time cost=1.411
2022-11-09T04:58:21Z INFO     Epoch[9] Validation-accuracy=0.977508
'''
class MyLogger(tf.keras.callbacks.Callback):
  def __init__(self, n, endpoint):
    self.n = n   # print loss & acc every n epochs
    self.endpoint = endpoint

  def on_epoch_end(self, epoch, logs={}):
    if epoch % self.n == 0:
        for curr in self.endpoint:
            val = logs.get(curr[0])
            if curr[1] == "MAPE":
                val = logs.get("val_rmsse")
                val = 100. * ( 1 - float(val) )
            logging.info(f"Epoch[{epoch+1}] {curr[1]}={val:0<.6f}")


def train2(batch_size=32, epochs=10, lr=0.01, num_layers=2, optimizer='adam'):
    print("-- train >> Load and prepare the MNIST dataset.")
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    
    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    
    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
    
    print("-- train >> Build a machine learning model.")
    layers = [tf.keras.layers.Flatten(input_shape=(28, 28))]
    for _ in range(0, num_layers):
        layers.append(tf.keras.layers.Dense(128, activation='relu'))
        layers.append(tf.keras.layers.Dropout(0.2))
    layers.append(tf.keras.layers.Dense(10, activation='softmax'))
    model = tf.keras.models.Sequential(layers)
    
    print("-- train >> Set the optimizer")
    if optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    if optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    if optimizer == "ftrl":
        optimizer = tf.keras.optimizers.Ftrl(learning_rate=lr)

    print("-- train >> Set the optimizer")
    MAPE = tf.keras.losses.MeanAbsolutePercentageError(name='mape')
    RMSSE = tf.keras.losses.SparseCategoricalCrossentropy(name='rmsse', from_logits=False)
    ACC = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[ACC, RMSSE, MAPE]
    )
    
    print("-- train >> Train and evaluate your model")
    my_logger = MyLogger(n=1, endpoint=[('accuracy', 'Train-accuracy'), ('val_accuracy', 'Validation-accuracy'), ('val_rmsse', 'RMSSE'), ('val_mape', 'MAPE')])
    model.fit(
        ds_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=ds_test,
        callbacks=[my_logger],
        verbose=3
    )
    


if __name__ == "__main__":
    print(f"-- argparse >> {opt}")
    train2(opt.batch_size, opt.epochs, opt.lr, opt.num_layers, opt.optimizer)

