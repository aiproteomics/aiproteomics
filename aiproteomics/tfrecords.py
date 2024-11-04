import tensorflow as tf
from functools import partial

AUTOTUNE = tf.data.AUTOTUNE

BATCH_SIZE = 64
VOCABULARY_SIZE = 30
SEQUENCE_LENGTH = 50
OUTPUT_SHAPE = 392

def read_tfrecord(example, labeled):
    print(f"This is what an example looks like: {example}")

    # Record spec
    if labeled:
        tfrecord_format = (
            {
                "charge": tf.io.FixedLenFeature([1], tf.int64),
                "msms": tf.io.FixedLenFeature([OUTPUT_SHAPE], tf.float32),
                "pep": tf.io.FixedLenFeature([SEQUENCE_LENGTH], tf.int64)
            }
        )
    else:
        tfrecord_format = (
            {
                "charge": tf.io.FixedLenFeature([1], tf.int64),
                "pep": tf.io.FixedLenFeature([SEQUENCE_LENGTH], tf.int64)
            }
        )

    # Parse data according to spec
    example = tf.io.parse_single_example(example, tfrecord_format)

    print(f"Parsed example: {example}")

    features = {"charge": tf.cast(example["charge"], tf.float32),
                "pep": tf.cast(example["pep"], tf.int32)}

    if labeled:
        label = tf.cast(example["msms"], tf.float32, name="msms")

        return features, label

    return features


def load_dataset(filenames, labeled):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        filenames
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(
        partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE
    )

    return dataset


def get_dataset(filenames, labeled=True):
    dataset = load_dataset(filenames, labeled)
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset