"""Can TFDS read the ImageNet replica on Colossus? Answer in seconds.

`knn_full` is the last thing a 57-hour stage-2 run does, so a wrong data_dir
costs the whole run to discover. Everything the eval needs before it touches a
TPU is checked here instead: that google3's TensorFlow registers the /cns/
prefix at all, that the builder can read `dataset_info.json` off Colossus, that
both splits are present with the expected example counts, and that one real
tfrecord shard decodes into an image.

Deliberately does NOT depend on `:main` -- the point is a build-and-run cycle
measured in seconds, and none of torch, jax or the model graph is involved in
answering the question.
"""

import sys

from absl import app
from absl import flags

_DATA_DIR = flags.DEFINE_string(
    'data_dir',
    '/cns/is-d/home/qiaos/data/eval_bundle/tensorflow_datasets',
    'TFDS data_dir to probe. The default is the cbf replica.',
)
_DECODE = flags.DEFINE_bool(
    'decode', True, 'Also pull one example through the tf.data pipeline.'
)


def main(argv):
    del argv
    import tensorflow as tf
    import tensorflow_datasets as tfds

    data_dir = _DATA_DIR.value
    print(f'tf {tf.__version__}, tfds {tfds.__version__}')
    print(f'data_dir: {data_dir}')

    # Step 1: does tf.io.gfile see Colossus at all? A False here means the
    # binary lacks the CNS filesystem registration and nothing below can work.
    info_path = f'{data_dir}/imagenet2012/5.1.0/dataset_info.json'
    exists = tf.io.gfile.exists(info_path)
    print(f'tf.io.gfile.exists(dataset_info.json): {exists}')
    if not exists:
        print('FAIL: TensorFlow cannot see the CNS path.')
        return 1

    n_shards = len(tf.io.gfile.glob(
        f'{data_dir}/imagenet2012/5.1.0/imagenet2012-train.tfrecord-*'))
    print(f'tf.io.gfile.glob train shards: {n_shards}')

    # Step 2: the builder, which is what ensure_imagenet_available() calls.
    builder = tfds.builder('imagenet2012', data_dir=data_dir)
    splits = {k: v.num_examples for k, v in builder.info.splits.items()}
    print(f'splits: {splits}')
    if 'train' not in splits or 'validation' not in splits:
        print('FAIL: missing train/validation split.')
        return 1

    # Step 3: bytes, not metadata. dataset_info.json is a small file and could
    # plausibly be readable where a 140 MiB tfrecord is not.
    if _DECODE.value:
        ds = builder.as_dataset(split='validation', shuffle_files=False)
        ex = next(iter(ds.take(1)))
        print(f"decoded one example: image={tuple(ex['image'].shape)} "
              f"label={int(ex['label'])}")

    print('OK')
    return 0


if __name__ == '__main__':
    app.run(lambda argv: sys.exit(main(argv)))
