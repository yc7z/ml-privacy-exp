# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""JAX efficiently trains a differentially private conv net on MNIST.

This script contains a JAX implementation of Differentially Private Stochastic
Gradient Descent (https://arxiv.org/abs/1607.00133). DPSGD requires clipping
the per-example parameter gradients, which is non-trivial to implement
efficiently for convolutional neural networks.  The JAX XLA compiler shines in
this setting by optimizing the minibatch-vectorized computation for
convolutional architectures. Train time takes a few seconds per epoch on a
commodity GPU.

This code depends on tensorflow_privacy (https://github.com/tensorflow/privacy)
  Install instructions:
    $ pip install tensorflow
    $ git clone https://github.com/tensorflow/privacy
    $ cd privacy
    $ pip install .

The results match those in the reference TensorFlow baseline implementation:
  https://github.com/tensorflow/privacy/tree/master/tutorials

Example invocations:
  # this non-private baseline should get ~99% acc
  python -m examples.differentially_private_sgd \
    --dpsgd=False \
    --learning_rate=.1 \
    --epochs=20 \

   this private baseline should get ~95% acc
  python -m examples.differentially_private_sgd \
   --dpsgd=True \
   --noise_multiplier=1.3 \
   --l2_norm_clip=1.5 \
   --epochs=15 \
   --learning_rate=.25 \

  # this private baseline should get ~96.6% acc
  python -m examples.differentially_private_sgd \
   --dpsgd=True \
   --noise_multiplier=1.1 \
   --l2_norm_clip=1.0 \
   --epochs=60 \
   --learning_rate=.15 \

  # this private baseline should get ~97% acc
  python -m examples.differentially_private_sgd \
   --dpsgd=True \
   --noise_multiplier=0.7 \
   --l2_norm_clip=1.5 \
   --epochs=45 \
   --learning_rate=.25 \
"""

import itertools
import time
import warnings

from absl import app
from absl import flags

import jax
from jax import grad
from jax import jit
from jax import random
from jax import vmap
from jax.example_libraries import optimizers
from jax.example_libraries import stax
from jax.tree_util import tree_flatten, tree_unflatten
import jax.numpy as jnp
import datasets
import numpy.random as npr
import cProfile

# https://github.com/tensorflow/privacy
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
    'train with vanilla SGD.')
flags.DEFINE_float('learning_rate', .15, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 1.1,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 256, 'Batch size')
flags.DEFINE_integer('epochs', 60, 'Number of epochs')
flags.DEFINE_integer('seed', 0, 'Seed for jax PRNG')
flags.DEFINE_integer(
    'microbatches', None, 'Number of microbatches '
    '(must evenly divide batch_size)')
flags.DEFINE_string('model_dir', None, 'Model directory')
flags.DEFINE_integer(
    'noise_size', 10, 'Noise size for pre-generation')

pr = cProfile.Profile()

init_random_params, predict = stax.serial(
    stax.Conv(16, (8, 8), padding='SAME', strides=(2, 2)),
    stax.Relu,
    stax.MaxPool((2, 2), (1, 1)),
    stax.Conv(32, (4, 4), padding='VALID', strides=(2, 2)),
    stax.Relu,
    stax.MaxPool((2, 2), (1, 1)),
    stax.Flatten,
    stax.Dense(32),
    stax.Relu,
    stax.Dense(10),
)


def loss(params, batch):
  inputs, targets = batch
  logits = predict(params, inputs)
  logits = stax.logsoftmax(logits)  # log normalize
  return -jnp.mean(jnp.sum(logits * targets, axis=1))  # cross entropy loss


def accuracy(params, batch):
  inputs, targets = batch
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(predict(params, inputs), axis=1)
  return jnp.mean(predicted_class == target_class)


def clipped_grad(params, l2_norm_clip, single_example_batch):
  """Evaluate gradient for a single-example batch and clip its grad norm."""
  grads = grad(loss)(params, single_example_batch)
  nonempty_grads, tree_def = tree_flatten(grads)
  total_grad_norm = jnp.linalg.norm(
      [jnp.linalg.norm(neg.ravel()) for neg in nonempty_grads])
  divisor = jnp.maximum(total_grad_norm / l2_norm_clip, 1.)
  normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
  return tree_unflatten(tree_def, normalized_nonempty_grads)


def private_grad(params, batch, rng, l2_norm_clip, noise_multiplier,
                 batch_size, current_noise):
  """Return differentially private gradients for params, evaluated on batch."""
  clipped_grads = vmap(clipped_grad, (None, None, 0))(params, l2_norm_clip, batch)
  clipped_grads_flat, grads_treedef = tree_flatten(clipped_grads)
  aggregated_clipped_grads = [g.sum(0) for g in clipped_grads_flat]
  noised_aggregated_clipped_grads = [
      g + l2_norm_clip * noise_multiplier * current_noise[i]
      for i, g in zip(range(len(aggregated_clipped_grads)), aggregated_clipped_grads)]
  normalized_noised_aggregated_clipped_grads = [
      g / batch_size for g in noised_aggregated_clipped_grads]
  return tree_unflatten(grads_treedef, normalized_noised_aggregated_clipped_grads)


def shape_as_image(images, labels, dummy_dim=False):
  target_shape = (-1, 1, 28, 28, 1) if dummy_dim else (-1, 28, 28, 1)
  return jnp.reshape(images, target_shape), labels


def compute_epsilon(steps, num_examples=60000, target_delta=1e-5):
  if num_examples * target_delta > 1.:
    warnings.warn('Your delta might be too high.')
  q = FLAGS.batch_size / float(num_examples)
  orders = list(jnp.linspace(1.1, 10.9, 99)) + list(range(11, 64))
  rdp_const = compute_rdp(q, FLAGS.noise_multiplier, steps, orders)
  eps, _, _ = get_privacy_spent(orders, rdp_const, target_delta=target_delta)
  return eps

# splitting version
def pre_generate_noises(grad_shapes, rng, i):
  rng = random.fold_in(rng, i)
  rngs = random.split(rng, len(grad_shapes))
  noises = []
  for i in range(FLAGS.noise_size):
    noises.append([])
  for r, shape in zip(rngs, grad_shapes):
    long_noise = random.normal(r, shape)
    splitted_noises = jax.numpy.split(long_noise, FLAGS.noise_size)
    # print("shape of long_noise is ", long_noise.shape)
    # splitted_noises = vmap(jax.numpy.split, (0 ,0))(long_noise, FLAGS.noise_size)
    for i in range(len(noises)):
      noises[i].append(jax.lax.collapse(splitted_noises[i], 0, 2))

  return noises


# indexing version 
# def pre_generate_noises(grad_shapes, rng, i):
#   rng = random.fold_in(rng, i)
#   rngs = random.split(rng, len(grad_shapes))
#   noises = []
#   for i in range(FLAGS.noise_size):
#     noises.append([])
#   for r, shape in zip(rngs, grad_shapes):
#     long_noise = random.normal(r, shape)
#     # splitted_noises = jax.numpy.split(long_noise, FLAGS.noise_size)
#     # print("shape of long_noise is ", long_noise.shape)
#     # splitted_noises = vmap(jax.numpy.split, (0 ,0))(long_noise, FLAGS.noise_size)
#     for i in range(len(noises)):
#       noises[i].append(long_noise[0])

#   return noises



def main(_):
  if FLAGS.microbatches:
    raise NotImplementedError(
        'Microbatches < batch size not currently supported'
    )

  grad_shapes = [(FLAGS.noise_size, 8, 8, 1, 16), (FLAGS.noise_size, 1, 1, 1, 16), (FLAGS.noise_size, 4, 4, 16, 32),\
     (FLAGS.noise_size, 1, 1, 1, 32), (FLAGS.noise_size, 512, 32), (FLAGS.noise_size, 32,), (FLAGS.noise_size, 32, 10),\
      (FLAGS.noise_size, 10,)]
  


  train_images, train_labels, test_images, test_labels = datasets.mnist()
  num_train = train_images.shape[0]
  num_complete_batches, leftover = divmod(num_train, FLAGS.batch_size)
  num_batches = num_complete_batches + bool(leftover)
  print("number of batches is ", num_batches)
  key = random.PRNGKey(FLAGS.seed)

  def data_stream():
    rng = npr.RandomState(FLAGS.seed)
    while True:
      perm = rng.permutation(num_train)
      for i in range(num_batches):
        batch_idx = perm[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size]
        yield train_images[batch_idx], train_labels[batch_idx]

  batches = data_stream()

  opt_init, opt_update, get_params = optimizers.sgd(FLAGS.learning_rate)

  @jit
  def update(_, i, opt_state, batch):
    params = get_params(opt_state)
    return opt_update(i, grad(loss)(params, batch), opt_state)

  @jit
  def private_update(rng, i, opt_state, batch, current_noise):
    # a = time.time()
    # pr.enable()
    params = get_params(opt_state)
    rng = random.fold_in(rng, i)  # get new key for new random numbers
    result = opt_update(
        i,
        private_grad(params, batch, rng, FLAGS.l2_norm_clip,
                     FLAGS.noise_multiplier, FLAGS.batch_size, current_noise), opt_state)
    # pr.disable()
    # print("updated time is ", time.time() - a)
    # pr.dump_stats("private_new.prof")
    return result

  _, init_params = init_random_params(key, (-1, 28, 28, 1))
  opt_state = opt_init(init_params)
  itercount = itertools.count()

  steps_per_epoch = 60000 // FLAGS.batch_size

  print("steps_per_epoch is ", steps_per_epoch)
  print("num_batches is ", num_batches)

  print('\nStarting training...')
  for epoch in range(1, FLAGS.epochs + 1):
    start_time = time.time()
    pr.enable()
    for turn in range(num_batches):
      if FLAGS.dpsgd:
        looper = turn % FLAGS.noise_size
        if looper == 0:
          pre_noise = pre_generate_noises(grad_shapes, key, 1) # it is not possible to get current value of itertools
        opt_state = \
            private_update(
                key, next(itercount), opt_state,
                shape_as_image(*next(batches), dummy_dim=True), pre_noise[looper])
      else:
        opt_state = update(
            key, next(itercount), opt_state, shape_as_image(*next(batches)))
    epoch_time = time.time() - start_time
    pr.disable()
    if FLAGS.dpsgd:
      pr.dump_stats(f"Pregenerate_noise_private{FLAGS.noise_multiplier}|{FLAGS.l2_norm_clip}|{FLAGS.batch_size}|{FLAGS.noise_size}.prof")
    else:
      pr.dump_stats(f"public{FLAGS.noise_multiplier}|{FLAGS.l2_norm_clip}|{FLAGS.batch_size}.prof")
    print(f'Epoch {epoch} in {epoch_time:0.2f} sec')
    print(f'batch training time {epoch} in {epoch_time/num_batches:0.10f} sec')

    # evaluate test accuracy
    params = get_params(opt_state)
    test_acc = accuracy(params, shape_as_image(test_images, test_labels))
    test_loss = loss(params, shape_as_image(test_images, test_labels))
    print('Test set loss, accuracy (%): ({:.2f}, {:.2f})'.format(
        test_loss, 100 * test_acc))

    # determine privacy loss so far
    if FLAGS.dpsgd:
      delta = 1e-5
      num_examples = 60000
      eps = compute_epsilon(epoch * steps_per_epoch, num_examples, delta)
      print(
          f'For delta={delta:.0e}, the current epsilon is: {eps:.2f}')
    else:
      print('Trained with vanilla non-private SGD optimizer')


if __name__ == '__main__':
  app.run(main)
