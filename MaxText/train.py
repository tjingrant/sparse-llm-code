"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# pylint: disable=g-bad-todo, abstract-method, consider-using-with, ungrouped-imports
"""Training loop and Decoding of the model."""

# Calling jax.device_count here prevents a "TPU platform already registered" error.
# See github.com/google/maxtext/issues/20 for more

import datetime
import os
import sys
import functools
import time

from typing import Sequence, Optional
from absl import app
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
import grain.python as grain
import jax
import numpy as np
import orbax.checkpoint
import orbax.checkpoint.experimental.emergency.checkpoint_manager as emergency_checkpoint_manager

import checkpointing
import max_utils
import maxtext_utils
import max_logging
import optimizers
import profiler
import pyconfig
import pathwaysutils  # pylint: disable=unused-import

from vertex_tensorboard import VertexTensorboardManager
# Placeholder: internal

from input_pipeline.input_pipeline_interface import create_data_iterator
from layers import models

import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh
from jax.experimental import checkify

from cloud_tpu_diagnostics import diagnostic
from cloud_tpu_diagnostics.configuration import debug_configuration
from cloud_tpu_diagnostics.configuration import diagnostic_configuration
from cloud_tpu_diagnostics.configuration import stack_trace_configuration

from layers import quantizations

from ml_goodput_measurement import goodput
from ml_goodput_measurement import monitoring

# pylint: disable=too-many-positional-arguments

Transformer = models.Transformer
EPS = 1e-8
_DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE = 2 * 1024**3


def validate_train_config(config):
  """Validates the configuration is set correctly for train.py"""

  assert config.run_name, "Erroring out, need a real run_name"
  if not config.dataset_path.startswith("gs://"):
    max_logging.log("WARNING: 'dataset_path' might be pointing your local file system")
  if not config.base_output_directory.startswith("gs://"):
    max_logging.log("WARNING: 'base_output_directory' might be pointing your local file system")
  assert config.steps > 0, "You must set steps or learning_rate_schedule_steps to a positive integer."
  if config.quantization == "fp8":
    # pylint: disable=line-too-long
    assert (
        config.gradient_accumulation_steps == 1
    ), "fp8 can't be used with gradient_accumulation_steps right now. Please use other quantization or set gradient_accumulation_steps to 1"


def get_first_step(state):
  with jax.spmd_mode("allow_all"):
    return int(state.step)


def load_next_batch(train_iter, example_batch, config):
  """Loads the next batch. Can keep reusing the same batch for performance reasons"""

  if config.reuse_example_batch and example_batch is not None:
    return example_batch
  else:
    return next(train_iter)


def record_scalar_metrics(metrics, step_time_delta, per_device_tflops, lr, per_device_tokens):
  """Records scalar metrics to be written to tensorboard"""
  metrics["scalar"].update({"perf/step_time_seconds": step_time_delta.total_seconds()})
  metrics["scalar"].update({"perf/per_device_tflops": per_device_tflops})
  metrics["scalar"].update({"perf/per_device_tflops_per_sec": per_device_tflops / step_time_delta.total_seconds()})
  metrics["scalar"].update({"perf/per_device_tokens": per_device_tokens})
  metrics["scalar"].update({"perf/per_device_tokens_per_sec": per_device_tokens / step_time_delta.total_seconds()})
  metrics["scalar"].update({"learning/current_learning_rate": lr})


_buffered_step = None
_buffered_metrics = None


def write_metrics(writer, local_metrics_file, running_gcs_metrics, metrics, step, config, is_training=True):
  """Entry point for all metrics writing in Train's Main.
  TODO: would be better as a Class in the future (that initialized all state!)

  To avoid introducing an unnecessary dependency, we "double buffer" -- we hold
  onto the last metrics and step and only publish when we receive a new metrics and step.
  The logic is that this ensures that Jax is able to queues train_steps and we
  don't block when turning "lazy" Jax arrays into real Python numbers.
  """
  metrics_to_write, steps_to_write = None, None
  if is_training:
    global _buffered_step, _buffered_metrics
    if _buffered_metrics is not None:
      if _buffered_step is None:
        raise ValueError(f"When writing metrics, {_buffered_step=} was none")
      metrics_to_write = _buffered_metrics
      steps_to_write = _buffered_step
  else:
    metrics_to_write = metrics
    steps_to_write = step

  if metrics_to_write:
    write_metrics_to_tensorboard(writer, metrics_to_write, steps_to_write, config, is_training)

    if config.metrics_file:
      max_utils.write_metrics_locally(metrics_to_write, steps_to_write, config, local_metrics_file, is_training)

    if config.gcs_metrics and jax.process_index() == 0:
      running_gcs_metrics = max_utils.write_metrics_for_gcs(
          metrics_to_write, steps_to_write, config, running_gcs_metrics, is_training
      )

  if is_training:
    _buffered_step = step
    _buffered_metrics = metrics


def write_metrics_to_tensorboard(writer, metrics, step, config, is_training=True):
  """Writes metrics to tensorboard"""
  with jax.spmd_mode("allow_all"):
    if jax.process_index() == 0:
      for metric_name in metrics.get("scalar", []):
        writer.add_scalar(metric_name, np.array(metrics["scalar"][metric_name]), step)
      for metric_name in metrics.get("scalars", []):
        writer.add_scalars(metric_name, metrics["scalars"][metric_name], step)

    if is_training:
      full_log = step % config.log_period == 0

      max_logging.log(
          f"completed step: {step}, seconds: {metrics['scalar']['perf/step_time_seconds']:.3f}, "
          f"TFLOP/s/device: {metrics['scalar']['perf/per_device_tflops_per_sec']:.3f}, "
          f"Tokens/s/device: {metrics['scalar']['perf/per_device_tokens_per_sec']:.3f}, "
          f"total_weights: {metrics['scalar']['learning/total_weights']}, "
          f"loss: {metrics['scalar']['learning/loss']:.3f}"
      )

      if full_log and jax.process_index() == 0:
        max_logging.log(f"To see full metrics 'tensorboard --logdir={config.tensorboard_dir}'")
        writer.flush()


def clear_buffered_metrics():
  global _buffered_step
  global _buffered_metrics
  _buffered_step = None
  _buffered_metrics = None


def save_checkpoint(
    checkpoint_manager,
    step,
    state,
    dataset_type="c4",
    data_iterator=None,
    config: Optional[pyconfig.config] = None,
    force: bool = False,
) -> bool:
  """Wrapper for saving checkpoint."""
  if config and config.enable_checkpointing:
    if (step % config.checkpoint_period == 0) or (
        config.enable_emergency_checkpoint and step % config.local_checkpoint_period == 0
    ):
      blocking_until_ready_start = time.time()
      max_logging.log(f"Waiting for step {step} to finish before checkpoint...")
      # We block here on the step finishing so that our checkpointing metrics
      # measure only checkpointing time, not training time.
      jax.block_until_ready(state)
      max_logging.log(
          f"Waited {time.time() - blocking_until_ready_start} seconds for step "
          f"{step} to finish before starting checkpointing."
      )

  # specify chunk_byte_size to force orbax to control maximum file size in checkpoint
  chunk_byte_size = _DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE
  if config:
    chunk_byte_size = config.checkpoint_storage_target_data_file_size_bytes
  save_args = jax.tree.map(lambda _: orbax.checkpoint.SaveArgs(chunk_byte_size=chunk_byte_size), state)

  if isinstance(checkpoint_manager, emergency_checkpoint_manager.CheckpointManager):
    return checkpoint_manager.save(
        step,
        args=orbax.checkpoint.args.PyTreeSave(item=state, save_args=save_args, ocdbt_target_data_file_size=chunk_byte_size),
        force=force,
    )

  if dataset_type == "grain":
    return checkpoint_manager.save(
        step,
        args=orbax.checkpoint.args.Composite(
            items=orbax.checkpoint.args.PyTreeSave(
                item=state, save_args=save_args, ocdbt_target_data_file_size=chunk_byte_size
            ),
            iter=grain.PyGrainCheckpointSave(data_iterator.local_iterator),
        ),
        force=force,
    )
  else:
    return checkpoint_manager.save(
        step,
        args=orbax.checkpoint.args.Composite(
            items=orbax.checkpoint.args.PyTreeSave(
                item=state, save_args=save_args, ocdbt_target_data_file_size=chunk_byte_size
            )
        ),
        force=force,
    )


# -----------------------------------------------------------------------------
# Top-level Functions
# -----------------------------------------------------------------------------


def record_activation_metrics(output_metrics, intermediate_outputs, config):
  """Adds the activation metrics to the metrics dict"""

  if config.scan_layers:
    metrics_dict = intermediate_outputs["intermediates"]["decoder"]["decoder"]

    for layer_num in range(config.num_decoder_layers):
      output_metrics["scalar"][f"activ_fraction_zero/layer_{layer_num:03d}"] = metrics_dict["activation_fraction_zero"][0][
          layer_num
      ]
      output_metrics["scalar"][f"activ_mean/layer_{layer_num:03d}"] = metrics_dict["activation_mean"][0][layer_num]
      output_metrics["scalar"][f"activ_stdev/layer_{layer_num:03d}"] = metrics_dict["activation_stdev"][0][layer_num]
  else:
    for layer_num in range(config.num_decoder_layers):
      layer = intermediate_outputs["intermediates"]["decoder"][f"layers_{layer_num}"]
      output_metrics["scalar"][f"activ_fraction_zero/layer_{layer_num:03d}"] = layer["activation_fraction_zero"][0]
      output_metrics["scalar"][f"activ_mean/layer_{layer_num:03d}"] = layer["activation_mean"][0]
      output_metrics["scalar"][f"activ_stdev/layer_{layer_num:03d}"] = layer["activation_stdev"][0]


def loss_fn(model, config, data, dropout_rng, params, is_train=True):
  """loss_fn for both train and eval.

  Args:
    model: A nn.Module
    config: Config of parameters
    data: Batch of data to apply to the model
    dropout_rng: A key to use to generate rng for dropout
    params: Model params
    is_train: True for train_step and False for eval_step

  Returns:
    loss: average loss
    aux: a dictionary including intermediate_outputs, total_loss, and total_weights
  """
  # inputs, targets, segments, positions = apply_args
  rng1, aqt_rng = jax.random.split(dropout_rng)

  # decimate proportion of data when per_device_batch_size<1
  if is_train:
    for k, v in data.items():
      data[k] = v[: config.micro_batch_size_to_train_on, :]
  else:
    for k, v in data.items():
      data[k] = v[: config.micro_batch_size_to_eval_on, :]

  logits, intermediate_outputs = model.apply(
      params,
      data["inputs"],
      data["inputs_position"],
      decoder_segment_ids=data["inputs_segmentation"],
      enable_dropout=config.enable_dropout if is_train else False,
      rngs={"dropout": rng1, "params": aqt_rng},
      mutable="intermediates",
  )
  one_hot_targets = jax.nn.one_hot(data["targets"], config.vocab_size)
  xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot_targets, 0.0)
  xent = nn.with_logical_constraint(xent, ("activation_embed_and_logits_batch", "activation_length"))
  # Mask out paddings at the end of each example.
  xent = xent * (data["targets_segmentation"] != 0)
  total_loss = jnp.sum(xent)
  total_weights = jnp.sum(data["targets_segmentation"] != 0)
  loss = total_loss / (total_weights + EPS)
  # get moe load balance loss
  moe_lb_loss = 0.0
  if config.num_experts > 1:
    nested_key = ("intermediates", "decoder", "layers", "moe_lb_loss")
    total_moe_lb_loss = maxtext_utils.get_nested_value(intermediate_outputs, nested_key, 0.0)
    moe_lb_loss = jnp.mean(jnp.array(total_moe_lb_loss))
    loss += moe_lb_loss
  aux = {
      "intermediate_outputs": intermediate_outputs,
      "total_loss": total_loss,
      "total_weights": total_weights,
      "moe_lb_loss": moe_lb_loss,
  }
  return loss, aux


def calculate_per_step_sparsity(num_pruning_iterations, target_sparsity):
  # Calculate the per-step sparsity required to reach a given 
  # target sparsity after a number of iterations
  per_step_sparsity = 1. - (1. - target_sparsity) ** (1. / num_pruning_iterations)
  return per_step_sparsity


def prune(state, percent_to_keep, structure, do_not_prune_kw, scoring_method):
  """Prune the model parameters to a given sparsity level."""
  assert structure in ["none", "2:4"], (
      "Unsupported pruning structure: %s" % structure
  )
  assert scoring_method in ["magnitude", "random"], (
      "Unsupported pruning scoring method: %s" % scoring_method
  )

  def collect_importance_score(params, structure, scoring_method, do_not_prune_kw, scope):
    """Collect importance score of all prunable weights for determining

    the global importance score cutoff for pruning.

    Prunable weights are those that satisfy the following conditions:
    1. The weight is a kernel (i.e. has 'kernel' in its name)
    2. The weight is not masked out.
    3. In the event of 2:4 structured pruning, the weight is not among the
       top 2 weights magnitude-wise in each group of consecutive 4 weights.
    """

    all_scores = []
    num_remaining = 0
    param_items = list(params.items())
    for name, param in param_items:
      full_name = f"{scope}.{name}"
      if isinstance(param, dict):
        child_importance, child_num_remaining, child_params = collect_importance_score(
            param, structure, scoring_method, do_not_prune_kw, full_name
        )
        all_scores.extend(child_importance)
        num_remaining += child_num_remaining
        params[name] = child_params
      elif name.endswith("kernel"):
        # If do_not_prune_kw prohibits pruning, skip to next param.
        do_not_prune = (
            True if do_not_prune_kw and do_not_prune_kw in full_name else False
        )
        if do_not_prune:
          continue
        mask_name = name.replace("kernel", "mask")
        mask = params[mask_name]
        if structure == "2:4":
          assert scoring_method == "magnitude", (
              "Unsupported scoring method for 2:4 structured pruning: %s"
              % scoring_method
          )
          prunable_weights = jnp.multiply(mask, param)
          prunable_weights = jnp.reshape(
              prunable_weights,
              (prunable_weights.size // 4, 4),
          )
          # Sort the weights in each group by magnitude and zero out the top 2
          # because they cannot be pruned.
          prunable_weights = jnp.sort(jnp.abs(prunable_weights), axis=-1)
          prunable_weights = prunable_weights[:, 0:2]
          all_scores.append(prunable_weights.flatten())
          num_remaining += jnp.sum(mask)
        elif structure == "none":
          if scoring_method == "magnitude":
            # Only take the weights that are not masked out
            all_scores.append(
                jnp.multiply(mask.flatten(), param.flatten())
            )
          elif scoring_method == "random":
            # Assign random float from 0.1 to 1 to unmasked positions as their
            # importance score. Masked positions will stay masked as they
            # have a uniform importance score of 0.
            mask = jnp.multiply(
                mask,
                jax.random.uniform(
                    jax.random.PRNGKey(0),
                    shape=mask.shape,
                    minval=0.1,
                    maxval=1.0,
                ),
            )
            params[mask_name] = mask
            all_scores.append(mask.flatten())
          num_remaining += jnp.sum(mask)
        else:
          raise ValueError("Unsupported pruning structure: %s" % structure)
    return all_scores, num_remaining, params

  def apply_global_threshold(params, threshold, structure, scoring_method, do_not_prune_kw, scope):
    """Apply the global threshold to each kernel tensor and store the mask."""
    new_params = {}
    for name, param in params.items():
      full_name = f"{scope}.{name}"
      if isinstance(param, dict):
        new_params[name] = apply_global_threshold(
            param, threshold, structure, scoring_method, do_not_prune_kw, full_name
        )
      else:
        do_not_prune = (
            True if do_not_prune_kw and do_not_prune_kw in full_name else False
        )
        if name.endswith("kernel") and not do_not_prune:
          if structure == "2:4":
            assert scoring_method == "magnitude", (
                "Unsupported scoring method for 2:4 structured pruning: %s"
                % scoring_method
            )
            # Create a mask that is 1 for weights that are:
            #   - Above a global threshold OR
            #   - Above a local threshold (i.e. in the top 2 weights in each
            #     group of 4 weights)
            # and 0 otherwise.
            grouped_param = jnp.reshape(param, (param.size // 4, 4))
            local_threshold = jnp.sort(jnp.abs(grouped_param), axis=-1)[:, 1]
            local_threshold = jnp.expand_dims(local_threshold, axis=-1)
            # Keep weights if they are above either global or local threshold.
            combined_threshold = jnp.minimum(threshold, local_threshold)
            prune_mask = jnp.abs(grouped_param) <= combined_threshold
            grouped_mask = jnp.logical_not(prune_mask)
            # sparsity = jnp.mean(grouped_mask)
            # jax.debug.print("Sparsity {sparsity}", sparsity=sparsity)
            mask = jnp.reshape(grouped_mask, param.shape).astype(param.dtype)
            mask_name = name.replace("kernel", "mask")
            prev_mask = params[mask_name]
            new_params[mask_name] = prev_mask + mask - prev_mask
          elif structure == "none":
            mask_name = name.replace("kernel", "mask")
            prev_mask = params[mask_name]
            if scoring_method == "magnitude":
              # Create a mask w/ 1 for weights above a threshold and 0 otherwise
              mask = (jnp.abs(param) >= threshold).astype(param.dtype)
              new_params[mask_name] = jnp.abs(mask * prev_mask)
            elif scoring_method == "random":
              # Unmasked positions have random importance score, masked ones 
              # have zero scores. So update the mask to have 1s for weights 
              # whose random score is above a threshold, and 0 otherwise.
              new_params[mask_name] = (jnp.abs(prev_mask) >= threshold).astype(
                  param.dtype
              ) * prev_mask

        # Copy other parameters unchanged
        if not name.endswith("mask") or (name.endswith("mask") and do_not_prune):
          # Ensure we don't overwrite existing masks
          new_params[name] = param
    return new_params

  # Step 1: Collect all scores
  root_scope = "root"
  all_scores, _, params = collect_importance_score(
      dict(state.params), structure, scoring_method, do_not_prune_kw, root_scope
  )
  # Concatenate all collected scores into a single array
  all_scores_concat = jnp.concatenate(all_scores)

  # Step 2: Determine the global threshold
  if structure == "none":
    num_elements_total = all_scores_concat.size
    num_to_keep_remaining = int(percent_to_keep * num_elements_total)
    threshold_idx = num_elements_total - num_to_keep_remaining
  elif structure == "2:4":
    num_unprunable = all_scores_concat.size  # Half of the weights cannot go.
    num_elements_total = all_scores_concat.size + num_unprunable
    assert percent_to_keep > 0.499, (
        "percent_to_keep_remaining must be always >= 50% for 2:4 structured"
        " pruning"
    )
    # Use ceiling to ensure we always make progress.
    num_to_keep_remaining = math.ceil(
        percent_to_keep * num_elements_total
    )
    threshold_idx = num_elements_total - num_to_keep_remaining
    assert threshold_idx >= 0, "threshold_idx must be non-negative."
    # threshold_idx = threshold_idx - num_unprunable
    jax.debug.print(
        "threshold_idx = {threshold_idx}, ", threshold_idx=threshold_idx
    )
  else:
    raise ValueError("Unsupported pruning structure: %s" % structure)

  sorted_magnitude = jnp.sort(jnp.abs(all_scores_concat), stable=False)
  # Note that the following line does not work if there are 2^31 elements.
  # XLA does not support int64 indexing.
  global_threshold = sorted_magnitude[threshold_idx]

  # Step 3: Apply the global threshold to each kernel tensor
  root_scope = "root"
  new_params = apply_global_threshold(
      params, global_threshold, structure, scoring_method, do_not_prune_kw, root_scope
  )

  pruning_metrics = {
      "pruning_debug/threshold_idx": threshold_idx,
      "pruning_debug/global_threshold": global_threshold,
      "pruning_debug/percent_to_keep_remaining": percent_to_keep,
  }
  # Return the new state with updated parameters and masks
  return state.replace(params=new_params), pruning_metrics



def zero_mask_grad(grads):
  """Zeros out gradients for any tensor in the PyTree whose path ends with 'mask'."""

  def zero_if_mask(path, value):
    # Check if the path ends with 'mask'
    if path[-1].key.endswith("mask"):
      return jnp.zeros_like(value)
    else:
      return value

  # Use tree_map to apply zeroing conditionally based on the path
  new_grads = jax.tree_util.tree_map_with_path(zero_if_mask, grads, is_leaf=lambda x: isinstance(x, jnp.ndarray))
  return new_grads

def binarize_mask(params):
  """Zeros out gradients for any tensor in the PyTree whose path ends with 'mask'."""

  def binarize_if_mask(path, value):
    # Check if the path ends with 'mask'
    if path[-1].key.endswith("mask"):
      return jnp.where(value > 0, 1.0, 0.0)
    else:
      return value

  # Use tree_map to apply zeroing conditionally based on the path
  new_params = jax.tree_util.tree_map_with_path(
      binarize_if_mask, params, is_leaf=lambda x: isinstance(x, jnp.ndarray)
  )
  return new_params


def train_step(model, config, state, data, dropout_rng):
  """

  Args:
    model: A nn.Module
    state: A pytree of the current state of the model
    data: Batch of data to apply to the model
    dropout_rng: A key to use to generate rng for dropout

  Returns:
    new_state: Same format as state.
    metrics: Dictionary of model metrics such as loss, training rate, etc.
    rng2: A new rng key that can be used in future calls.

  """
  if config.gradient_accumulation_steps > 1:

    def accumulate_gradient(acc_grad_and_loss, data):
      grad_func = jax.value_and_grad(loss_fn, argnums=4, has_aux=True)
      (_, aux), cur_batch_gradient = grad_func(model, config, data, dropout_rng, state.params, is_train=True)
      acc_grad_and_loss["loss"] += aux["total_loss"]
      acc_grad_and_loss["moe_lb_loss"] += aux["moe_lb_loss"]
      acc_grad_and_loss["grad"] = jax.tree_util.tree_map(
          lambda x, y: x * aux["total_weights"] + y, cur_batch_gradient, acc_grad_and_loss["grad"]
      )
      acc_grad_and_loss["total_weights"] += aux["total_weights"]
      return acc_grad_and_loss, aux

    def reshape_to_microbatch_accumulations(batch_arr):
      """Reshape global batch to microbatches, assuming batch axis is leading."""
      microbatches = config.gradient_accumulation_steps
      microbatch_shape = (microbatches, batch_arr.shape[0] // microbatches) + batch_arr.shape[1:]
      return jnp.reshape(batch_arr, microbatch_shape)

    data = jax.tree_util.tree_map(reshape_to_microbatch_accumulations, data)
    init_grad = jax.tree_util.tree_map(jnp.zeros_like, state.params)
    init_grad_and_loss = {"loss": 0.0, "grad": init_grad, "total_weights": 0, "moe_lb_loss": 0.0}

    grad_and_loss, aux = jax.lax.scan(
        accumulate_gradient, init_grad_and_loss, data, length=config.gradient_accumulation_steps
    )
    loss = (
        grad_and_loss["loss"] / grad_and_loss["total_weights"]
        + grad_and_loss["moe_lb_loss"] / config.gradient_accumulation_steps
    )
    raw_grads = jax.tree_util.tree_map(lambda arr: arr / grad_and_loss["total_weights"], grad_and_loss["grad"])
    aux = jax.tree_map(lambda x: jnp.sum(x, axis=0), aux)
  else:
    grad_func = jax.value_and_grad(loss_fn, argnums=4, has_aux=True)
    (loss, aux), raw_grads = grad_func(model, config, data, dropout_rng, state.params, is_train=True)
  intermediate_outputs = aux["intermediate_outputs"]
  total_weights = aux["total_weights"]
  moe_lb_loss = aux["moe_lb_loss"]

  if config.gradient_clipping_threshold > 0:
    grads = maxtext_utils.apply_gradient_clipping(raw_grads, state, config.gradient_clipping_threshold)
  else:
    grads = raw_grads
  
  grads = zero_mask_grad(grads)
  new_state = state.apply_gradients(grads=grads)
  new_param = binarize_mask(new_state.params)
  new_state = new_state.replace(params=new_param)
  
  num_param = max_utils.count_parameters_pytree(new_state.params)
  num_prunable = max_utils.count_prunable_parameters_pytree(new_state.params)
  nnz_prunable = max_utils.count_nonzero_prunable_parameters_pytree(new_state.params)
  if config.do_not_prune_kw:
    num_prunable_kw = max_utils.count_prunable_kw_pytree(new_state.params, config.do_not_prune_kw)
    nnz_prunable_kw = max_utils.count_nonzero_kw_pytree(new_state.params, config.do_not_prune_kw)
    print(f"do_not_prune_kw = {config.do_not_prune_kw}, num_prunable_kw = {num_prunable_kw}, nnz_prunable_kw = {nnz_prunable_kw}")
  else:
    nnz_prunable_kw = 0
    num_prunable_kw = 0

  jax.debug.print("num_prunable = {num_prunable}, nnz_prunable = {nnz_prunable}", num_prunable=num_prunable, nnz_prunable=nnz_prunable)
  density = nnz_prunable / num_prunable
  jax.debug.print("density = {density}", density=density)

  metrics = {
      "scalar": {
          "learning/loss": loss,
          "learning/moe_lb_loss": moe_lb_loss,
          "learning/total_weights": total_weights,
          "learning/grad_norm": max_utils.l2norm_pytree(grads),
          "learning/raw_grad_norm": max_utils.l2norm_pytree(raw_grads),
          "learning/param_norm": max_utils.l2norm_pytree(new_state.params),
          "pruning/num_param": num_param,
          "pruning/prunable": num_prunable - num_prunable_kw,
          "pruning/nonzero_prunable": nnz_prunable - nnz_prunable_kw,
          "pruning/density":  (nnz_prunable - nnz_prunable_kw) / (num_prunable - num_prunable_kw),
          f"pruning/nnz_prunable_kw_{config.do_not_prune_kw}": nnz_prunable_kw,
          f"pruning/num_prunable_kw_{config.do_not_prune_kw}": num_prunable_kw,
      },
      "scalars": {},
  }

  if config.record_internal_nn_metrics:
    record_activation_metrics(metrics, intermediate_outputs, config)

  return new_state, metrics


def eval_step(model, config, state, data, dropout_rng):
  """eval_step no backprop and new state compared with train_step."""
  eval_loss_fn = functools.partial(loss_fn, model, config, data, dropout_rng, is_train=False)
  loss, aux = eval_loss_fn(state.params)
  total_loss = aux["total_loss"]
  total_weights = aux["total_weights"]
  moe_lb_loss = aux["moe_lb_loss"]
  metrics = {
      "scalar": {
          "evaluation/loss": loss,
          "evaluation/total_loss": total_loss,
          "evaluation/total_weights": total_weights,
          "evaluation/moe_lb_loss": moe_lb_loss,
      },
  }

  return metrics


def create_goodput_recorder(config):
  if config.enable_goodput_recording:
    logger_name = f"goodput_{config.run_name}"
    recorder = goodput.GoodputRecorder(config.run_name, logger_name, jax.process_index() == 0)
    return recorder
  return None


def record_goodput(
    recorder,
    config,
    record_func,
    *args,
):
  """Record data for Goodput and Badput computation."""
  if recorder and config.enable_goodput_recording:
    record_func(*args)


def check_example_batch(config, example_batch):
  if config.max_checkify:
    jittable_f = checkify.checkify(lambda x: checkify.check(jnp.any(x > -1), "Batch contains bad synthetic data!"))
    # Check if inputs in batch contains bad synthetic data.
    # pylint: disable=not-callable
    err, _ = jax.jit(jittable_f)(example_batch["inputs"][: config.global_batch_size_to_train_on, :])
    err.throw()


def setup_mesh_and_model(config):
  """Set up the mesh and the model for training

  Args:
    config

  Returns:
    init_rng: RNG key
    writer: Summary writer for tensorboard
    checkpoint_manager: Orbax checkpointer
    state_mesh_annotations: the mesh annotations for the train state
    model:
    mesh:
    learning_rate_schedule:
    tx:
  """

  init_rng = random.PRNGKey(config.init_weights_seed)
  writer = max_utils.initialize_summary_writer(config)

  # Mesh definition
  devices_array = max_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)

  # Model and Optimizer definition
  quant = quantizations.configure_quantization(config)
  model = Transformer(config, mesh, quant=quant)
  learning_rate_schedule = max_utils.create_learning_rate_schedule(config)
  tx = optimizers.get_optimizer(config, learning_rate_schedule)
  logger = checkpointing.setup_checkpoint_logger(config)
  if config.enable_emergency_checkpoint:
    abstract_state, _, _ = max_utils.get_abstract_state(model, tx, config, init_rng, mesh, is_training=True)
    checkpoint_manager = checkpointing.create_orbax_emergency_checkpoint_manager(
        config.local_checkpoint_directory,
        config.checkpoint_dir,
        mesh,
        abstract_state,
        config.local_checkpoint_period,
        config.checkpoint_period,
        logger,
    )
  else:
    # TODO(b/368121306): Remove this once zarr3 support is plumbed on the backend
    use_ocdbt = config.checkpoint_storage_use_ocdbt
    use_zarr3 = config.checkpoint_storage_use_zarr3
    if config.enable_single_controller:
      use_ocdbt, use_zarr3 = False, False
    checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
        config.checkpoint_dir,
        config.enable_checkpointing,
        config.async_checkpointing,
        config.checkpoint_period,
        config.dataset_type,
        logger,
        use_ocdbt,
        use_zarr3,
    )

  return init_rng, writer, checkpoint_manager, mesh, model, learning_rate_schedule, tx


def setup_train_loop(config):
  """Set up prerequisites for the training loop -
      checkpoint_manager, PRNG keys, Mesh, Model and optimizer.
      Set up data iterator and tokenizer, initialize the model.

  Args:
    config

  Returns:
    init_rng:
    writer: Summary writer for tensorboard
    checkpoint_manager: Orbax checkpointer
    state_mesh_annotations: the mesh annotations for the train state
    model:
    mesh:
    learning_rate_schedule:
    data_iterator:
    state: the initialized train state
  """
  recorder = create_goodput_recorder(config)
  record_goodput(recorder, config, recorder.record_tpu_init_start_time if recorder else None)
  init_rng, writer, checkpoint_manager, mesh, model, learning_rate_schedule, tx = setup_mesh_and_model(config)
  record_goodput(recorder, config, recorder.record_tpu_init_end_time if recorder else None)
  record_goodput(recorder, config, recorder.record_training_preparation_start_time if recorder else None)
  data_iterator, eval_data_iterator = create_data_iterator(config, mesh)

  state, state_mesh_annotations, data_iterator = max_utils.setup_training_state(
      model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager
  )

  if not config.using_pipeline_parallelism:
    # The vocab tensor(s) of shape [vocab, embed] (and transpose) are not sharded by stage
    maxtext_utils.assert_params_sufficiently_sharded(state.params, mesh, tolerance=0.02)
  record_goodput(recorder, config, recorder.record_training_preparation_end_time if recorder else None)
  return (
      init_rng,
      writer,
      checkpoint_manager,
      state_mesh_annotations,
      model,
      mesh,
      learning_rate_schedule,
      data_iterator,
      eval_data_iterator,
      state,
  )


def train_loop(config, state=None):
  """Main Training loop.
  Args:
    config:
    state:
    ckpt_path:
  Returns:
  """
  # Create a GoodputRecorder to log information
  recorder = create_goodput_recorder(config)
  record_goodput(recorder, config, recorder.record_job_start_time if recorder else None)

  (
      init_rng,
      writer,
      checkpoint_manager,
      state_mesh_annotations,
      model,
      mesh,
      learning_rate_schedule,
      data_iterator,
      eval_data_iterator,
      state,
  ) = setup_train_loop(config)
  # pylint: disable=line-too-long
  (
      functional_train,
      in_shard_train,
      out_shard_train,
      static_argnums_train,
      donate_argnums_train,
  ) = maxtext_utils.get_functional_train_with_signature(train_step, mesh, state_mesh_annotations, model, config)

  if eval_data_iterator:
    # pylint: disable=line-too-long
    (
        functional_eval,
        in_shard_eval,
        out_shard_eval,
        static_argnums_eval,
        donate_argnums_eval,
    ) = maxtext_utils.get_functional_eval_with_signature(eval_step, mesh, state_mesh_annotations, model, config)

  num_model_parameters = max_utils.calculate_num_params_from_pytree(state.params)
  max_logging.log(f"number parameters: {num_model_parameters/1e9:.3f} billion")
  per_device_tflops, _, _ = maxtext_utils.calculate_tflops_training_per_device(config)
  per_device_tokens = maxtext_utils.calculate_tokens_training_per_device(config)

  # Write train config params, num model params, and XLA flags to tensorboard
  max_utils.add_text_to_summary_writer("num_model_parameters", str(num_model_parameters), writer)
  max_utils.add_text_to_summary_writer("libtpu_init_args", os.environ["LIBTPU_INIT_ARGS"], writer)
  max_utils.add_config_to_summary_writer(config, writer)

  # Define the compilation of functional_train, either by loading the compiled version or wrapping a new one in a jit
  if config.compiled_trainstep_file != "":
    print("Loading the compiled function...", flush=True)
    # Need to pass train signature and state to determine i/o shapes of train_state for now.
    p_train_step = maxtext_utils.load_compiled(config, functional_train, state)
    # TODO: p_eval_step is not yet supported in load_compiled
    p_eval_step = None
    print("Loaded compiled function!", flush=True)
  else:
    p_train_step = jax.jit(
        functional_train,
        in_shardings=in_shard_train,
        out_shardings=out_shard_train,
        static_argnums=static_argnums_train,
        donate_argnums=donate_argnums_train,
    )

    if eval_data_iterator:
      p_eval_step = jax.jit(
          functional_eval,
          in_shardings=in_shard_eval,
          out_shardings=out_shard_eval,
          static_argnums=static_argnums_eval,
          donate_argnums=donate_argnums_eval,
      )
    else:
      p_eval_step = None

  local_metrics_file = open(config.metrics_file, "a", encoding="utf8") if config.metrics_file else None
  running_gcs_metrics = [] if config.gcs_metrics else None

  start_step = get_first_step(state)  # this is the start_step for training
  first_profiling_step = start_step + config.skip_first_n_steps_for_profiler
  if config.profiler != "" and first_profiling_step >= config.steps:
    raise ValueError("Profiling requested but initial profiling step set past training final step")
  last_profiling_step = np.clip(first_profiling_step + config.profiler_steps - 1, first_profiling_step, config.steps - 1)

  example_batch = None
  last_step_completion = datetime.datetime.now()
  prof = profiler.Profiler(config)
  for step in np.arange(start_step, config.steps):
    if step == first_profiling_step:
      if config.profile_cleanly:
        jax.block_until_ready(state)  # Block until previous state finishes to start profile cleanly
      prof.activate()

    with jax.profiler.StepTraceAnnotation("train", step_num=step):
      record_goodput(recorder, config, recorder.record_data_loading_start_time if recorder else None)
      example_batch = load_next_batch(data_iterator, example_batch, config)
      record_goodput(recorder, config, recorder.record_data_loading_end_time if recorder else None)
      check_example_batch(config, example_batch=example_batch)
      # pylint: disable=not-callable
      nextrng = jax.jit(jax.random.fold_in)(init_rng, step)
      record_goodput(recorder, config, recorder.record_step_start_time if recorder else None, step)

      pruning_metrics = {}
      with jax.spmd_mode("allow_all"):
        if config.pruning_method == "one-shot":
          # If in one-shot pruning mode, simply prune the model at step 0.
          if step == 0:
            raise NotImplementedError("One-shot pruning is not yet implemented.")
            state, _ = prune(state, config.sparsity)
        elif config.pruning_method == "global_gradual":
          # If in GMP pruning mode, prune the model at a fixed frequency.
          if (step >= config.gmp_start_step and
              step <= config.gmp_end_step and
              step % config.gmp_pruning_frequency == 0):
            assert config.gmp_start_step <= config.gmp_end_step
            assert (config.gmp_end_step - config.gmp_start_step) % config.gmp_pruning_frequency == 0
            num_pruning_iterations = (config.gmp_end_step - config.gmp_start_step) // config.gmp_pruning_frequency + 1
            per_step_sparsity = calculate_per_step_sparsity(num_pruning_iterations, config.sparsity)
            current_pruning_iterations = (step - config.gmp_start_step) // config.gmp_pruning_frequency
            percent_to_keep_remaining = (1. - per_step_sparsity) ** (current_pruning_iterations + 1)
            print(f"Pruning at step {step} with percent_to_keep_remaining {percent_to_keep_remaining}")
            state, pruning_metrics = prune(state, percent_to_keep_remaining, config.pruning_structure, config.do_not_prune_kw, config.pruning_score)
        else:
          raise ValueError(f"Unsupported pruning method: {config.pruning_method}")

      with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        state, metrics = p_train_step(state, example_batch, nextrng)

    new_time = datetime.datetime.now()
    record_scalar_metrics(
        metrics, new_time - last_step_completion, per_device_tflops, learning_rate_schedule(step), per_device_tokens
    )
    last_step_completion = new_time

    if checkpoint_manager is not None:
      if save_checkpoint(checkpoint_manager, int(step), state, config.dataset_type, data_iterator, config):
        max_logging.log(f"saved a checkpoint at step {step}")

      # Upon preemption, exit when and only when all ongoing saves are complete.
      if checkpoint_manager.reached_preemption(step):
        checkpoint_manager.wait_until_finished()
        sys.exit()

    write_metrics(writer, local_metrics_file, running_gcs_metrics, metrics, step, config)

    if config.eval_interval > 0 and step > start_step and ((step + 1) % config.eval_interval == 0 or step == config.steps - 1):
      assert eval_data_iterator
      cumulative_eval_metrics = {
          "scalar": {
              "eval/total_loss": 0.0,
              "eval/total_weights": 0.0,
              "eval/avg_loss": 0.0,
              "eval/moe_lb_loss": 0.0,
          }
      }
      eval_step_count = 0
      # pylint: disable=not-callable
      for eval_batch in eval_data_iterator:
        if config.eval_steps > 0 and eval_step_count >= config.eval_steps:
          break
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
          eval_metrics = p_eval_step(state, eval_batch, nextrng)
        cumulative_eval_metrics["scalar"]["eval/total_loss"] += float(eval_metrics["scalar"]["evaluation/total_loss"])
        cumulative_eval_metrics["scalar"]["eval/total_weights"] += float(eval_metrics["scalar"]["evaluation/total_weights"])
        cumulative_eval_metrics["scalar"]["eval/moe_lb_loss"] += float(eval_metrics["scalar"]["evaluation/moe_lb_loss"])
        max_logging.log(f"Completed eval step {eval_step_count}")
        eval_step_count += 1
      eval_loss = (
          cumulative_eval_metrics["scalar"]["eval/total_loss"]
          / (cumulative_eval_metrics["scalar"]["eval/total_weights"] + EPS)
          + cumulative_eval_metrics["scalar"]["eval/moe_lb_loss"] / eval_step_count
      )
      cumulative_eval_metrics["scalar"]["eval/avg_loss"] = eval_loss
      write_metrics(
          writer, local_metrics_file, running_gcs_metrics, cumulative_eval_metrics, step, config, is_training=False
      )
      max_logging.log(
          f"average loss after {step=}: {eval_step_count=}, {eval_loss=}, total_weights={cumulative_eval_metrics['scalar']['eval/total_weights']}"
      )
      if eval_loss <= config.target_eval_loss:
        max_logging.log(f"Early stop and exit loop after reaching {config.target_eval_loss=}")
        prof.deactivate()
        break

    if step == last_profiling_step:
      if config.profile_cleanly:
        jax.block_until_ready(state)  # Block until current state finishes to end profile cleanly
      prof.deactivate()

  if checkpoint_manager is not None:
    if save_checkpoint(checkpoint_manager, int(config.steps), state, config.dataset_type, data_iterator, force=True):
      max_logging.log(f"saved a final checkpoint at step {config.steps}")
    checkpoint_manager.wait_until_finished()
  write_metrics(writer, local_metrics_file, running_gcs_metrics, metrics, config.steps - 1, config)  # final step metrics
  max_utils.close_summary_writer(writer)
  record_goodput(recorder, config, recorder.record_job_end_time if recorder else None)
  clear_buffered_metrics()
  return state


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
  pyconfig.initialize(argv)
  max_utils.print_system_information()
  config = pyconfig.config
  validate_train_config(config)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  vertex_tensorboard_manager = VertexTensorboardManager()
  if config.use_vertex_tensorboard or os.environ.get("UPLOAD_DATA_TO_TENSORBOARD"):
    vertex_tensorboard_manager.configure_vertex_tensorboard(config)

  if config.monitor_goodput and jax.process_index() == 0:
    logger_name = f"goodput_{config.run_name}"
    goodput_monitor = monitoring.GoodputMonitor(
        job_name=config.run_name,
        logger_name=logger_name,
        tensorboard_dir=config.tensorboard_dir,
        upload_interval=config.goodput_upload_interval_seconds,
        monitoring_enabled=True,
        include_badput_breakdown=True,
    )
    goodput_monitor.start_goodput_uploader()
    max_logging.log("Started Goodput upload to Tensorboard in the background!")
  debug_config = debug_configuration.DebugConfig(
      stack_trace_config=stack_trace_configuration.StackTraceConfig(
          collect_stack_trace=config.collect_stack_trace,
          stack_trace_to_cloud=config.stack_trace_to_cloud,
          stack_trace_interval_seconds=config.stack_trace_interval_seconds,
      )
  )
  diagnostic_config = diagnostic_configuration.DiagnosticConfig(debug_config)
  with diagnostic.diagnose(diagnostic_config):
    train_loop(config)


if __name__ == "__main__":
  app.run(main)
