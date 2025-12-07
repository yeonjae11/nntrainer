// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   checkpoint_block.cpp
 * @date   23 Oct 2025
 * @brief  Checkpoint block implementation
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Yeonjae Kim <duswo1120@snu.ac.kr>
 * @bug    No known bugs except for NYI items
 */

#include <stdexcept>

#include <checkpoint_block.h>

namespace nntrainer {

CheckpointBlock::CheckpointBlock(
  const std::string &_block_name,
  const std::vector<std::shared_ptr<LayerNode>> &_layer_nodes) :
  block_name(_block_name), layer_nodes(_layer_nodes) {
  if (layer_nodes.empty())
    throw std::invalid_argument("CheckpointBlock: layer list cannot be empty");

  // Generate block ID if not provided
  if (block_name.empty())
    block_name = "checkpoint_block_" + layer_nodes.front()->getName() + "_to_" +
                 layer_nodes.back()->getName();

  for (auto &lnode : layer_nodes) {
    lnode->setCheckpointBlockName(block_name);
    lnode->setCheckpointed();
  }
}

static bool isInputCheckpointLayer(
  const std::shared_ptr<LayerNode> &layer,
  const std::vector<std::shared_ptr<LayerNode>> &block_layers) {
  const std::vector<std::string> &inputs = layer->getInputConnections();
  for (const std::string &input : inputs) {
    for (const auto &block_layer : block_layers) {
      if (block_layer->getName() == input) {
        return false;
      }
    }
  }
  return true;
}

static bool isOutputCheckpointLayer(
  const std::shared_ptr<LayerNode> &layer,
  const std::vector<std::shared_ptr<LayerNode>> &block_layers) {
  const std::vector<std::string> &outputs = layer->getOutputConnections();
  for (const std::string &output : outputs) {
    for (const auto &block_layer : block_layers) {
      if (block_layer->getName() == output) {
        return false;
      }
    }
  }
  return true;
}

void CheckpointBlock::setSortedLayerNodes(
  std::vector<std::shared_ptr<LayerNode>> _sorted_layer_nodes) {
  sorted_layer_nodes = _sorted_layer_nodes;
  sorted_layer_nodes.back()->setLastCheckpointLayer();
  for (auto &block_layer : sorted_layer_nodes) {
    if (isInputCheckpointLayer(block_layer, sorted_layer_nodes)) {
      block_layer->setInputCheckpointLayer();
    }
    if (isOutputCheckpointLayer(block_layer, sorted_layer_nodes)) {
      block_layer->setOutputCheckpointLayer();
    }
  }
}

} // namespace nntrainer
