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

#include <checkpoint_block.h>
#include <stdexcept>

namespace nntrainer {

CheckpointBlock::CheckpointBlock(const std::vector<std::string> &layers,
                                 const std::string &id) :
  layer_names(layers), block_id(id) {

  if (layers.empty())
    throw std::invalid_argument("CheckpointBlock: layer list cannot be empty");

  // Generate block ID if not provided
  if (block_id.empty())
    block_id = "checkpoint_block_" + layers.front() + "_to_" + layers.back();
}

std::string CheckpointBlock::getFirstLayerName() const {
  // @TODO: it may not be safe before realization
  if (layer_names.empty())
    throw std::runtime_error("CheckpointBlock: empty");
  return layer_names[0];
}

bool CheckpointBlock::hasLayer(const std::string &layer_name) const {
  for (const std::string &layer : layer_names)
    if (layer.compare(layer_name) == 0)
      return true;

  return false;
}

void CheckpointBlock::insertAfter(const std::string &layer,
                                  const std::string &new_layer) {
  for (size_t i = 0; i < layer_names.size(); ++i) {
    if (layer_names[i].compare(layer) == 0) {
      layer_names.insert(layer_names.begin() + i + 1, new_layer);
      return;
    }
  }
}

} // namespace nntrainer
