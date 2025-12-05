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
    block_name = "checkpoint_block_" + layer_nodes.front()->getName() +
                 "_to_" + layer_nodes.back()->getName();
}

} // namespace nntrainer
