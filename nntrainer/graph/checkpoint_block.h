// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   checkpoint_block.h
 * @date   23 Oct 2025
 * @brief  Checkpoint block for gradient checkpointing
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Yeonjae Kim <duswo1120@snu.ac.kr>
 * @bug    No known bugs except for NYI items
 */

#ifndef __CHECKPOINT_BLOCK_H__
#define __CHECKPOINT_BLOCK_H__

#include <memory>
#include <string>
#include <vector>

#include <layer_node.h>

namespace nntrainer {

/**
 * @class CheckpointBlock
 * @brief Represents a block of layers for gradient checkpointing
 *
 * @details Gradient checkpointing saves memory by not storing intermediate
 * activations during forward pass. Instead, they are recomputed during
 * backward pass when needed.
 */
class CheckpointBlock {
public:
  /**
   * @brief Construct a new Checkpoint Block object
   *
   * @param _block_name Unique identifier for this block
   * @param _layer_nodes Vector of layers to include in this block
   */
  CheckpointBlock(
    const std::string &_block_name,
    const std::vector<std::shared_ptr<LayerNode>> &_layer_nodes);

  /**
   * @brief Construct a new Checkpoint Block object
   *
   * @param _layer_nodes Vector of layers to include in this block
   */
  CheckpointBlock(
    const std::vector<std::shared_ptr<LayerNode>> &_layer_nodes) :
    CheckpointBlock("", _layer_nodes) {}

  /**
   * @brief Get the block name
   *
   * @return std::string Block identifier
   */
  std::string getBlockName() const { return block_name; }

  /**
   * @brief Get the number of layers in this block
   *
   * @return size_t Number of layers
   */
  size_t size() const { return layer_nodes.size(); }

private:
  std::string block_name;               /**< Unique identifier for this block */
  std::vector<std::shared_ptr<LayerNode>> layer_nodes; /**< layer nodes */
};

} // namespace nntrainer

#endif // __CHECKPOINT_BLOCK_H__
