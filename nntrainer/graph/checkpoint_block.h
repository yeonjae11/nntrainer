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

#include <string>
#include <vector>

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
   * @param layers Vector of layer names to include in this block
   * @param id Unique identifier for this block
   */
  CheckpointBlock(const std::vector<std::string> &layers,
                  const std::string &id = "");

  /**
   * @brief Get the layer names in this block
   *
   * @return const std::vector<std::string>& Layer names
   */
  const std::vector<std::string> &getLayerNames() const { return layer_names; }

  /**
   * @brief Get the block ID
   *
   * @return std::string Block identifier
   */
  std::string getBlockId() const { return block_id; }

  /**
   * @brief Get the number of layers in this block
   *
   * @return size_t Number of layers
   */
  size_t size() const { return layer_names.size(); }

  /**
   * @brief Get the start layer name
   *
   * @return std::string Name of the first layer
   */
  std::string getFirstLayerName() const;

  /**
   * @brief Check if the block contains a specific layer
   *
   * @return bool true if the layer is in the block, false otherwise
   */
  bool hasLayer(const std::string &layer_name) const;

  /**
   * @brief Insert a new layer name after a specified layer
   */
  void insertAfter(const std::string &layer, const std::string &new_layer);

private:
  std::vector<std::string> layer_names; /**< Names of layers in this block */
  std::string block_id;                 /**< Unique identifier for this block */
};

} // namespace nntrainer

#endif // __CHECKPOINT_BLOCK_H__
