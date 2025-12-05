// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file realizer.h
 * @date 09 October 2021
 * @brief NNTrainer graph realizer which preprocess graph representation as a
 * lowering process of compile
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __REALIZER_H__
#define __REALIZER_H__

#include <memory>
#include <optional>
#include <vector>

#include <checkpoint_block.h>
#include <compiler_fwd.h>
#include <nntrainer_log.h>

namespace nntrainer {

/**
 * @brief Graph realizer class
 *
 */
class GraphRealizer {
public:
  /**
   * @brief Destroy the Graph Realizer object
   *
   */
  virtual ~GraphRealizer() {}

  /**
   * @brief graph realizer creates a new graph based on the reference
   * @todo consider void GraphRepresentation &
   */
  virtual GraphRepresentation realize(const GraphRepresentation &reference) = 0;

  /**
   * @brief graph realizer creates a new graph based on the reference and modify
   * gradient checkpoint blocks correctly
   */
  virtual GraphRepresentation
  realize(const GraphRepresentation &reference,
          std::optional<std::reference_wrapper<std::vector<CheckpointBlock>>>
            checkpoint_blocks) {
    ml_loge("Warning: graph realize without gradient checkpoint blocks");
    return realize(reference);            
  }
};

} // namespace nntrainer

#endif // __REALIZER_H__
