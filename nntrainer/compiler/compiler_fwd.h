// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file compiler_fwd.h
 * @date 09 October 2021
 * @brief NNTrainer graph compiler related forward declarations
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __COMPILER_FWD_H__
#define __COMPILER_FWD_H__

#include <memory>
#include <string>
#include <vector>

namespace nntrainer {
class LayerNode;
class NetworkGraph;

struct GCBlock {
  std::string block_name;
  std::vector<std::string> layer_names;
};

using GraphRepresentation = std::vector<std::shared_ptr<LayerNode>>;
using GCBlockRepresentation = std::vector<GCBlock>;
using ExecutableGraph = NetworkGraph;

} // namespace nntrainer

#endif // __COMPILER_FWD_H__
