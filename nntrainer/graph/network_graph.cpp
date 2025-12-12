// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    network_graph.h
 * @date    19 Oct 2020
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This is Network Graph Class for Neural Network
 *
 * @todo    Support multi-input graph.
 */

#include <activation_layer.h>
#include <addition_layer.h>
#include <bn_layer.h>
#include <concat_layer.h>
#include <connection.h>
#include <cross_entropy_loss_layer.h>
#include <cross_entropy_sigmoid_loss_layer.h>
#include <cross_entropy_softmax_loss_layer.h>
#include <engine.h>
#include <flatten_layer.h>
#include <grucell.h>
#include <identity_layer.h>
#include <input_layer.h>
#include <layer_node.h>
#include <layer_normalization_layer.h>
#include <lstmcell.h>
#include <multiout_layer.h>
#include <network_graph.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <profiler.h>
#include <rnn.h>
#include <rnncell.h>
#include <split_layer.h>
#include <tensor_layer.h>
#include <time_dist.h>
#include <tracer.h>
#include <util_func.h>
#include <weight_layer.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

#include "graph_node.h"
#include "tensor.h"

#define LNODE(x) std::static_pointer_cast<LayerNode>(x)

namespace nntrainer {

namespace {

constexpr unsigned lifespanMask(TensorLifespan span) {
  return static_cast<unsigned>(span);
}

constexpr unsigned FORWARD_MASK =
  lifespanMask(TensorLifespan::FORWARD_FUNC_LIFESPAN);

constexpr unsigned FORWARD_RECOMPUTE_MASK =
  lifespanMask(TensorLifespan::FORWARD_RECOMPUTE_LIFESPAN);

constexpr TensorLifespan promoteToRecompute(TensorLifespan span) {
  auto mask = lifespanMask(span);
  if (mask & FORWARD_MASK) {
    mask &= ~FORWARD_MASK;
    mask |= FORWARD_RECOMPUTE_MASK;
    return static_cast<TensorLifespan>(mask);
  }
  return span;
}

std::string lifespanToString(TensorLifespan ls, 
                             const std::tuple<unsigned, unsigned, unsigned, unsigned, unsigned> &exec_order) {
  unsigned mask = static_cast<unsigned>(ls);
  std::vector<std::string> components;
  std::vector<unsigned> active_orders;
  
  auto [forward, recompute, calc_grad, calc_deriv, apply_grad] = exec_order;
  
  if (mask & lifespanMask(TensorLifespan::FORWARD_FUNC_LIFESPAN)) {
    components.push_back("FORWARD_FUNC");
    active_orders.push_back(forward);
  }
  if (mask & lifespanMask(TensorLifespan::CALC_DERIV_LIFESPAN)) {
    components.push_back("CALC_DERIV");
    active_orders.push_back(calc_deriv);
  }
  if (mask & lifespanMask(TensorLifespan::CALC_GRAD_LIFESPAN)) {
    components.push_back("CALC_GRAD");
    active_orders.push_back(calc_grad);
  }
  if (mask & lifespanMask(TensorLifespan::CALC_AGRAD_LIFESPAN)) {
    components.push_back("CALC_AGRAD");
    active_orders.push_back(apply_grad);
  }
  if (mask & lifespanMask(TensorLifespan::FORWARD_RECOMPUTE_LIFESPAN)) {
    components.push_back("FORWARD_RECOMPUTE");
    active_orders.push_back(recompute);
  }
  if (mask & lifespanMask(TensorLifespan::FORWARD_INFER_LIFESPAN)) {
    components.push_back("FORWARD_INFER");
    active_orders.push_back(forward);
  }
  
  if (components.empty())
    return "UNMANAGED";
  
  std::string result;
  for (size_t i = 0; i < components.size(); i++) {
    if (i > 0) result += "|";
    result += components[i];
  }
  
  // Calculate validity range
  unsigned validity_start = active_orders.empty() ? 0 : *std::min_element(active_orders.begin(), active_orders.end());
  unsigned validity_end = active_orders.empty() ? 0 : *std::max_element(active_orders.begin(), active_orders.end());
  
  return result + " (mask=" + std::to_string(mask) + 
         ", lifetime=[" + std::to_string(validity_start) + "," + std::to_string(validity_end) + "])";
}

} // namespace
int NetworkGraph::compile(const std::string &loss_type) {
  int status = ML_ERROR_NONE;

  status = isCompilable();
  NN_RETURN_STATUS();

  try {
    setOutputConnections();
  } catch (std::exception &e) {
    ml_loge("setting output layer failed, reason: %s", e.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  graph.realizeInputOutputNode();

  if (exec_mode != ExecutionMode::INFERENCE) {
    try {
      /// @todo realize loss beforehand
      status = addLossLayer(loss_type);
      NN_RETURN_STATUS();
    } catch (const std::exception &e) {
      ml_loge("%s", e.what());
      status = ML_ERROR_INVALID_PARAMETER;
      NN_RETURN_STATUS();
    }
  } else {
    if (!loss_type.empty()) {
      ml_loge("Warning : Loss type is given in inference mode. Ignoring loss "
              "type.");
    }
  }

  graph.topologicalSort();

  setExecutionOrder();
  forward_iter_end = (*(cend() - 1)).get();

  inPlaceOptimize();

  status = checkCompiledGraph();
  NN_RETURN_STATUS();

  compiled = true;

  return status;
}

void NetworkGraph::setExecutionOrder() {
  auto backward_order = graph.size();
  
  // Track which checkpoint blocks have been processed (same as backwarding logic)
  std::set<std::string> recomputed_blocks;
  
  // Map to store recompute orders for each layer
  std::map<std::string, unsigned int> layer_recompute_orders;
  
  // Simulate the backwarding flow to assign execution orders
  for (auto iter = getBackwardingBeginIter(); iter != getBackwardingEndIter();
       iter++) {
    auto &node = *iter;
    auto order_idx = getBackwardingEndIter() - iter - 1;
    auto forward_order = order_idx;
    
    // Recompute checkpoint block if needed (same logic as backwarding)
    if (node->isCheckpointed()) {
      const std::string &block_id = node->getCheckpointBlockId();
      
      // Only recompute each block once (when we encounter the first layer in backward order)
      if (recomputed_blocks.find(block_id) == recomputed_blocks.end()) {
        // Collect all layers in this checkpoint block (in forward order)
        std::vector<std::shared_ptr<LayerNode>> block_layers;
        for (auto fwd_iter = cbegin(); fwd_iter != cend(); ++fwd_iter) {
          auto &fwd_node = *fwd_iter;
          if (fwd_node->getCheckpointBlockId() == block_id && fwd_node->isCheckpointed()) {
            block_layers.push_back(fwd_node);
          }
        }
        
        // Assign recompute orders in forward order (same as recomputeCheckpointBlock)
        for (auto &layer : block_layers) {
          layer_recompute_orders[layer->getName()] = backward_order;
          backward_order++;
        }
    
        recomputed_blocks.insert(block_id);
      }
    }
    
    // Now assign backward execution orders
    auto calc_gradient_order = backward_order;
    if (node->getTrainable())
      backward_order++;
    auto calc_derivative_order = backward_order;
    if (node->getTrainable())
      backward_order++;
    auto apply_gradient_order = backward_order++;

    // Get recompute order (0 if not checkpointed)
    unsigned int recompute_order = 0;
    if (node->isCheckpointed()) {
      recompute_order = layer_recompute_orders[node->getName()];
    }

    node->setExecutionOrder({forward_order, recompute_order, calc_gradient_order,
                             calc_derivative_order, apply_gradient_order});
    
    printf("[EXEC-ORDER] Layer: %s, forward=%ld, recompute=%u, calc_grad=%u, calc_deriv=%u, apply_grad=%u\n",
           node->getName().c_str(), forward_order, recompute_order, calc_gradient_order,
           calc_derivative_order, apply_gradient_order);
  }

  /**
   * This sets max execution order temporarily till model is initialized.
   * This set max execution order is used to extend gradient exec orders for
   * clipping.
   */
  graph_exec_end = std::get<4>((*(cbegin()))->getExecutionOrder());
}

void NetworkGraph::addLayerNode(std::unique_ptr<Layer> layer) {
  graph.addNode(std::make_unique<LayerNode>(std::move(layer)));
}

int NetworkGraph::addLossLayer(const std::string &loss_type_) {
  for (unsigned int i = 0; i < graph.getNumOutputNodes(); ++i) {
    auto output_layer_node = LNODE(graph.getOutputNode(i));
    std::string loss_type = loss_type_;

    if (output_layer_node->requireLabel())
      continue;

    if (loss_type.empty())
      continue;

    auto second_to_last_layer_node = output_layer_node;
    bool is_cross_entropy_loss =
      istrequal(loss_type, CrossEntropyLossLayer::type);
    if (is_cross_entropy_loss) {
      auto type = output_layer_node->getType();

      if (type != ActivationLayer::type) {
        throw exception::not_supported(
          "Error: Cross Entropy need last layer to have softmax or sigmoid"
          "activation.");
      }

      switch (output_layer_node->getActivationType()) {
      case ActivationType::ACT_SIGMOID:
        loss_type = CrossEntropySigmoidLossLayer::type;
        break;
      case ActivationType::ACT_SOFTMAX:
        loss_type = CrossEntropySoftmaxLossLayer::type;
        break;
      default:
        throw exception::not_supported(
          "Error: Cross Entropy not supported without softmax or sigmoid.");
      }

      second_to_last_layer_node =
        LNODE(graph.getNode(output_layer_node->getInputConnectionName(0)));
    }

    std::shared_ptr<LayerNode> lnode = createLayerNode(loss_type);
    graph.ensureName(*lnode);

    if (second_to_last_layer_node->getDistribute()) {
      lnode->setProperty({"distribute=true"});
    }

    /// @todo remove this by add loss at realization
    second_to_last_layer_node->setOutputLayers({lnode->getName()});
    lnode->setProperty(
      {"input_layers=" + second_to_last_layer_node->getName()});

    if (is_cross_entropy_loss) {
      graph.replaceNode(output_layer_node, lnode);
    } else {
      graph.addNode(lnode, false);
    }
    graph.replaceOutputNode(i, lnode);
  }

  return ML_ERROR_NONE;
}

void NetworkGraph::setOutputConnections() {
  for (auto layer_iter = cbegin(); layer_iter != cend(); layer_iter++) {
    const auto &node = *layer_iter;
    for (auto i = 0u, num_inode = node->getNumInputConnections(); i < num_inode;
         ++i) {
      const auto &name = node->getInputConnectionName(i);
      const auto &idx = node->getInputConnectionIndex(i);

      auto node_setting_output = getLayerNode(name);
      node_setting_output->setOutputConnection(idx, node->getName(), i);
    }
  }
}

int NetworkGraph::isCompilable() {
  if (compiled) {
    ml_loge("Graph is already compiled");
    return ML_ERROR_NOT_SUPPORTED;
  }

  if (graph.empty()) {
    ml_loge("Graph is empty");
    return ML_ERROR_INVALID_PARAMETER;
  }

  return ML_ERROR_NONE;
}

int NetworkGraph::checkCompiledGraph() {
  /** Dimension of input layers must be known */
  for (auto iter = cbegin(); iter != cend(); iter++) {
    auto lnode = (*iter);
    if (lnode->getNumInputConnections() == 0) {
      if (!lnode->hasInputShapeProperty()) {
        ml_loge("Layer with no inbound connection need input_shape property");
        return ML_ERROR_INVALID_PARAMETER;
      }
    }
  }

  return ML_ERROR_NONE;
}

void NetworkGraph::markNodesForBackwarding() {
  /** accumulate all the nodes which must support backwarding */
  std::unordered_set<std::string> must_support_backwarding;
  if (exec_mode == ExecutionMode::INFERENCE) {
    for (auto iter = cbegin(); iter != cend(); iter++) {
      auto lnode = (*iter);
      lnode->needsCalcGradient(false);
      lnode->needsCalcDerivative(false);
    }
    return;
  }

  /**
   * if a node is trainable, then all the nodes ahead of it must support
   * backwarding operation
   */
  for (auto iter = cbegin(); iter != cend(); iter++) {
    auto lnode = (*iter);
    if (lnode->getTrainable() ||
        must_support_backwarding.find(lnode->getName()) !=
          must_support_backwarding.end()) {
      if (lnode->getTrainable()) {
        lnode->needsCalcGradient(true);
      }
#ifdef ENABLE_TEST
      if (lnode->supportBackwarding() && !optimize_memory) {
        lnode->needsCalcDerivative(true);
      }
#endif

      for (auto i = 0u, num_node = lnode->getNumOutputConnections();
           i < num_node; ++i) {
        auto conn = lnode->getOutputConnection(i);
        if (!conn) {
          continue;
        }

        must_support_backwarding.insert(conn->getName());
      }
    }
  }

  /** mark all the required nodes support backwarding */
  for (auto const &node_name : must_support_backwarding) {
    auto ln = LNODE(graph.getNode(node_name)).get();
    ln->needsCalcDerivative(true);
  }
}

void NetworkGraph::setBatchSize(unsigned int batch_size) {
  if (batch_size == this->batch_size)
    return;

  this->batch_size = batch_size;
  if (!input_list.empty() && getInputDimension()[0].batch() == batch_size)
    return;

  auto allocated = tensor_manager->isAllocated();

  if (allocated)
    deallocateTensors();

  for (auto iter = cbegin(); iter != cend(); iter++) {
    if ((*iter)->isFinalized()) {
      /// resize tensors spec
      /// @todo remove below, if custom tensor needs to change dimension
      /// according to the tensor, it must be done explicitly, or at least have
      /// a property to control the behavior
      const RunLayerContext &context = (*iter)->getRunContext();
      for (unsigned int idx = 0; idx < context.getNumTensors(); idx++) {
        auto const &ts = context.getTensor(idx);
        tensor_manager->setBatchSize(ts.getName(), ts.getDim().batch());
        if (context.tensorHasGradient(idx)) {
          auto const &ts_grad = context.getTensorGrad(idx);
          tensor_manager->setBatchSize(ts_grad.getName(),
                                       ts_grad.getDim().batch());
        }
      }
      /// override setting batch as per request
      (*iter)->setBatch(batch_size);
    }
  }
  /// resize input and output spec
  tensor_manager->setBatchSize(batch_size);

  if (allocated)
    allocateTensors(exec_mode);

  /** update input and label dimensions */
  for (unsigned int idx = 0; idx < input_list.size(); idx++)
    input_dims_[idx] = tensor_manager->getTensor(input_list[idx])->getDim();
  for (unsigned int idx = 0; idx < label_list.size(); idx++)
    label_dims_[idx] = tensor_manager->getTensor(label_list[idx])->getDim();
}

void NetworkGraph::resetInputDimension(std::vector<TensorDim> dims) {
  auto allocated = tensor_manager->isAllocated();

  if (allocated)
    deallocateTensors();

  for (auto iter = cbegin(); iter != cend(); iter++) {
    if ((*iter)->isFinalized()) {
      (*iter)->updateTensorsByInputDimensions(dims);
    }
  }

  if (allocated)
    allocateTensors(exec_mode);

  /** update input and label dimensions */
  for (unsigned int idx = 0; idx < input_list.size(); idx++)
    input_dims_[idx] = tensor_manager->getTensor(input_list[idx])->getDim();
  for (unsigned int idx = 0; idx < label_list.size(); idx++)
    label_dims_[idx] = tensor_manager->getTensor(label_list[idx])->getDim();
}

void NetworkGraph::applyGradients(
  LayerNode *node, const std::function<void(Weight &)> &apply_func) {
  if (!node->getTrainable())
    return;

  TRACE_MEMORY() << node->getName() + ": AG";
  TRACE_TIME() << node->getName() + ": AG";

  auto &rc = node->getRunContext();
  auto num_weight = rc.getNumWeights();
  for (unsigned i = 0; i < num_weight; ++i) {
    if (!rc.weightHasGradient(i)) {
      continue;
    }

    if (!rc.isGradientLastAccess(i)) {
      /// @note instead of checking the last access of the weight, checking
      /// if weights are dependent to others to minimize overhead.
      /// this logic assume that the source of the dependent weight must be
      /// prior to the dependent.
      continue;
    }

    if (rc.isGradientClipByGlobalNorm(i) || rc.isMixedPrecision(i)) {
      /**
       * @note the weights whose gradient are to be clipped by global norm will
       * be clipped at once at the end of iteration and applied then.
       * For those weights where mixed precision is uesed, their gradient
       * updates might be delayed until they confirm whether their loss scales
       * are appropeiate.
       */
      continue;
    }

    apply_func(rc.getWeightObject(i));
  }
}

sharedConstTensors NetworkGraph::forwarding(
  bool training,
  std::function<void(std::shared_ptr<LayerNode>, bool)> forwarding_op,
  std::function<bool(void *userdata)> stop_cb, void *userdata) {
  for (auto iter = cbegin(); iter != cend() && !stop_cb(userdata); iter++) {
    auto &ln = *iter;
    bool is_checkpointed = ln->isCheckpointed();
    
    if (is_checkpointed) {
      ln->getRunContext().setInitialForward(true);
      
      // Save inputs and weights BEFORE forwarding for verification
      // This matches the timing of recompute verification
      if (checkpoint_verification_enabled) {
        auto &rc = ln->getRunContext();
        
        // Save inputs (using initial_inputs at this point)
        std::vector<Tensor> inputs;
        for (unsigned int i = 0; i < ln->getNumInputs(); ++i) {
          inputs.push_back(ln->getInput(i));
        }
        saveForwardInputs(ln->getName(), inputs);
        
        // Save weights
        std::vector<Tensor> weights;
        for (unsigned int i = 0; i < rc.getNumWeights(); ++i) {
          weights.push_back(rc.getWeight(i));
        }
        saveForwardWeights(ln->getName(), weights);
      }
    }
    
    PROFILE_TIME_START(profile_keys.at(ln->getType()));
    forwarding_op(*iter, training);
    PROFILE_TIME_END(profile_keys.at(ln->getType()));
    
    // Tensor dump for debugging:
    // - For checkpointed layers: dump after initial forward (before setInitialForward(false))
    // - For non-checkpointed layers: dump after normal forward
    if (tensor_dump_enabled) {
      auto &rc = ln->getRunContext();
      
      // Dump inputs
      for (unsigned int i = 0; i < ln->getNumInputs(); ++i) {
        dumpTensor(ln->getName(), "input", i, ln->getInput(i));
      }
      
      // Dump outputs
      for (unsigned int i = 0; i < ln->getNumOutputs(); ++i) {
        dumpTensor(ln->getName(), "output", i, ln->getOutput(i));
      }
      
      // Dump weights
      for (unsigned int i = 0; i < rc.getNumWeights(); ++i) {
        dumpTensor(ln->getName(), "weight", i, rc.getWeight(i));
      }
      
      // Dump intermediate tensors
      for (unsigned int i = 0; i < rc.getNumTensors(); ++i) {
        dumpTensor(ln->getName(), "tensor", i, rc.getTensor(i));
      }
    }
    
    if (is_checkpointed) {
      // Save outputs and tensors AFTER forwarding for verification
      // IMPORTANT: Must save BEFORE setInitialForward(false) to capture initial_tensors
      if (checkpoint_verification_enabled) {
        auto &rc = ln->getRunContext();
        
        // Save outputs
        std::vector<Tensor> outputs;
        for (unsigned int i = 0; i < ln->getNumOutputs(); ++i) {
          outputs.push_back(ln->getOutput(i));
        }
        saveForwardOutputs(ln->getName(), outputs);
        
        // Save tensors (intermediate tensors) - must be done while is_initial_forward=true
        // to capture initial_tensors which were actually used during forward
        std::vector<Tensor> tensors;
        for (unsigned int i = 0; i < rc.getNumTensors(); ++i) {
          tensors.push_back(rc.getTensor(i));
        }
        saveForwardTensors(ln->getName(), tensors);
      }
      
      ln->getRunContext().setInitialForward(false);
    }
  }

  sharedConstTensors out;
  for (unsigned int i = 0; i < graph.getNumOutputNodes(); ++i) {
    auto const &output_layer_node = LNODE(graph.getOutputNode(i));
    for (unsigned int j = 0; j < output_layer_node->getNumOutputs(); ++j) {
      // @todo we should determine what type to return
      // out.push_back(MAKE_SHARED_TENSOR(
      //   output_layer_node->getOutput(j).clone(TensorDim::DataType::FP32)));
      out.push_back(MAKE_SHARED_TENSOR(output_layer_node->getOutput(j)));
    }
  }

  return out;
}

sharedConstTensors NetworkGraph::incremental_forwarding(
  unsigned int from, unsigned int to, bool training,
  std::function<void(std::shared_ptr<LayerNode>, bool)> forwarding_op,
  std::function<bool(void *userdata)> stop_cb, void *userdata) {
  for (auto iter = cbegin(); iter != cend() && !stop_cb(userdata); iter++) {
    auto &ln = *iter;
    if (ln->isCheckpointed()) {
      ln->getRunContext().setInitialForward(true);
    }
    PROFILE_TIME_START(profile_keys.at(ln->getType()));
    forwarding_op(*iter, training);
    PROFILE_TIME_END(profile_keys.at(ln->getType()));
    if (ln->isCheckpointed()) {
      ln->getRunContext().setInitialForward(false);
    }
  }

  sharedConstTensors out;
  for (unsigned int i = 0; i < graph.getNumOutputNodes(); ++i) {
    auto const &output_layer_node = LNODE(graph.getOutputNode(i));
    for (unsigned int j = 0; j < output_layer_node->getNumOutputs(); ++j) {
      out.push_back(MAKE_SHARED_TENSOR(output_layer_node->getOutput(j)));
    }
  }

  return out;
}

bool NetworkGraph::backwarding(
  int iteration,
  std::function<void(std::shared_ptr<LayerNode>, bool)> &forwarding_op,
  std::function<bool(std::shared_ptr<LayerNode>, int)> &backwarding_op,
  std::function<void(Weight &, int)> &lazy_apply_grad_op,
  std::function<bool(void *userdata)> stop_cb, void *userdata) {
  /**
   * last layer backwarding is run out of this loop
   */
  auto iter_begin = getBackwardingBeginIter();
  auto iter_end = getBackwardingEndIter();
  bool is_valid = true;

  /// there is no layer to train, so backwarding is essentially noop
  if (iter_begin == iter_end) {
    return true;
  }

  auto const &lptr_begin = (*iter_begin);
  // graph_const_reverse_iterator
  auto iter_ = iter_begin;

  if (lptr_begin->requireLabel() == false)
    throw std::runtime_error(
      "Error: last layer does not accept label, we can't train");

  // Track which checkpoint blocks have been recomputed
  std::set<std::string> recomputed_blocks;
  
  for (iter_ = iter_begin; iter_ != iter_end && !stop_cb(userdata); iter_++) {
    auto &ln = *iter_;
    
    // Recompute checkpoint block if needed
    if (ln->isCheckpointed()) {
      const std::string &block_id = ln->getCheckpointBlockId();
      
      // Only recompute each block once (when we encounter the first layer in backward order)
      if (recomputed_blocks.find(block_id) == recomputed_blocks.end()) {
        printf("Triggering recomputation for checkpoint block '%s' at layer '%s'\n",
                block_id.c_str(), ln->getName().c_str());
        recomputeCheckpointBlock(block_id);
        recomputed_blocks.insert(block_id);
      }
    }
    
    PROFILE_TIME_START(profile_keys.at(ln->getType()));
    is_valid = backwarding_op(ln, iteration);
    PROFILE_TIME_END(profile_keys.at(ln->getType()));

    // Dump gradients after backward pass
    if (tensor_dump_enabled) {
      auto &rc = ln->getRunContext();
      
      // Dump input/output used in backward (for debugging)
      for (unsigned int i = 0; i < ln->getNumInputs(); ++i) {
        try {
          const Tensor &in = ln->getInput(i);
          dumpTensor(ln->getName(), "backward_input", i, in);
        } catch (...) {}
      }
      for (unsigned int i = 0; i < ln->getNumOutputs(); ++i) {
        try {
          const Tensor &out = ln->getOutput(i);
          dumpTensor(ln->getName(), "backward_output", i, out);
        } catch (...) {}
      }
      
      // Dump outgoing derivative (gradient to previous layer)
      for (unsigned int i = 0; i < ln->getNumInputs(); ++i) {
        try {
          const Tensor &grad = rc.getOutgoingDerivative(i);
          dumpTensor(ln->getName(), "outgoing_grad", i, grad);
        } catch (...) {}
      }
      
      // Dump weight gradients
      for (unsigned int i = 0; i < rc.getNumWeights(); ++i) {
        try {
          if (rc.isGradientFirstAccess(i)) {
            const Tensor &wgrad = rc.getWeightGrad(i);
            dumpTensor(ln->getName(), "weight_grad", i, wgrad);
          }
        } catch (...) {}
      }
    }

    if (!is_valid) {
      break;
    }
  }

  if (!is_valid) {
    /** if has NaN
     * 1. reset the loss scale. : @todo Backoff_factor : default --> 0.5
     * 2. run forwarding from cur_iter to cend() && !stop_cb(userdata);
     * 3. return false --> run backwarding again;
     */
    float scale = (*iter_)->getRunContext().getLossScale();

    NNTR_THROW_IF(scale - 1.0f < 10e-6, std::invalid_argument)
      << "Loss Scale Factor is 1.0f";

    float s = scale > 1.5f ? scale * 0.5f : 1.0f;

    resetLossScale(s);

    auto f_iter = cbegin() + graph.getSortedNodeIdx((*iter_)->getName());

    for (auto iter = f_iter; iter != cend() && !stop_cb(userdata); iter++) {
      auto &ln = *iter;
      ln->reStoreData(true);
    }

    for (auto iter = f_iter; iter != cend() && !stop_cb(userdata); iter++) {
      auto &ln = *iter;
      PROFILE_TIME_START(profile_keys.at(ln->getType()));
      forwarding_op(*iter, true);
      PROFILE_TIME_END(profile_keys.at(ln->getType()));
    }

    return false;
  }

  /** perform clipping of the gradients by global norm if any */
  if (lazy_weights.empty())
    return true;

  if (is_clip_grad) {
    /** calculate the global norm */
    Tensor global_norm_t(
      TensorDim({1u, 1u, 1u, (unsigned int)lazy_weights.size()}));
    float *global_norm_data = global_norm_t.getData();

    for (unsigned int idx = 0; idx < lazy_weights.size(); idx++) {
      auto const &w = lazy_weights[idx];

      if (isMixedPrecision()) {
        Tensor scaled_grad =
          w->getGradientRef().clone(TensorDim::DataType::FP32);
        scaled_grad.divide_i(loss_scale);
        global_norm_data[idx] = scaled_grad.l2norm();
      } else {
        global_norm_data[idx] = w->getGradientNorm();
      }
    }
    float global_norm = global_norm_t.l2norm();
    /** apply the gradient with the above global norm */
    for (auto w : lazy_weights) {
      w->clipGradientByGlobalNorm(global_norm);
    }
  }
  /** apply the gradient with the above global norm */
  for (auto w : lazy_weights) {
    lazy_apply_grad_op(*w, iteration);
  }
  nan_count++;

  /** @todo : handle as property : growth_interval : default --> 2000 */
  if (nan_count > 2000) {
    float scale = (*iter_)->getRunContext().getLossScale();
    /** @todo growth_factor : default --> 2.0 */
    float s = scale * 2.0f;
    resetLossScale(s);
    nan_count = 0;
  }

  return true;
}

LayerNode *NetworkGraph::computeBackwardEnd() {
  int max_exec_order = -1;
  LayerNode *node = nullptr;

  if (!optimize_memory) {
    return (*cbegin()).get();
  }

  for (auto iter = getBackwardingBeginIter(); iter != getBackwardingEndIter();
       iter++) {
    auto &ln = *iter;
    const auto &exec_order = ln->getExecutionOrder();
    int cur_order = std::get<0>(exec_order);
    if (ln->needsCalcDerivative() || ln->needsCalcGradient()) {
#ifdef ENABLE_TEST
      cur_order = std::get<3>(exec_order);
#else
      cur_order = std::get<2>(exec_order);
#endif
    }

    NNTR_THROW_IF(max_exec_order == cur_order, std::invalid_argument)
      << "layer node: " << ln->getName()
      << " has duplicated max_exec_order, this should not happen, current "
         "execution order: "
      << max_exec_order;

    if (max_exec_order < cur_order) {
      max_exec_order = cur_order;
      node = ln.get();
    }
  }

  return node;
}

/**
 * @brief Allocate memory for all the managed tensors
 */
void NetworkGraph::allocateTensors(ExecutionMode exec_mode_) {
  exec_mode = exec_mode_;
  if (exec_mode == ExecutionMode::INFERENCE)
    /**
     * get the order of execution/usage order for the forwarding of the last
     * layer and pass that as the max_exec_order ensuring that all tensors
     * with usage less than the max_exec_order are allocated.
     */
    tensor_manager->allocateTensors(
      std::get<0>((*(cend() - 1))->getExecutionOrder()));
  else {
    /**
     * get the order of execution/usage order for the backwarding of the first
     * layer (as that will be the last layer to executed in the backwarding)
     * and pass that as the max_exec_order ensuring that all tensors with
     * usage less than the max_exec_order are allocated.
     * @todo if model is gradient clipping, we have to add last execution order
     * + 1
     */
    tensor_manager->allocateTensors(
      std::get<4>(backward_iter_end->getExecutionOrder()));
  }
}

std::vector<TensorDim> NetworkGraph::getInputDimension() const {
  NNTR_THROW_IF(input_dims_.empty(), std::invalid_argument)
    << "[NetworkGraph] the graph has no node identified as input!";
  return input_dims_;
}

unsigned int NetworkGraph::getBatchSize() const { return batch_size; }

std::vector<TensorDim> NetworkGraph::getOutputDimension() const {
  NNTR_THROW_IF(label_dims_.empty(), std::invalid_argument)
    << "[NetworkGraph] the graph has no node identified as output!";
  /// for now, outputting label_dims_ works, later label dim will be different
  /// from output dimension
  return label_dims_;
}

std::vector<std::shared_ptr<LayerNode>>
NetworkGraph::getUnsortedLayers(const std::string &input_layer,
                                const std::string &output_layer) const {
  /// @fixme: this won't work if input, output layers are not in order
  /// Further, this function must be removed. There should be rather
  /// getAllNames and getLayerByName instead of getUnsortedLayers.

  /** count layers after output layer */
  unsigned int num_layers_remove_end = 0;
  if (!output_layer.empty()) {
    for (auto iter = graph.crbegin(); iter != graph.crend(); iter++) {
      if ((*iter)->getName() != output_layer)
        num_layers_remove_end++;
      else
        break;
    }
  }

  if (num_layers_remove_end == graph.size())
    return {};

  /** count layers before input layer */
  unsigned int num_layers_remove_start = 0;
  if (!input_layer.empty()) {
    for (auto iter = graph.cbegin();
         iter != graph.cend() - num_layers_remove_end; iter++) {
      if ((*iter)->getName() != input_layer)
        num_layers_remove_start++;
      else
        break;
    }
  }

  /** copy the graph and return */
  std::vector<std::shared_ptr<LayerNode>> ret;
  std::transform(graph.cbegin() + num_layers_remove_start,
                 graph.cend() - num_layers_remove_end, std::back_inserter(ret),
                 [](auto const &elem) { return LNODE(elem); });

  return ret;
}

std::vector<std::shared_ptr<LayerNode>> NetworkGraph::getLayerNodes() const {
  return std::vector<std::shared_ptr<LayerNode>>(cbegin(), cend());
}

void NetworkGraph::addLayer(std::shared_ptr<LayerNode> layer) {
  if (compiled)
    throw std::runtime_error("Cannot modify graph after compile");

  /** Insert the layer to the graph */
  graph.addNode(layer);
}

InPlaceType
NetworkGraph::canExecuteInPlace(const std::shared_ptr<LayerNode> &lnode) {
  InPlaceType inplace_type = lnode->initializeInPlace();

  if (inplace_type == InPlaceType::NONE) {
    return inplace_type;
  }

  if (lnode->getType() == InputLayer::type &&
      !istrequal(getTensorType()[2], "FP32")) {
    return InPlaceType::NONE;
  }

  if (lnode->getType() == MultiOutLayer::type) {
    return InPlaceType::RESTRICTING;
  }

  /** A case where it can operate in-place even if there is a multi-out type
   * input connection. */
  if (inplace_type == InPlaceType::RESTRICTING) {
    for (size_t i = 0, num_node = lnode->getNumInputConnections(); i < num_node;
         ++i) {
      const std::string &input_name = lnode->getInputConnectionName(i);
      if (getLayerNode(input_name)->getInPlaceType() ==
          InPlaceType::RESTRICTING)
        return inplace_type;
    }
    return InPlaceType::NON_RESTRICTING;
  }
  /** A case where it cannot operate in-place if there is a multi-out type
   * input connection. */
  else {
    /** condition: NON_RESTRICTING */
    for (size_t i = 0, num_node = lnode->getNumInputConnections(); i < num_node;
         ++i) {
      const std::string &input_name = lnode->getInputConnectionName(i);
      if (getLayerNode(input_name)->getInPlaceType() ==
          InPlaceType::RESTRICTING)
        return InPlaceType::NONE;
    }
    return inplace_type;
  }
}

void NetworkGraph::inPlaceOptimize() {
  if (optimize_memory) {
    for (unsigned int idx = 0; idx < graph.size(); ++idx) {
      auto const &lnode = getSortedLayerNode(idx);
      lnode->setInPlaceType(canExecuteInPlace(lnode));
    }
  }
}

/**
 * @brief Set the Inplace Shared Memory Config By Layer object
 *
 * @param lnode layer node object
 * @param shared_var if the variable should be shared
 * @param shared_grad if the gradient should be shared
 */
static void
setInplaceSharedMemoryConfigByLayer(const std::shared_ptr<LayerNode> &lnode,
                                    bool &shared_var, bool &shared_grad) {
  /** for multiout layer, variables are shared but gradients are not */
  if (lnode->getType() == MultiOutLayer::type) {
    shared_var = true;
    shared_grad = false;
  } else {
    shared_var = true;
    shared_grad = true;
  }

  /**
   * @todo for layers which support in-place, both variables and gradients
   * will be shared.
   */
}

std::vector<Var_Grad *>
NetworkGraph::finalizeContext(const std::shared_ptr<LayerNode> &lnode,
                              const std::vector<Var_Grad *> &prev_inputs) {
  const GraphNode &gnode = *lnode.get();
  std::vector<TensorDim> input_dims;
  input_dims.reserve(prev_inputs.size());
  std::transform(prev_inputs.begin(), prev_inputs.end(),
                 std::back_inserter(input_dims),
                 [](const Var_Grad *vg) { return vg->getDim(); });

  /** finalize the layer and get the final context */
  auto init_context = lnode->finalize(input_dims, getTensorType(), exec_mode);
  
  // Gradient checkpointing: Track if this is a checkpointed layer
  bool is_checkpointed_layer = lnode->isCheckpointed();
  
  const auto &ct_engine = nntrainer::Engine::Global();

  /**
   * Request manager for either a pre-allocated output as input or a newly
   * allocated output. This is necessary for manager to know when this
   * output node is going to be used.
   */
  std::vector<std::string> input_names;
  input_names.reserve(prev_inputs.size());
  std::transform(
    prev_inputs.begin(), prev_inputs.end(), std::back_inserter(input_names),
    [](auto const &vg) -> const auto & { return vg->getName(); });
  const std::vector<Var_Grad *> &inputs = tensor_manager->requestInputs(
    gnode, init_context.getInputDimensions(), input_names);
  
  // Print input allocation info
  printf("[TENSOR-ALLOC] ===== Layer: %s - INPUTS =====\n", lnode->getName().c_str());
  for (size_t i = 0; i < inputs.size(); i++) {
    // Note: inputs are typically READ_ONLY_VIEW references to previous layer outputs
    // Their actual lifespan is determined by the source output tensor
    printf("[TENSOR-ALLOC] Layer: %s, Type: INPUT, Name: %s, Ptr: %p (reference to prev output)\n",
           lnode->getName().c_str(), 
           inputs[i]->getName().c_str(),
           (void*)inputs[i]);
  }

  /** In-Place optimizations */
  /**
   * Request manager for either a pre-allocated input as output or a newly
   * allocated output. This is necessary for manager to know when this
   * output node is going to be used with in-place optimizations.
   */
  auto out_specs = init_context.getOutSpecs();
  
  // Gradient checkpointing: Prepare for two sets of outputs
  std::vector<Var_Grad *> checkpoint_initial_outputs;
  std::vector<Var_Grad *> checkpoint_recompute_outputs;

  /// @note try move inplace control to finalize
  bool shared_var = false, shared_grad = false;

  if (lnode->getInPlaceType() != InPlaceType::NONE && lnode->supportInPlace()) {
    setInplaceSharedMemoryConfigByLayer(lnode, shared_var, shared_grad);

    for (unsigned int i = 0; i < out_specs.size(); ++i) {
      auto &s = out_specs.at(i);
      if (shared_var) {
        s.variable_spec.request_type =
          TensorSpecV2::RequestType::READ_ONLY_VIEW;
        if (lnode->getType() == IdentityLayer::type) {
          s.variable_spec.reference_name = inputs[i]->getName();
          s.variable_spec.dim.setFormat(inputs[i]->getDim().getFormat());
        } else if (lnode->getInPlaceDirection() == InPlaceDirection::RIGHT) {
          s.variable_spec.reference_name = inputs[1]->getName();
          s.variable_spec.dim.setFormat(inputs[1]->getDim().getFormat());
        } else if (lnode->getType() == WeightLayer::type) {
          WeightSpec w_spec = init_context.getWeightsSpec()[i];
          s.variable_spec.reference_name = std::get<8>(w_spec);
          s.variable_spec.dim.setFormat(std::get<0>(w_spec).getFormat());
        } else if (lnode->getType() == TensorLayer::type) {
          InitLayerContext::TensorSpec t_spec =
            init_context.getTensorsSpec()[i];
          s.variable_spec.reference_name = std::get<3>(t_spec);
          s.variable_spec.dim.setFormat(std::get<0>(t_spec).getFormat());
        } else {
          s.variable_spec.reference_name = inputs[0]->getName();
          s.variable_spec.dim.setFormat(inputs[0]->getDim().getFormat());
        }
      }
      if (shared_grad && s.gradient_spec) {
        s.gradient_spec->request_type =
          TensorSpecV2::RequestType::READ_ONLY_VIEW;
        if (lnode->getType() == IdentityLayer::type) {
          s.gradient_spec->reference_name = inputs[i]->getGradientName();
          s.gradient_spec->dim.setFormat(inputs[i]->getDim().getFormat());
        } else if (lnode->getInPlaceDirection() == InPlaceDirection::RIGHT) {
          s.gradient_spec->reference_name = inputs[1]->getGradientName();
          s.gradient_spec->dim.setFormat(inputs[1]->getDim().getFormat());
        } else if (lnode->getType() == WeightLayer::type) {
          WeightSpec w_spec = init_context.getWeightsSpec()[i];
          s.gradient_spec->reference_name =
            std::get<8>(w_spec) + Var_Grad::grad_suffix;
          s.gradient_spec->dim.setFormat(std::get<0>(w_spec).getFormat());
        } else if (lnode->getType() == TensorLayer::type) {
          InitLayerContext::TensorSpec t_spec =
            init_context.getTensorsSpec()[i];
          s.gradient_spec->reference_name =
            std::get<3>(t_spec) + Var_Grad::grad_suffix;
          s.gradient_spec->dim.setFormat(std::get<0>(t_spec).getFormat());
        } else {
          s.gradient_spec->reference_name = inputs[0]->getGradientName();
          s.gradient_spec->dim.setFormat(inputs[0]->getDim().getFormat());
        }
      }
    }
  }
  if (lnode->requireLabel()) {
    NNTR_THROW_IF(out_specs.size() != 1, std::invalid_argument)
      << "out specification size must be 1 for label layer for now, "
      << lnode->getName() << " out spec size: " << out_specs.size();
    NNTR_THROW_IF(out_specs[0].gradient_spec == nullptr, std::invalid_argument)
      << "label space does not exist for " << lnode->getName();
    out_specs[0].gradient_spec->request_type =
      TensorSpecV2::RequestType::PLACEHOLDER;
  }

  /// @note below needs to be enabled only for inference mode, but need
  /// decision if we are going to separate inference initialization from
  /// train initialization this might not worth optimize because in general
  /// output of a neuralnet is very small
  if (lnode->getOutputConnections().size() == 0u) {
    std::for_each(out_specs.begin(), out_specs.end(),
                  [this](VarGradSpecV2 &spec) {
                    spec.variable_spec.additional_exec_order.push_back(
                      std::get<0>(forward_iter_end->getExecutionOrder()));
                  });
  }

  if (lnode->getType() == RNNCellLayer::type or
      lnode->getType() == LSTMCellLayer::type or
      lnode->getType() == GRUCellLayer::type) {
    std::for_each(out_specs.begin(), out_specs.end(), [](VarGradSpecV2 &spec) {
      spec.variable_spec.ls = TensorLifespan::FORWARD_GRAD_LIFESPAN;
    });
  }

  // Gradient checkpointing: Check if this layer needs initial_outputs
  // A layer needs initial_outputs if:
  // 1. It is checkpointed, OR
  // 2. It receives input from a checkpointed layer (to propagate initial values)
  const bool is_checkpoint_layer = lnode->isCheckpointed();
  const bool is_first_checkpoint_layer =
    is_checkpoint_layer && lnode->isFirstInCheckpointBlock();
  const bool is_last_checkpoint_layer = 
    is_checkpoint_layer && lnode->isLastInCheckpointBlock();

  bool needs_initial_outputs = is_checkpointed_layer;
  if (!needs_initial_outputs) {
    // Check if any input comes from a checkpointed layer
    for (unsigned int i = 0; i < lnode->getNumInputConnections(); ++i) {
      auto input_conn_name = lnode->getInputConnectionName(i);
      if (!input_conn_name.empty()) {
        auto prev_layer = getLayerNode(input_conn_name);
        if (prev_layer && prev_layer->isCheckpointed()) {
          needs_initial_outputs = true;
          printf("[CHECKPOINT] Layer '%s' (non-checkpointed) needs initial_outputs because it receives from checkpointed layer '%s'\n",
                  lnode->getName().c_str(), prev_layer->getName().c_str());
          break;
        }
      }
    }

    // If any of the consumers are checkpointed, we also need initial outputs
    if (!needs_initial_outputs) {
      for (unsigned int i = 0; i < lnode->getNumOutputConnections(); ++i) {
        auto conn = lnode->getOutputConnection(i);
        if (!conn)
          continue;
        auto sink_node = getLayerNode(conn->getName());
        if (sink_node && sink_node->isCheckpointed()) {
          needs_initial_outputs = true;
          printf("[CHECKPOINT] Layer '%s' (feeding checkpointed layer '%s') needs initial_outputs\n",
                 lnode->getName().c_str(), sink_node->getName().c_str());
          break;
        }
      }
    }
  }

  // CRITICAL: If this layer feeds a FIRST checkpoint layer, its normal output
  // must be kept alive until recompute. The first checkpoint layer uses
  // saved_inputs (normal inputs) instead of initial_outputs, so the producer's
  // normal output must persist until recompute time.
  bool feeds_first_checkpoint_layer = false;
  unsigned int first_checkpoint_recompute_order = 0;
  for (unsigned int i = 0; i < lnode->getNumOutputConnections(); ++i) {
    auto conn = lnode->getOutputConnection(i);
    if (!conn)
      continue;
    auto sink_node = getLayerNode(conn->getName());
    if (sink_node && sink_node->isCheckpointed() && sink_node->isFirstInCheckpointBlock()) {
      feeds_first_checkpoint_layer = true;
      first_checkpoint_recompute_order = std::get<1>(sink_node->getExecutionOrder());
      printf("[CHECKPOINT] Layer '%s' feeds FIRST checkpoint layer '%s' - extending normal output lifespan to recompute order %u\n",
             lnode->getName().c_str(), sink_node->getName().c_str(), first_checkpoint_recompute_order);
      break;
    }
  }

  if (is_checkpoint_layer) {
    std::for_each(out_specs.begin(), out_specs.end(),
                  [](VarGradSpecV2 &spec) {
                    spec.variable_spec.ls =
                      promoteToRecompute(spec.variable_spec.ls);
                  });
  }

  // Extend normal output lifespan if feeding first checkpoint layer
  if (feeds_first_checkpoint_layer && !is_checkpoint_layer) {
    for (auto &spec : out_specs) {
      // Add recompute order to keep output alive until recompute
      spec.variable_spec.additional_exec_order.push_back(first_checkpoint_recompute_order);
      printf("[CHECKPOINT] Layer '%s' output '%s' - added recompute order %u to additional_exec_order\n",
             lnode->getName().c_str(), spec.variable_spec.name.c_str(), first_checkpoint_recompute_order);
    }
  }

  // Request outputs (default: for recompute & backward)
  const std::vector<Var_Grad *> &outputs = tensor_manager->requestTensors(
    out_specs, Manager::TensorGroupType::OUTPUT, lnode->getExecutionOrder(),
    lnode->getName());
  
  // Print tensor allocation info
  auto exec_order = lnode->getExecutionOrder();
  printf("[TENSOR-ALLOC] ===== Layer: %s - OUTPUTS (recompute) =====\n", lnode->getName().c_str());
  for (size_t i = 0; i < out_specs.size(); i++) {
    printf("[TENSOR-ALLOC] Layer: %s, Type: OUTPUT, Name: %s, Ptr: %p, Lifespan: %s\n",
           lnode->getName().c_str(), out_specs[i].variable_spec.name.c_str(),
           i < outputs.size() ? (void*)outputs[i] : nullptr,
           lifespanToString(out_specs[i].variable_spec.ls, exec_order).c_str());
  }

  // Request initial forward outputs first (before modifying out_specs)
  std::vector<Var_Grad *> initial_outputs;
  if (needs_initial_outputs) {
    printf("[CHECKPOINT] Layer '%s' (checkpointed=%d) - allocating initial outputs\n", 
            lnode->getName().c_str(), is_checkpointed_layer);
    if (lnode->isCheckpointed() && is_last_checkpoint_layer) {
      initial_outputs = outputs;
      printf("[CHECKPOINT] Layer '%s' is last in checkpoint block - sharing recompute outputs as initial outputs\n",
             lnode->getName().c_str());
    } else {
      auto initial_out_specs = out_specs;
      const unsigned int terminal_forward_order =
        std::get<0>(forward_iter_end->getExecutionOrder());

      auto append_consumer_order = [](VarGradSpecV2 &spec,
                                      unsigned int order) {
        auto &orders = spec.variable_spec.additional_exec_order;
        if (std::find(orders.begin(), orders.end(), order) ==
            orders.end()) {
          orders.push_back(order);
        }
      };
      
      // Allocate initial_outputs for all layers
      // Note: For in-place layers like MultiOut, layer_context.cpp will automatically
      // return initial_inputs instead of initial_outputs during getOutput()
      for (size_t spec_idx = 0; spec_idx < initial_out_specs.size();
           ++spec_idx) {
        auto &spec = initial_out_specs[spec_idx];
        // Create new name instead of modifying in place
        spec.variable_spec.name = spec.variable_spec.name + "_initial";
        spec.variable_spec.ls = TensorLifespan::FORWARD_FUNC_LIFESPAN;

        // Initial forward doesn't need gradients - only variables
        spec.gradient_spec = nullptr;

        bool consumer_added = false;
        if (spec_idx < lnode->getNumOutputConnections()) {
          auto conn = lnode->getOutputConnection(spec_idx);
          if (conn) {
            auto sink_node = getLayerNode(conn->getName());
            if (sink_node) {
              // For in-place layers, we need to follow the chain to find all
              // actual consumers, because in-place layers share memory with input
              std::vector<std::shared_ptr<LayerNode>> consumers_to_process;
              consumers_to_process.push_back(sink_node);
              
              while (!consumers_to_process.empty()) {
                auto current = consumers_to_process.back();
                consumers_to_process.pop_back();
                
                // Use actual in-place type instead of hardcoded layer types
                // This correctly handles cases like GELU (not in-place) vs ReLU (in-place)
                bool is_inplace = (current->getInPlaceType() != InPlaceType::NONE);
                
                if (is_inplace) {
                  // In-place layer: follow to its consumers
                  for (unsigned int i = 0; i < current->getNumOutputConnections(); ++i) {
                    auto next_conn = current->getOutputConnection(i);
                    if (next_conn) {
                      auto next_node = getLayerNode(next_conn->getName());
                      if (next_node) {
                        consumers_to_process.push_back(next_node);
                      }
                    }
                  }
                } else {
                  // Non-in-place layer: this is an actual consumer
                  auto sink_forward_order = std::get<0>(current->getExecutionOrder());
                  append_consumer_order(spec, sink_forward_order + 1);
                  consumer_added = true;
                  printf("[INITIAL_OUTPUT-LIFESPAN] Layer '%s' output[%zu] -> consumer '%s' (forward_order=%u, added order=%u)\n",
                         lnode->getName().c_str(), spec_idx, current->getName().c_str(), 
                         sink_forward_order, sink_forward_order + 1);
                }
              }
            }
          }
        }

        if (!consumer_added) {
          append_consumer_order(spec, terminal_forward_order + 1);
          printf("[INITIAL_OUTPUT-LIFESPAN] Layer '%s' output[%zu] -> no consumer, using terminal_forward_order+1=%u\n",
                 lnode->getName().c_str(), spec_idx, terminal_forward_order + 1);
        }
      }
      
      // Debug: print additional_exec_order before requesting tensors
      for (size_t i = 0; i < initial_out_specs.size(); i++) {
        printf("[DEBUG] Layer '%s' initial_out_specs[%zu] additional_exec_order: [", 
               lnode->getName().c_str(), i);
        for (auto o : initial_out_specs[i].variable_spec.additional_exec_order) {
          printf("%u ", o);
        }
        printf("]\n");
      }
      
      initial_outputs = tensor_manager->requestTensors(
        initial_out_specs, Manager::TensorGroupType::OUTPUT, 
        lnode->getExecutionOrder(), lnode->getName());
      
      printf("[TENSOR-ALLOC] ===== Layer: %s - INITIAL_OUTPUTS =====\n", lnode->getName().c_str());
      for (size_t i = 0; i < initial_outputs.size(); i++) {
        printf("[TENSOR-ALLOC] Layer: %s, Type: INITIAL_OUTPUT, Name: %s, Ptr: %p, Lifespan: %s\n",
               lnode->getName().c_str(),
               initial_outputs[i]->getName().c_str(),
               (void*)initial_outputs[i],
               lifespanToString(TensorLifespan::FORWARD_FUNC_LIFESPAN, exec_order).c_str());
      }
    }
    ml_logd("Layer '%s' is checkpointed - requested %zu initial outputs (FORWARD_FUNC_LIFESPAN)",
            lnode->getName().c_str(), initial_outputs.size());
  }

  /** create shared weight names if requested */
  std::vector<std::string> shared_weight_names;
  std::vector<std::string> shared_tensor_names;
  if (auto shared_node_str = lnode->getSharedFrom(); !shared_node_str.empty()) {
    /// @note below is commented but kept from quick fix to be referenced
    /// for later(#1707)
    // auto shared_node = getLayerNode(shared_node_str).get();
    // NNTR_THROW_IF(shared_node == nullptr, std::invalid_argument)
    //   << "shared_node requested but it is not registered in the graph,
    //   name:
    //   "
    //   << shared_node_str << " requested from " << lnode->getName();
    // NNTR_THROW_IF(shared_node->getType() != lnode->getType(),
    //               std::invalid_argument)
    //   << " shared_node and lnode type mismatch, source node type: "
    //   << shared_node->getType() << " depedent node type: " <<
    //   lnode->getType()
    //   << " depedent node name: " << lnode->getName();
    // NNTR_THROW_IF(!shared_node->isFinalized(), std::invalid_argument)
    //   << "shared node must be prior to the dependent node and it should
    //   be
    //   "
    //      "finalized beforehand, shared node name: "
    //   << shared_node_str << " dependent node name: " << lnode->getName();
    // auto num_weight = shared_node->getNumWeights();
    // shared_weight_names.reserve(num_weight);
    // for (auto i = 0u; i < num_weight; ++i) {
    //   shared_weight_names.emplace_back(shared_node->getWeightName(i));
    // }
    // auto &rc = node->getRunContext();

    /// @fixme tensor should be only shared if context explicitly requested
    /// to do so. This has to be added to the part of tensor spec, other
    /// wise it will break many things
    const auto &t_specs = init_context.getTensorsSpec();
    for (auto i = 0u; i < t_specs.size(); ++i) {
      shared_tensor_names.emplace_back(std::get<3>(t_specs.at(i)));
    }

    const auto &w_specs = init_context.getWeightsSpec();
    for (auto i = 0u; i < w_specs.size(); ++i) {
      shared_weight_names.emplace_back(std::get<8>(w_specs.at(i)));
    }
  }
  lnode->setDataType(init_context.getWeightDataType(),
                     init_context.getActivationDataType());
  bool trainable = lnode->getTrainable();
  if (exec_mode == ExecutionMode::INFERENCE)
    trainable = false;

  auto context = ct_engine.getRegisteredContext(lnode->getComputeEngineType());

  auto ct_data = context->getContextData();

  // Gradient checkpointing: Handle intermediate tensors for checkpointed layers
  auto tensors_spec = init_context.getTensorsSpec();
  std::vector<Var_Grad *> initial_tensors;
  
  if (lnode->isCheckpointed()) {
    // Check if any intermediate tensors are used in backward
    bool has_backward_tensors = false;
    for (auto &spec : tensors_spec) {
      auto &[dim, init, need_grad, name, lifespan, engine] = spec;
      // If tensor has gradient or used in CALC_GRAD/CALC_DERIV, it's needed in backward
      if (need_grad || 
          lifespan == TensorLifespan::CALC_GRAD_LIFESPAN ||
          lifespan == TensorLifespan::CALC_DERIV_LIFESPAN) {
        has_backward_tensors = true;
        break;
      }
    }
    
    if (has_backward_tensors) {
      // Dual allocation for intermediate tensors (like outputs)
      std::vector<InitLayerContext::TensorSpec> initial_tensors_spec;
      initial_tensors_spec.reserve(tensors_spec.size());
      
      for (const auto &spec : tensors_spec) {
        const auto &[dim, init, need_grad, name, lifespan, engine] = spec;
        // Create new spec with modified name and lifespan
        initial_tensors_spec.emplace_back(
          dim, init, need_grad, name + "_initial",
          TensorLifespan::FORWARD_FUNC_LIFESPAN, engine
        );
      }
      
      initial_tensors = tensor_manager->requestTensors(
        *lnode.get(), initial_tensors_spec, true, {});
      
      printf("[TENSOR-ALLOC] ===== Layer: %s - INITIAL_TENSORS =====\n", lnode->getName().c_str());
      for (size_t i = 0; i < initial_tensors.size(); i++) {
        printf("[TENSOR-ALLOC] Layer: %s, Type: INITIAL_TENSOR, Name: %s, Ptr: %p, Lifespan: %s\n",
               lnode->getName().c_str(),
               initial_tensors[i]->getName().c_str(),
               (void*)initial_tensors[i],
               lifespanToString(TensorLifespan::FORWARD_FUNC_LIFESPAN, exec_order).c_str());
      }
      
      // Recompute tensors: change lifespan to be available during backward
      for (auto &spec : tensors_spec) {
        auto &lifespan = std::get<4>(spec);
        if (lifespan == TensorLifespan::FORWARD_GRAD_LIFESPAN) {
          lifespan = TensorLifespan::CALC_DERIV_LIFESPAN;
        }
        lifespan = promoteToRecompute(lifespan);
      }
      
      ml_logd("Layer '%s' has intermediate tensors used in backward - dual allocation",
              lnode->getName().c_str());
    } else {
      // No backward usage: just make them short-lived
      for (auto &spec : tensors_spec) {
        auto &[dim, init, need_grad, name, lifespan, engine] = spec;
        if (lifespan == TensorLifespan::FORWARD_GRAD_LIFESPAN) {
          lifespan = TensorLifespan::FORWARD_FUNC_LIFESPAN;
        }
      }
      ml_logd("Layer '%s' intermediate tensors are short-lived (not used in backward)",
              lnode->getName().c_str());
    }
  }

  // Gradient checkpointing: Saved inputs will be set in initialize() 
  // after all layers are finalized and input_map is constructed
  std::vector<Var_Grad *> saved_inputs;
  if (is_first_checkpoint_layer) {
    saved_inputs = inputs;
    printf("[TENSOR-ALLOC] ===== Layer: %s - INITIAL_INPUTS (saved from inputs) =====\n", lnode->getName().c_str());
    for (size_t i = 0; i < saved_inputs.size(); i++) {
      printf("[TENSOR-ALLOC] Layer: %s, Type: INITIAL_INPUT, Name: %s, Ptr: %p\n",
             lnode->getName().c_str(),
             saved_inputs[i]->getName().c_str(),
             (void*)saved_inputs[i]);
    }
  }
  
  ml_logd("[CHECKPOINT] Layer '%s' - About to configure RunContext (checkpointed=%d, first=%d)", 
          lnode->getName().c_str(), lnode->isCheckpointed(), lnode->isFirstInCheckpointBlock());

  // Configure RunContext with recompute outputs (for input_map pointer connections)
  // This ensures that inputs[] in next layer points to recompute_outputs
  // Print tensor allocation info for intermediate tensors
  if (!tensors_spec.empty()) {
    printf("[TENSOR-ALLOC] ===== Layer: %s - TENSORS (recompute) =====\n", lnode->getName().c_str());
    for (size_t i = 0; i < tensors_spec.size(); i++) {
      auto &[dim, init, need_grad, name, lifespan, engine] = tensors_spec[i];
      printf("[TENSOR-ALLOC] Layer: %s, Type: TENSOR, Name: %s, Lifespan: %s\n",
             lnode->getName().c_str(), name.c_str(),
             lifespanToString(lifespan, exec_order).c_str());
    }
  }
  
  lnode->configureRunContext(
    // TODO: update weights spec for trainable based on layer trainable prop
    tensor_manager->requestWeights(gnode, init_context.getWeightsSpec(),
                                   trainable, shared_weight_names),
    inputs, outputs,  // outputs = recompute_outputs for checkpointed layers
    tensor_manager->requestTensors(gnode, tensors_spec,
                                   trainable, shared_tensor_names),
    init_context.getLossScale(), ct_data);

  // Gradient checkpointing: Configure checkpointed layers AFTER configureRunContext
  if (!initial_outputs.empty() && !lnode->isCheckpointed()) {
    lnode->getRunContext().setInitialOutputs(initial_outputs);
  }

  if (lnode->isCheckpointed()) {
    ml_logd("[CHECKPOINT] Layer '%s' - Configuring checkpoint settings after RunContext", lnode->getName().c_str());
    
    // Store initial forward outputs (short-lived, used only in initial forward)
    lnode->getRunContext().setInitialOutputs(initial_outputs);
    
    // Store recompute outputs (used for recompute forward & backward)
    lnode->getRunContext().setRecomputeOutputs(outputs);
    
    // Store initial tensors if allocated
    lnode->getRunContext().setInitialTensors(initial_tensors);
    
    // Store saved inputs if allocated (first layer only)
    lnode->getRunContext().setInitialInputs(saved_inputs);
    
    lnode->getRunContext().setCheckpointed(true);
    
    ml_logd("Layer '%s' configured as checkpointed with %zu initial outputs, %zu recompute outputs, %zu initial tensors, %zu saved inputs",
            lnode->getName().c_str(), initial_outputs.size(), outputs.size(), initial_tensors.size(), saved_inputs.size());
  }

  // Return outputs (recompute_outputs for checkpointed, normal outputs otherwise)
  // This is used for input_map pointer connections
  return outputs;
}

std::vector<Var_Grad *>
NetworkGraph::refinalizeContext(const std::shared_ptr<LayerNode> &lnode,
                                const std::vector<Var_Grad *> &prev_inputs) {
  const GraphNode &gnode = *lnode.get();
  std::vector<TensorDim> input_dims;
  input_dims.reserve(prev_inputs.size());
  std::transform(prev_inputs.begin(), prev_inputs.end(),
                 std::back_inserter(input_dims),
                 [](const Var_Grad *vg) { return vg->getDim(); });

  // DEBUG: Print input dimensions for refinalize
  printf("[DEBUG-REINIT] Layer '%s' (type: %s) - Input dimensions:\n", 
          lnode->getName().c_str(), lnode->getType().c_str());
  for (size_t i = 0; i < input_dims.size(); ++i) {
    printf("  Input[%zu]: %zu:%zu:%zu:%zu (buffer size: %zu bytes)\n", 
            i, input_dims[i].batch(), input_dims[i].channel(), 
            input_dims[i].height(), input_dims[i].width(),
            input_dims[i].getDataLen() * sizeof(float));
  }
  fflush(stdout);

  /** refinalize the layer and get the final context */
  auto init_context = lnode->refinalize(input_dims);
  
  // DEBUG: Print output dimensions after refinalize
  auto out_specs_debug = init_context.getOutSpecs();
  printf("[DEBUG-REINIT] Layer '%s' - Output dimensions:\n", lnode->getName().c_str());
  for (size_t i = 0; i < out_specs_debug.size(); ++i) {
    auto &dim = out_specs_debug[i].variable_spec.dim;
    printf("  Output[%zu]: %zu:%zu:%zu:%zu (buffer size: %zu bytes)\n", 
            i, dim.batch(), dim.channel(), dim.height(), dim.width(),
            dim.getDataLen() * sizeof(float));
  }
  fflush(stdout);
  const auto &ct_engine = nntrainer::Engine::Global();

  /**
   * Request manager for either a pre-allocated output as input or a newly
   * allocated output. This is necessary for manager to know when this
   * output node is going to be used.
   */
  std::vector<std::string> input_names;
  input_names.reserve(prev_inputs.size());
  std::transform(
    prev_inputs.begin(), prev_inputs.end(), std::back_inserter(input_names),
    [](auto const &vg) -> const auto & { return vg->getName(); });
  const std::vector<Var_Grad *> &inputs = tensor_manager->requestInputs(
    gnode, init_context.getInputDimensions(), input_names);

  /** In-Place optimizations */
  /**
   * Request manager for either a pre-allocated input as output or a newly
   * allocated output. This is necessary for manager to know when this
   * output node is going to be used with in-place optimizations.
   */
  auto out_specs = init_context.getOutSpecs();
  /// @note try move inplace control to finalize
  bool shared_var = false, shared_grad = false;
  if (lnode->getInPlaceType() != InPlaceType::NONE) {
    setInplaceSharedMemoryConfigByLayer(lnode, shared_var, shared_grad);
    for (unsigned int i = 0; i < out_specs.size(); ++i) {
      auto &s = out_specs.at(i);
      if (shared_var) {
        s.variable_spec.request_type =
          TensorSpecV2::RequestType::READ_ONLY_VIEW;
        if (lnode->getType() == IdentityLayer::type) {
          s.variable_spec.reference_name = inputs[i]->getName();
        } else if (lnode->getInPlaceDirection() == InPlaceDirection::RIGHT) {
          s.variable_spec.reference_name = inputs[1]->getName();
        } else {
          s.variable_spec.reference_name = inputs[0]->getName();
        }
      }
      if (shared_grad && s.gradient_spec) {
        s.gradient_spec->request_type =
          TensorSpecV2::RequestType::READ_ONLY_VIEW;
        if (lnode->getType() == IdentityLayer::type) {
          s.gradient_spec->reference_name = inputs[i]->getGradientName();
        } else if (lnode->getInPlaceDirection() == InPlaceDirection::RIGHT) {
          // @note With binary inputs, inputs[0] represents the left input
          // tensor while inputs[1] represents the right input tensor. As a
          // result, if the in-place direction is set to right, the in-place
          // memory is assigned to inputs[1].
          s.gradient_spec->reference_name = inputs[1]->getGradientName();
        } else {
          s.gradient_spec->reference_name = inputs[0]->getGradientName();
        }
      }
    }
  }
  if (lnode->requireLabel()) {
    NNTR_THROW_IF(out_specs.size() != 1, std::invalid_argument)
      << "out specification size must be 1 for label layer for now, "
      << lnode->getName() << " out spec size: " << out_specs.size();
    NNTR_THROW_IF(out_specs[0].gradient_spec == nullptr, std::invalid_argument)
      << "label space does not exist for " << lnode->getName();
    out_specs[0].gradient_spec->request_type =
      TensorSpecV2::RequestType::PLACEHOLDER;
  }

  /// @note below needs to be enabled only for inference mode, but need
  /// decision if we are going to separate inference initialization from
  /// train initialization this might not worth optimize because in general
  /// output of a neuralnet is very small
  if (lnode->getOutputConnections().size() == 0u) {
    std::for_each(out_specs.begin(), out_specs.end(),
                  [this](VarGradSpecV2 &spec) {
                    spec.variable_spec.additional_exec_order.push_back(
                      std::get<0>(forward_iter_end->getExecutionOrder()));
                  });
  }

  if (lnode->getType() == RNNCellLayer::type or
      lnode->getType() == LSTMCellLayer::type or
      lnode->getType() == GRUCellLayer::type) {
    std::for_each(out_specs.begin(), out_specs.end(), [](VarGradSpecV2 &spec) {
      spec.variable_spec.ls = TensorLifespan::FORWARD_GRAD_LIFESPAN;
    });
  }

  const std::vector<Var_Grad *> &outputs = tensor_manager->requestTensors(
    out_specs, Manager::TensorGroupType::OUTPUT, lnode->getExecutionOrder(),
    lnode->getName());

  /** create shared weight names if requested */
  std::vector<std::string> shared_weight_names;
  std::vector<std::string> shared_tensor_names;
  if (auto shared_node_str = lnode->getSharedFrom(); !shared_node_str.empty()) {
    /// @note below is commented but kept from quick fix to be referenced
    /// for later(#1707)
    // auto shared_node = getLayerNode(shared_node_str).get();
    // NNTR_THROW_IF(shared_node == nullptr, std::invalid_argument)
    //   << "shared_node requested but it is not registered in the graph,
    //   name:
    //   "
    //   << shared_node_str << " requested from " << lnode->getName();
    // NNTR_THROW_IF(shared_node->getType() != lnode->getType(),
    //               std::invalid_argument)
    //   << " shared_node and lnode type mismatch, source node type: "
    //   << shared_node->getType() << " depedent node type: " <<
    //   lnode->getType()
    //   << " depedent node name: " << lnode->getName();
    // NNTR_THROW_IF(!shared_node->isFinalized(), std::invalid_argument)
    //   << "shared node must be prior to the dependent node and it should
    //   be
    //   "
    //      "finalized beforehand, shared node name: "
    //   << shared_node_str << " dependent node name: " << lnode->getName();
    // auto num_weight = shared_node->getNumWeights();
    // shared_weight_names.reserve(num_weight);
    // for (auto i = 0u; i < num_weight; ++i) {
    //   shared_weight_names.emplace_back(shared_node->getWeightName(i));
    // }
    // auto &rc = node->getRunContext();

    /// @fixme tensor should be only shared if context explicitly requested
    /// to do so. This has to be added to the part of tensor spec, other
    /// wise it will break many things
    const auto &t_specs = init_context.getTensorsSpec();
    for (auto i = 0u; i < t_specs.size(); ++i) {
      shared_tensor_names.emplace_back(std::get<3>(t_specs.at(i)));
    }

    const auto &w_specs = init_context.getWeightsSpec();
    for (auto i = 0u; i < w_specs.size(); ++i) {
      shared_weight_names.emplace_back(std::get<8>(w_specs.at(i)));
    }
  }

  auto weights = lnode->getRunContext().getWeights();

  auto context = ct_engine.getRegisteredContext(lnode->getComputeEngineType());

  auto ct_data = context->getContextData();

  lnode->configureRunContext(
    // TODO: update weights spec for trainable based on layer trainable prop
    weights, inputs, outputs,
    tensor_manager->requestTensors(gnode, init_context.getTensorsSpec(),
                                   lnode->getTrainable(), shared_tensor_names),
    init_context.getLossScale(), ct_data);

  return outputs;
}

#ifdef ENABLE_TEST

std::map<std::string, std::vector<unsigned int>>
NetworkGraph::getLayerExecutionOrders(const std::shared_ptr<LayerNode> &lnode) {
  const auto &init_context = lnode->getInitContext();
  auto out_specs = init_context.getOutSpecs();
  auto weight_specs = init_context.getWeightsSpec();
  auto tensor_specs = init_context.getTensorsSpec();

  std::map<std::string, std::vector<unsigned int>> exec_orders;

  for (auto &spec : out_specs) {
    const auto &name = lnode->getName() + ":" + spec.variable_spec.name;
    auto orders = tensor_manager->getTensorExecutionOrders(name, false);
    exec_orders.insert({name, orders});
    try {
      auto orders_grad =
        tensor_manager->getTensorExecutionOrders(name + ":grad", false);
      exec_orders.insert({name + ":grad", orders_grad});
    } catch (const std::exception &e) {
      ml_logi("Cannot find grad tensor for %s:grad", name.c_str());
      continue;
    }
  }

  for (auto &spec : weight_specs) {
    const auto &name = std::get<const std::string>(spec);
    auto orders = tensor_manager->getTensorExecutionOrders(name, true);
    exec_orders.insert({name, orders});
    try {
      auto orders_grad =
        tensor_manager->getTensorExecutionOrders(name + ":grad", false);
      exec_orders.insert({name + ":grad", orders_grad});
    } catch (const std::exception &e) {
      ml_logi("Cannot find grad tensor for %s:grad", name.c_str());
      continue;
    }
  }

  for (auto &spec : tensor_specs) {
    const auto &name = std::get<const std::string>(spec);
    auto orders = tensor_manager->getTensorExecutionOrders(name, false);
    exec_orders.insert({name, orders});
    try {
      auto orders_grad =
        tensor_manager->getTensorExecutionOrders(name + ":grad", false);
      exec_orders.insert({name + ":grad", orders_grad});
    } catch (const std::exception &e) {
      ml_logi("Cannot find grad tensor for %s:grad", name.c_str());
      continue;
    }
  }

  return exec_orders;
}

#endif // ENABLE_TEST

int NetworkGraph::initialize(ExecutionMode mode,
                             const std::vector<Connection> &model_input_names,
                             const std::vector<Connection> &model_label_names) {
  ml_logi("[CHECKPOINT] NetworkGraph::initialize() starting...");
  exec_mode = mode;
  tensor_manager->setExecutionMode(mode);
  /**
   * this contains the map from node name to its input tensor names
   * @note: these input tensors have already been allocated
   */
  std::unordered_map<std::string, std::vector<Var_Grad *>> input_map;
  
  /**
   * this contains the map from node name to its INITIAL input tensor names
   * (for gradient checkpointing initial forward pass)
   * @note: these input tensors are short-lived, used only during initial forward
   */
  std::unordered_map<std::string, std::vector<Var_Grad *>> initial_input_map;

  /** check if the given config of node is of input node */
  auto is_input_node = [](const LayerNode *node) -> bool {
    return node->getInputConnections().empty();
  };

  // Register checkpoint recompute profiling event
  if (profile_keys.find("checkpoint_recompute") == profile_keys.end()) {
    int event_key = 0;
    PROFILE_TIME_REGISTER_EVENT(event_key, "checkpoint_recompute");
    profile_keys["checkpoint_recompute"] = event_key;
  }

  for (unsigned int idx = 0; idx < graph.size(); ++idx) {
    std::vector<Var_Grad *> inputs = {};
    auto const &lnode = getSortedLayerNode(idx);
    if (profile_keys.find(lnode->getType()) == profile_keys.end()) {
      int event_key = 0;
      PROFILE_TIME_REGISTER_EVENT(event_key, lnode->getType());
      profile_keys[lnode->getType()] = event_key;
    }

    /**
     * Set input dimension for all the layers.
     * For input layer, as input dimension is known, set input tensor.
     */
    if (!is_input_node(lnode.get())) {
      if (input_map.find(lnode->getName()) == input_map.end())
        throw std::runtime_error("Cannot find input buffers for the node");
      inputs = input_map.at(lnode->getName());
    }

    /**
     * Initialize all the layers, allocate output tensors for each layer
     * and add optimizer related weights for the layer
     */
    const std::vector<Var_Grad *> &outputs = finalizeContext(lnode, inputs);

    /** no need to update input_map for the last layer */
    if (idx == graph.size() - 1)
      break;

    // Collect initial forward outputs to propagate to downstream sinks.
    // Non-checkpointed layers can also own initial outputs if they feed a
    // checkpointed consumer.
    std::vector<Var_Grad *> initial_outputs =
      lnode->getRunContext().getInitialOutputs();

    auto reuse_recorded_initial_inputs = [&]() {
      auto init_it = initial_input_map.find(lnode->getName());
      if (init_it != initial_input_map.end() && !init_it->second.empty()) {
        initial_outputs = init_it->second;
        printf("[CHECKPOINT] Layer '%s' - reusing recorded initial inputs for connections\n",
               lnode->getName().c_str());
        return true;
      }
      return false;
    };

    // For checkpointed in-place layers, use initial_inputs instead of initial_outputs
    // This is because getOutput() returns initial_inputs for in-place layers
    // Use actual in-place type instead of hardcoded layer types
    // This correctly handles cases like GELU (not in-place) vs ReLU (in-place)
    bool is_inplace_layer = (lnode->getInPlaceType() != InPlaceType::NONE);
    if (lnode->isCheckpointed() && is_inplace_layer) {
      // For FIRST checkpoint layer, use normal inputs (from input_map) instead of initial_input_map
      // because first layer uses saved_inputs = inputs (normal inputs)
      std::vector<Var_Grad *> initial_inputs_vec;
      if (lnode->isFirstInCheckpointBlock()) {
        // First checkpoint layer: use normal inputs
        auto input_it = input_map.find(lnode->getName());
        if (input_it != input_map.end()) {
          initial_inputs_vec = input_it->second;
          printf("[CHECKPOINT] Layer '%s' is FIRST in-place checkpoint - using normal inputs (input_map) for connections\n",
                 lnode->getName().c_str());
        }
      } else {
        // Non-first checkpoint layer: use initial_input_map
        auto init_it = initial_input_map.find(lnode->getName());
        if (init_it != initial_input_map.end()) {
          initial_inputs_vec = init_it->second;
          printf("[CHECKPOINT] Layer '%s' is in-place - using initial_input_map as initial_outputs for connections\n",
                 lnode->getName().c_str());
        } else {
          initial_inputs_vec = lnode->getRunContext().getInitialInputs();
        }
      }
      if (!initial_inputs_vec.empty()) {
        initial_outputs = initial_inputs_vec;
        printf("[CHECKPOINT] Layer '%s' is in-place - using initial_inputs as initial_outputs for connections\n",
               lnode->getName().c_str());
        
        // For multi-output in-place layers (e.g., MultiOut), replicate the single initial_input
        // across all output indices since they share the same memory
        bool is_multiout_layer = (lnode->getType() == MultiOutLayer::type);
        if (is_multiout_layer && initial_outputs.size() == 1 && lnode->getNumOutputConnections() > 1) {
          auto shared_tensor = initial_outputs[0];
          initial_outputs.resize(lnode->getNumOutputConnections());
          for (size_t i = 0; i < initial_outputs.size(); ++i) {
            initial_outputs[i] = shared_tensor;
          }
          printf("[CHECKPOINT] Layer '%s' (MultiOut) - replicated initial_input across %zu outputs\n",
                 lnode->getName().c_str(), initial_outputs.size());
        }
      }
      else{
        printf("[CHECKPOINT] Layer '%s' is in-place - no initial_inputs found\n", lnode->getName().c_str());
      }
    }

    if (initial_outputs.empty()) {
      reuse_recorded_initial_inputs();
    }

    for (auto i = 0u, num_node = lnode->getNumOutputConnections(); i < num_node;
         ++i) {
      auto conn = lnode->getOutputConnection(i);
      if (!conn) {
        ml_logi("out connection not defined for  %s, %u",
                lnode->getName().c_str(), i);
        continue;
      }

      auto sink_node = getLayerNode(conn->getName());
      [[maybe_unused]] auto [it, b] =
        input_map.try_emplace({sink_node->getName(), {}});

      NNTR_THROW_IF(sink_node->getInputConnectionName(conn->getIndex()) !=
                      lnode->getName(),
                    std::invalid_argument)
        << "node pair does not match between " << lnode->getName() << ' '
        << sink_node->getName();

      auto &sink_tensors = it->second;
      sink_tensors.resize(sink_node->getNumInputConnections());
      sink_tensors[conn->getIndex()] = outputs[i];
      
      // Setup initial output connections (for initial forward)
      // If current layer is checkpointed, connect its initial_outputs (or initial_inputs for in-place) to next layer
      if (!initial_outputs.empty()) {
        [[maybe_unused]] auto [init_it, init_b] =
          initial_input_map.try_emplace({sink_node->getName(), {}});
        
        auto &init_sink_tensors = init_it->second;
        init_sink_tensors.resize(sink_node->getNumInputConnections());
        // For in-place layers, we only have 1 input/output, so use index 0
        unsigned int output_idx = (initial_outputs.size() == 1) ? 0 : i;
        init_sink_tensors[conn->getIndex()] = initial_outputs[output_idx];
        printf("[CHECKPOINT-CONNECT] %s[%u] -> %s[%u]: tensor_name='%s', ptr=%p, data_ptr=%p\n",
                lnode->getName().c_str(), output_idx, 
                sink_node->getName().c_str(), conn->getIndex(),
                initial_outputs[output_idx]->getName().c_str(),
                (void*)initial_outputs[output_idx],
                (void*)initial_outputs[output_idx]->getVariableRef().getData());
      }
    }
  }

  // CRITICAL: Set initial_inputs for ALL layers that receive inputs from checkpointed layers
  // This ensures that during initial forward, getInput() returns the correct initial outputs
  // EXCEPTION: First layer in checkpoint block already has initial_inputs set to normal inputs
  //            in finalizeContext() - do NOT overwrite with initial_input_map
  for (unsigned int idx = 0; idx < graph.size(); ++idx) {
    auto const &lnode = getSortedLayerNode(idx);
    
    // Skip first layer in checkpoint block - it already has initial_inputs = normal inputs
    // set in finalizeContext(). We don't want to overwrite with initial_outputs from
    // non-checkpointed previous layer.
    if (lnode->isCheckpointed() && lnode->isFirstInCheckpointBlock()) {
      printf("[CHECKPOINT-SET] Layer '%s' is FIRST in checkpoint block - keeping saved_inputs (normal inputs)\n",
              lnode->getName().c_str());
      continue;
    }
    
    if (initial_input_map.find(lnode->getName()) != initial_input_map.end()) {
      auto &rc = lnode->getRunContext();
      auto &initial_inputs_vec = initial_input_map.at(lnode->getName());
      rc.setInitialInputs(initial_inputs_vec);
      
      printf("[CHECKPOINT-SET] Layer '%s' (checkpointed=%d) received %zu initial_inputs:\n",
              lnode->getName().c_str(), lnode->isCheckpointed(), initial_inputs_vec.size());
      for (size_t i = 0; i < initial_inputs_vec.size(); ++i) {
        if (initial_inputs_vec[i]) {
          printf("[CHECKPOINT-SET]   [%zu] tensor_name='%s', ptr=%p, data_ptr=%p\n",
                 i, initial_inputs_vec[i]->getName().c_str(),
                 (void*)initial_inputs_vec[i],
                 (void*)initial_inputs_vec[i]->getVariableRef().getData());
        } else {
          printf("[CHECKPOINT-SET]   [%zu] NULL\n", i);
        }
      }
    }
  }

  for (unsigned int idx = 0; idx < graph.size(); ++idx) {
    auto const &lnode = getSortedLayerNode(idx);
    auto &rc = lnode->getRunContext();
    auto first_grad_access = std::get<2>(lnode->getExecutionOrder());
    auto last_grad_access = std::get<4>(lnode->getExecutionOrder());
    for (unsigned i = 0; i < rc.getNumWeights(); ++i) {
      if (!rc.weightHasGradient(i)) {
        /// @todo this is duck taping that MUST BE REMOVED. We will need to
        /// have, is weight first access kind of concept.
        if (tensor_manager->isFirstAccess(
              rc.getWeight(i).getName(),
              std::get<0>(lnode->getExecutionOrder()), true)) {
          rc.getWeightObject(i).setAsGradientFirstAccess();
        }
        if (tensor_manager->isLastAccess(rc.getWeight(i).getName(),
                                         last_grad_access, true)) {
          rc.getWeightObject(i).setAsGradientLastAccess();
        }
      } else {
        if (tensor_manager->isFirstAccess(rc.getWeightGrad(i).getName(),
                                          first_grad_access)) {
          rc.getWeightObject(i).setAsGradientFirstAccess();
        }
        /**
         * if the gradient is to be clipped by global norm, then the last
         * access is by clipping itself. However, as clipping is not a layer
         * and does not contain any weights, such weights never get assigned
         * gradient_last_access. This is a quick hotfix.
         * TODO: make an independent clipping layer which will execute at
         * the end, and will share ownership of weights which it will clip.
         * This will remove this hot fix, and also remove the checks of if
         * weights require clipping.
         */
        if (tensor_manager->isLastAccess(rc.getWeightGrad(i).getName(),
                                         last_grad_access) ||
            ((rc.isGradientClipByGlobalNorm(i) || rc.isMixedPrecision(i)) &&
             tensor_manager->isSecondLastAccess(rc.getWeightGrad(i).getName(),
                                                last_grad_access))) {
          rc.getWeightObject(i).setAsGradientLastAccess();
        }
      }
    }
  }

  /**** identify model input / output to be set externally later ****/
  auto identify_as_model_input = [this](LayerNode *node) {
    auto num_input = node->getNumInputs();
    NNTR_THROW_IF(num_input != 1, std::invalid_argument)
      << "Input layer is supposed to have exactly one input, but more then "
         "one input detected, num inputs: "
      << num_input;

    input_list.push_back(node->getInput(0).getName());
    input_dims_.push_back(node->getInputDimensions()[0]);
  };

  auto is_label_node = [](LayerNode *node) { return node->requireLabel(); };

  auto identify_as_model_label = [this](LayerNode *node) {
    /// @todo change this as lnode->getNumLabels of sorts
    auto num_label = node->getNumOutputs();
    NNTR_THROW_IF(!node->getOutputConnections().empty(), std::invalid_argument)
      << "label layer is supposed to be a leaf for now";
    NNTR_THROW_IF(num_label != 1, std::invalid_argument)
      << "label layer is supposed to have exactly one label, but more then "
         "one label detected, num labels: "
      << num_label;

    /// @todo implement and use getLabel(0) instead.
    output_list.push_back(node->getOutput(0).getName());
    label_list.push_back(node->getOutputGrad(0).getName());
    label_dims_.push_back(node->getOutputDimensions()[0]);
  };

  auto identify_external_tensors = [this](const std::vector<Connection> &conns,
                                          auto &&pred, auto &&identify) {
    if (conns.empty()) {
      for (unsigned int i = 0; i < graph.size(); ++i) {
        auto lnode = getSortedLayerNode(i).get();
        if (!pred(lnode)) {
          continue;
        }
        /// when name is empty, we identify everything as the node, all of
        /// them must be having identical dimensions
        identify(lnode);
      }
    } else {
      for (auto &conn : conns) {
        auto lnode = getLayerNode(conn.getName()).get();
        NNTR_THROW_IF(!pred(lnode), std::invalid_argument)
          << "given node is not of that kind, name: " << conn.getName();
        identify(lnode);
      }
      unsigned int num_node_of_kind = 0;
      for (unsigned int i = 0; i < graph.size(); ++i) {
        auto lnode = getSortedLayerNode(i).get();
        if (!pred(lnode)) {
          continue;
        }
        num_node_of_kind++;
      }
      NNTR_THROW_IF(num_node_of_kind != conns.size(), std::invalid_argument)
        << "conns given but there are not identified node of the kind, num "
           "node of kind: "
        << num_node_of_kind << " identifier size: " << conns.size();
    }
  };

  identify_external_tensors(model_input_names, is_input_node,
                            identify_as_model_input);
  identify_external_tensors(model_label_names, is_label_node,
                            identify_as_model_label);
  /** mark the nodes which will be backwarded during the graph operation */
  try {
    markNodesForBackwarding();
    backward_iter_end = computeBackwardEnd();
  } catch (std::exception &e) {
    ml_loge("Backwarding required from layer which doesn't support "
            "backwarding: %s",
            e.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  /** select weights which would require clipping of the gradients by global
   * norm if any */
  lazy_weights = tensor_manager->getWeights([](const Weight *w) {
    return w->hasGradient() && w->isGradientLastAccess() &&
           (w->isGradientClipByGlobalNorm() || w->isMixedPrecision());
  });

  is_clip_grad = false;
  for (auto w : lazy_weights) {
    if (w->isGradientClipByGlobalNorm()) {
      is_clip_grad = true;
      break;
    }
  }
  return ML_ERROR_NONE;
}

int NetworkGraph::reinitialize(
  const std::vector<Connection> &model_input_names,
  const std::vector<Connection> &model_label_names) {
  input_dims_.clear();
  label_dims_.clear();
  tensor_manager->reinitialize();

  /**
   * this contains the map from node name to its input tensor names
   * @note: these input tensors have already been allocated
   */
  std::unordered_map<std::string, std::vector<Var_Grad *>> input_map;

  /** check if the given config of node is of input node */
  auto is_input_node = [](const LayerNode *node) -> bool {
    return node->getInputConnections().empty();
  };

  // Register checkpoint recompute profiling event
  if (profile_keys.find("checkpoint_recompute") == profile_keys.end()) {
    int event_key = 0;
    PROFILE_TIME_REGISTER_EVENT(event_key, "checkpoint_recompute");
    profile_keys["checkpoint_recompute"] = event_key;
  }

  for (unsigned int idx = 0; idx < graph.size(); ++idx) {
    std::vector<Var_Grad *> inputs = {};
    auto const &lnode = getSortedLayerNode(idx);

    if (profile_keys.find(lnode->getType()) == profile_keys.end()) {
      int event_key = 0;
      PROFILE_TIME_REGISTER_EVENT(event_key, lnode->getType());
      profile_keys[lnode->getType()] = event_key;
    }

    /**
     * Set input dimension for all the layers.
     * For input layer, as input dimension is known, set input tensor.
     */
    if (!is_input_node(lnode.get())) {
      if (input_map.find(lnode->getName()) == input_map.end())
        throw std::runtime_error("Cannot find input buffers for the node");
      inputs = input_map.at(lnode->getName());
    }

    /**
     * Reinitialize all the layers, allocate output tensors for each layer
     * init2and add optimizer related weights for the layer
     */
    const std::vector<Var_Grad *> &outputs = refinalizeContext(lnode, inputs);

    /** no need to update input_map for the last layer */
    if (idx == graph.size() - 1)
      break;

    for (auto i = 0u, num_node = lnode->getNumOutputConnections(); i < num_node;
         ++i) {
      auto conn = lnode->getOutputConnection(i);
      if (!conn) {
        ml_logi("out connection not defined for  %s, %u",
                lnode->getName().c_str(), i);
        continue;
      }

      auto sink_node = getLayerNode(conn->getName());
      [[maybe_unused]] auto [it, b] =
        input_map.try_emplace({sink_node->getName(), {}});

      NNTR_THROW_IF(sink_node->getInputConnectionName(conn->getIndex()) !=
                      lnode->getName(),
                    std::invalid_argument)
        << "node pair does not match between " << lnode->getName() << ' '
        << sink_node->getName();

      auto &sink_tensors = it->second;
      sink_tensors.resize(sink_node->getNumInputConnections());
      sink_tensors[conn->getIndex()] = outputs[i];
    }
  }

  for (unsigned int idx = 0; idx < graph.size(); ++idx) {
    auto const &lnode = getSortedLayerNode(idx);
    auto &rc = lnode->getRunContext();
    auto first_grad_access = std::get<2>(lnode->getExecutionOrder());
    auto last_grad_access = std::get<4>(lnode->getExecutionOrder());
    for (unsigned i = 0; i < rc.getNumWeights(); ++i) {
      if (!rc.weightHasGradient(i)) {
        /// @todo this is duck taping that MUST BE REMOVED. We will need to
        /// have, is weight first access kind of concept.
        if (tensor_manager->isFirstAccess(
              rc.getWeight(i).getName(),
              std::get<0>(lnode->getExecutionOrder()), true)) {
          rc.getWeightObject(i).setAsGradientFirstAccess();
        }
        if (tensor_manager->isLastAccess(rc.getWeight(i).getName(),
                                         last_grad_access, true)) {
          rc.getWeightObject(i).setAsGradientLastAccess();
        }
      } else {
        if (tensor_manager->isFirstAccess(rc.getWeightGrad(i).getName(),
                                          first_grad_access)) {
          rc.getWeightObject(i).setAsGradientFirstAccess();
        }
        /**
         * if the gradient is to be clipped by global norm, then the last
         * access is by clipping itself. However, as clipping is not a layer
         * and does not contain any weights, such weights never get assigned
         * gradient_last_access. This is a quick hotfix.
         * TODO: make an independent clipping layer which will execute at
         * the end, and will share ownership of weights which it will clip.
         * This will remove this hot fix, and also remove the checks of if
         * weights require clipping.
         */
        if (tensor_manager->isLastAccess(rc.getWeightGrad(i).getName(),
                                         last_grad_access) ||
            (rc.isGradientClipByGlobalNorm(i) &&
             tensor_manager->isSecondLastAccess(rc.getWeightGrad(i).getName(),
                                                last_grad_access))) {
          rc.getWeightObject(i).setAsGradientLastAccess();
        }
      }
    }
  }
  /**** identify model input / output to be set externally later ****/
  auto identify_as_model_input = [this](LayerNode *node) {
    auto num_input = node->getNumInputs();
    NNTR_THROW_IF(num_input != 1, std::invalid_argument)
      << "Input layer is supposed to have exactly one input, but more then "
         "one input detected, num inputs: "
      << num_input;

    // input_list.push_back(node->getInput(0).getName());
    input_dims_.push_back(node->getInputDimensions()[0]);
  };

  auto is_label_node = [](LayerNode *node) { return node->requireLabel(); };

  auto identify_as_model_label = [this](LayerNode *node) {
    /// @todo change this as lnode->getNumLabels of sorts
    auto num_label = node->getNumOutputs();
    NNTR_THROW_IF(!node->getOutputConnections().empty(), std::invalid_argument)
      << "label layer is supposed to be a leaf for now";
    NNTR_THROW_IF(num_label != 1, std::invalid_argument)
      << "label layer is supposed to have exactly one label, but more then "
         "one label detected, num labels: "
      << num_label;

    /// @todo implement and use getLabel(0) instead.
    // output_list.push_back(node->getOutput(0).getName());
    // label_list.push_back(node->getOutputGrad(0).getName());
    label_dims_.push_back(node->getOutputDimensions()[0]);
  };

  auto identify_external_tensors = [this](const std::vector<Connection> &conns,
                                          auto &&pred, auto &&identify) {
    if (conns.empty()) {
      for (unsigned int i = 0; i < graph.size(); ++i) {
        auto lnode = getSortedLayerNode(i).get();
        if (!pred(lnode)) {
          continue;
        }
        /// when name is empty, we identify everything as the node, all of
        /// them must be having identical dimensions
        identify(lnode);
      }
    } else {
      for (auto &conn : conns) {
        auto lnode = getLayerNode(conn.getName()).get();
        NNTR_THROW_IF(!pred(lnode), std::invalid_argument)
          << "given node is not of that kind, name: " << conn.getName();
        identify(lnode);
      }
      unsigned int num_node_of_kind = 0;
      for (unsigned int i = 0; i < graph.size(); ++i) {
        auto lnode = getSortedLayerNode(i).get();
        if (!pred(lnode)) {
          continue;
        }
        num_node_of_kind++;
      }
      NNTR_THROW_IF(num_node_of_kind != conns.size(), std::invalid_argument)
        << "conns given but there are not identified node of the kind, num "
           "node of kind: "
        << num_node_of_kind << " identifier size: " << conns.size();
    }
  };

  identify_external_tensors(model_input_names, is_input_node,
                            identify_as_model_input);
  identify_external_tensors(model_label_names, is_label_node,
                            identify_as_model_label);

  return ML_ERROR_NONE;
}

void NetworkGraph::setExternalTensors(const std::vector<Tensor> &data,
                                      const std::vector<std::string> names) {
  /// feed or clear label
  for (unsigned int idx = 0; idx < names.size(); idx++) {
    if (data.empty())
      tensor_manager->fillPlaceholder(names[idx], Tensor());
    else if (data.size() == 1)
      tensor_manager->fillPlaceholder(names[idx], data[0]);
    else
      tensor_manager->fillPlaceholder(names[idx], data[idx]);
  }
}

void NetworkGraph::setInputsLabels(const std::vector<Tensor> &inputs,
                                   const std::vector<Tensor> &labels) {
  NNTR_THROW_IF(labels.size() > 1 && labels.size() != label_list.size(),
                std::invalid_argument)
    << "label size does not match with the network requirements"
    << " label size: " << labels.size()
    << " requirements size: " << label_list.size();

  NNTR_THROW_IF(inputs.size() > 1 && inputs.size() != input_list.size(),
                std::invalid_argument)
    << "input size does not match with the network requirements"
    << " input size: " << inputs.size()
    << " requirements size: " << input_list.size();

  setExternalTensors(inputs, input_list);
  setExternalTensors(labels, label_list);
}

void NetworkGraph::setInputsLabels(sharedConstTensors &inputs,
                                   sharedConstTensors &labels) {
  std::vector<Tensor> ins;
  std::transform(
    inputs.begin(), inputs.end(), std::back_inserter(ins),
    [](auto const &val) -> const auto & { return *val.get(); });

  std::vector<Tensor> labs;
  std::transform(
    labels.begin(), labels.end(), std::back_inserter(labs),
    [](auto const &val) -> const auto & { return *val.get(); });

  setInputsLabels(ins, labs);
}

std::vector<Tensor> NetworkGraph::getOutputTensors() const {
  std::vector<Tensor> output_tensors;
  output_tensors.reserve(output_list.size());

  for (auto const &name : output_list)
    output_tensors.push_back(*tensor_manager->getTensor(name));

  return output_tensors;
}

void NetworkGraph::flushCache() { tensor_manager->flushCache(); }

void NetworkGraph::flushCacheExcept(unsigned int order) {
  tensor_manager->flushCacheExcept(order);
}

void NetworkGraph::LoadTensors(unsigned int order, unsigned int lookahead) {
  tensor_manager->LoadTensors(order, lookahead);
}

bool NetworkGraph::checkLoadComplete(unsigned int order) {
  return tensor_manager->checkLoadComplete(order);
}

bool NetworkGraph::inActive(unsigned int order) {
  return tensor_manager->inActive(order);
}

bool NetworkGraph::checkUnloadComplete(unsigned int order) {
  return tensor_manager->checkUnloadComplete(order);
}

void NetworkGraph::UnloadTensors(unsigned int order) {
  tensor_manager->UnloadTensors(order);
}

void NetworkGraph::requestOptimizerVariable(
  std::function<std::vector<TensorDim>(const TensorDim &)> cb,
  bool request_only_trainable) {
  for (auto const &w : tensor_manager->getWeights()) {
    if (w->isGradientLastAccess() && w->hasGradient()) {
      const TensorDim &dim = w->getDim();
      std::vector<TensorDim> dims = cb(dim);
      w->setOptimizerVariables(tensor_manager->requestWeightOptimizerVariables(
        dims, w->getName(), ":opt", TensorLifespan::MAX_LIFESPAN,
        w->isGradientClipByGlobalNorm(), w->isMixedPrecision(),
        Initializer::ZEROS));
    }
  }
}

void NetworkGraph::resetLossScale(float scale) {
  loss_scale = scale;
  for (auto iter = cbegin(); iter != cend(); iter++) {
    auto &ln = *iter;
    ln->getRunContext().setLossScale(scale);
  }
}

void NetworkGraph::applyCheckpointBlocks(
  const std::vector<CheckpointBlock> &checkpoint_blocks) {
  
  ml_logi("Applying %zu checkpoint blocks to the graph", checkpoint_blocks.size());
  
  for (const auto &block : checkpoint_blocks) {
    if (!block.isEnabled()) {
      continue;
    }
    
    const auto &user_layer_names = block.getLayerNames();
    const std::string &block_id = block.getBlockId();
    
    if (user_layer_names.size() < 2) {
      ml_logw("Checkpoint block '%s' has less than 2 layers, skipping",
              block_id.c_str());
      continue;
    }
    
    // Expand layer names to include auto-generated layers
    // Auto-generated layers have names like "{original_layer}/suffix"
    // We find all layers whose name starts with a user-specified layer name + "/"
    std::vector<std::string> layer_names;
    
    for (const auto &user_name : user_layer_names) {
      // Add the user-specified layer
      layer_names.push_back(user_name);
      
      // Find all auto-generated layers for this user layer
      // These have names like "{user_name}/activation_realized", "{user_name}/generated_out_0", etc.
      std::string prefix = user_name + "/";
      for (auto iter = cbegin(); iter != cend(); ++iter) {
        auto &node = *iter;
        const std::string &node_name = node->getName();
        // Check if this layer's name starts with the user layer's name + "/"
        if (node_name.size() > prefix.size() && 
            node_name.compare(0, prefix.size(), prefix) == 0) {
          layer_names.push_back(node_name);
          ml_logi("  Auto-including '%s' in checkpoint block (child of '%s')", 
                  node_name.c_str(), user_name.c_str());
        }
      }
    }
    
    ml_logi("  Expanded checkpoint block from %zu user layers to %zu total layers",
            user_layer_names.size(), layer_names.size());
    
    ml_logi("Processing checkpoint block '%s' with %zu layers (expanded from %zu)", 
            block_id.c_str(), layer_names.size(), user_layer_names.size());
    
    for (size_t i = 0; i < layer_names.size(); ++i) {
      const auto &layer_name = layer_names[i];
      
      try {
        auto layer_node = getLayerNode(layer_name);
        
        // All layers in the block are checkpointed
        layer_node->setCheckpointed(true);
        layer_node->setCheckpointBlockId(block_id);
        
        // Mark first layer in the block
        if (i == 0) {
          layer_node->setFirstInCheckpointBlock(true);
          ml_logd("  Layer '%s' marked as first in checkpoint block", layer_name.c_str());
        } else {
          layer_node->setFirstInCheckpointBlock(false);
          ml_logd("  Layer '%s' marked as checkpointed (not first)", 
                  layer_name.c_str());
        }

        if (i == layer_names.size() - 1){
          layer_node->setLastInCheckpointBlock(true);
        } else{
          layer_node->setLastInCheckpointBlock(false);
        }
        
        // Disable in-place optimization for checkpointed layers
        if (layer_node->getInPlaceType() != InPlaceType::NONE) {
          ml_logd("  Disabling in-place optimization for checkpointed layer '%s'", 
                  layer_name.c_str());
          layer_node->setInPlaceType(InPlaceType::NONE);
        }
        
      } catch (const std::exception &e) {
        ml_loge("Failed to apply checkpoint block to layer '%s': %s", 
                layer_name.c_str(), e.what());
        throw;
      }
    }
  }
  
  ml_logi("Successfully applied checkpoint blocks");
}

void NetworkGraph::recomputeCheckpointBlock(const std::string &block_id) {
  printf("Recomputing checkpoint block '%s'\n", block_id.c_str());
  
  // Collect all layers in this checkpoint block (including boundary)
  std::vector<std::shared_ptr<LayerNode>> all_block_layers;
  
  for (auto iter = cbegin(); iter != cend(); ++iter) {
    auto &node = *iter;
    if (node->getCheckpointBlockId() == block_id) {
      all_block_layers.push_back(node);
    }
  }
  
  if (all_block_layers.empty()) {
    printf("No layers found for checkpoint block '%s'\n", block_id.c_str());
    return;
  }
  
  // Filter: only recompute checkpointed layers (exclude first boundary layer)
  std::vector<std::shared_ptr<LayerNode>> layers_to_recompute;
  for (auto &layer : all_block_layers) {
    if (layer->isCheckpointed()) {
      layers_to_recompute.push_back(layer);
    } else {
      printf("Skipping boundary layer '%s' (output already exists)\n",
              layer->getName().c_str());
    }
  }
  
  if (layers_to_recompute.empty()) {
    printf("No layers to recompute in block '%s' (only boundary layer)\n",
            block_id.c_str());
    return;
  }
  
  printf("Recomputing %zu layers in block '%s' (skipping %zu boundary layers)\n", 
          layers_to_recompute.size(), block_id.c_str(),
          all_block_layers.size() - layers_to_recompute.size());
  
  for (size_t i = 0; i < layers_to_recompute.size(); ++i) {
    auto &layer = layers_to_recompute[i];
    printf("[DEBUG] Recomputing layer '%s' (first_in_block=%d)\n", 
           layer->getName().c_str(), layer->isFirstInCheckpointBlock());
    
    // Debug: print is_initial_forward and first input value for first checkpoint layer
    if (layer->isFirstInCheckpointBlock() && layer->getNumInputs() > 0) {
      auto &rc_debug = layer->getRunContext();
      fprintf(stderr, "[DEBUG-RECOMPUTE] First checkpoint layer '%s' is_initial_forward=%d\n",
              layer->getName().c_str(), rc_debug.isInitialForward());
      Tensor &input = layer->getInput(0);
      if (input.getData() != nullptr && input.size() > 0) {
        fprintf(stderr, "[DEBUG-RECOMPUTE] First checkpoint layer '%s' input[0][0]=%.6f\n",
                layer->getName().c_str(), input.getData()[0]);
      }
    }
    
    // During recompute, all layers should use normal forward mode (is_initial_forward = false)
    // - First layer: temporarily uses initial_forward to access initial_inputs (boundary outputs)
    // - Other layers: uses inputs (which point to previous layer's recompute outputs)
    // DO NOT set is_initial_forward=true for non-first layers, as this would use initial_tensors
    // which are not available during recompute phase
    
    // Get run context once for all operations
    auto &rc = layer->getRunContext();
    
    // Verify inputs, tensors, and weights before recompute if verification is enabled
    if (checkpoint_verification_enabled) {
      // Verify inputs
      std::vector<Tensor> recomputed_inputs;
      for (unsigned int j = 0; j < layer->getNumInputs(); ++j) {
        recomputed_inputs.push_back(layer->getInput(j));
      }
      verifyRecomputedInputs(layer->getName(), recomputed_inputs);
      
      // Verify weights (should be identical)
      std::vector<Tensor> recomputed_weights;
      for (unsigned int j = 0; j < rc.getNumWeights(); ++j) {
        recomputed_weights.push_back(rc.getWeight(j));
      }
      verifyRecomputedWeights(layer->getName(), recomputed_weights);
    }
    
    // Reset ITERATION_LIFESPAN tensors before recompute to match initial forward state
    // This is critical for layers like MHA that have internal state tensors
    printf("[DEBUG] Resetting %u tensors for layer '%s' before recompute\n", 
           rc.getNumTensors(), layer->getName().c_str());
    for (unsigned int j = 0; j < rc.getNumTensors(); ++j) {
      Tensor &tensor = rc.getTensor(j);
      std::string tensor_name = tensor.getName();
      
      printf("[DEBUG] Zeroing tensor[%u] '%s' (size=%zu)\n", 
             j, tensor_name.c_str(), tensor.size());
      // Zero out the tensor to ensure clean state
      tensor.setZero();
    }
    
    layer->forwarding(true);
    
    // Verify recomputed outputs and tensors if verification is enabled
    if (checkpoint_verification_enabled) {
      auto &rc = layer->getRunContext();
      
      // Verify outputs
      std::vector<Tensor> recomputed_outputs;
      for (unsigned int j = 0; j < layer->getNumOutputs(); ++j) {
        recomputed_outputs.push_back(layer->getOutput(j));
      }
      verifyRecomputedOutputs(layer->getName(), recomputed_outputs);
      
      // Verify tensors (intermediate tensors)
      std::vector<Tensor> recomputed_tensors;
      for (unsigned int j = 0; j < rc.getNumTensors(); ++j) {
        recomputed_tensors.push_back(rc.getTensor(j));
      }
      // Get forward tensor indices from the layer (empty means verify all)
      auto forward_tensor_indices = layer->getForwardTensorIndices();
      verifyRecomputedTensors(layer->getName(), recomputed_tensors, forward_tensor_indices);
    }
  }
  
  printf("[DEBUG] Completed recomputation for block '%s'\n", block_id.c_str());
}

void NetworkGraph::enableCheckpointVerification(bool enable) {
  checkpoint_verification_enabled = enable;
  if (enable) {
    printf("[INFO] Gradient checkpointing verification enabled\n");
    verification_stats = VerificationStats();
  } else {
    printf("[INFO] Gradient checkpointing verification disabled\n");
  }
}

void NetworkGraph::saveForwardInputs(const std::string &layer_name,
                                     const std::vector<Tensor> &inputs) {
  if (!checkpoint_verification_enabled) {
    return;
  }
  
  // Deep copy tensors to save
  std::vector<Tensor> saved_inputs;
  saved_inputs.reserve(inputs.size());
  
  for (const auto &input : inputs) {
    Tensor copied_tensor = input.clone();
    saved_inputs.push_back(copied_tensor);
  }
  
  saved_forward_inputs[layer_name] = saved_inputs;
  printf("[DEBUG] Saved forward inputs for layer '%s' (%zu tensors)\n", 
         layer_name.c_str(), inputs.size());
}

void NetworkGraph::saveForwardOutputs(const std::string &layer_name,
                                      const std::vector<Tensor> &outputs) {
  if (!checkpoint_verification_enabled) {
    return;
  }
  
  // Deep copy tensors to save
  std::vector<Tensor> saved_outputs;
  saved_outputs.reserve(outputs.size());
  
  for (const auto &output : outputs) {
    Tensor copied_tensor = output.clone();
    saved_outputs.push_back(copied_tensor);
  }
  
  saved_forward_outputs[layer_name] = saved_outputs;
  printf("[DEBUG] Saved forward outputs for layer '%s' (%zu tensors)\n", 
         layer_name.c_str(), outputs.size());
}

void NetworkGraph::saveForwardTensors(const std::string &layer_name,
                                      const std::vector<Tensor> &tensors) {
  if (!checkpoint_verification_enabled) {
    return;
  }
  
  // Deep copy tensors to save
  std::vector<Tensor> saved_tensors;
  saved_tensors.reserve(tensors.size());
  
  for (const auto &tensor : tensors) {
    // Skip stateful tensors like KV cache that accumulate across iterations
    std::string tensor_name = tensor.getName();
    if (tensor_name.find("cache_key") != std::string::npos ||
        tensor_name.find("cache_value") != std::string::npos) {
      printf("[DEBUG] Skipping stateful tensor '%s' for verification\n", tensor_name.c_str());
      continue;
    }
    
    Tensor copied_tensor = tensor.clone();
    saved_tensors.push_back(copied_tensor);
  }
  
  saved_forward_tensors[layer_name] = saved_tensors;
  printf("[DEBUG] Saved forward tensors for layer '%s' (%zu tensors)\n", 
         layer_name.c_str(), tensors.size());
}

void NetworkGraph::saveForwardWeights(const std::string &layer_name,
                                      const std::vector<Tensor> &weights) {
  if (!checkpoint_verification_enabled) {
    return;
  }
  
  // Deep copy weights to save
  std::vector<Tensor> saved_weights;
  saved_weights.reserve(weights.size());
  
  for (const auto &weight : weights) {
    Tensor copied_tensor = weight.clone();
    saved_weights.push_back(copied_tensor);
  }
  
  saved_forward_weights[layer_name] = saved_weights;
  printf("[DEBUG] Saved forward weights for layer '%s' (%zu weights)\n", 
         layer_name.c_str(), weights.size());
}

bool NetworkGraph::verifyRecomputedInputs(const std::string &layer_name,
                                          const std::vector<Tensor> &recomputed_inputs) {
  printf("Verifying recomputed inputs for layer '%s'\n", layer_name.c_str());
  if (!checkpoint_verification_enabled) {
    return true;
  }
  
  auto it = saved_forward_inputs.find(layer_name);
  if (it == saved_forward_inputs.end()) {
    printf("[WARNING] No saved inputs found for layer '%s'\n", layer_name.c_str());
    return false;
  }
  
  const auto &saved_inputs = it->second;
  
  if (saved_inputs.size() != recomputed_inputs.size()) {
    printf("[ERROR] Input count mismatch for layer '%s': saved=%zu, recomputed=%zu\n",
           layer_name.c_str(), saved_inputs.size(), recomputed_inputs.size());
    return false;
  }
  
  bool all_match = true;
  
  for (size_t i = 0; i < saved_inputs.size(); ++i) {
    const auto &saved = saved_inputs[i];
    const auto &recomputed = recomputed_inputs[i];
    
    // Check dimensions
    if (saved.getDim() != recomputed.getDim()) {
      std::ostringstream saved_dim_str, recomputed_dim_str;
      saved_dim_str << saved.getDim();
      recomputed_dim_str << recomputed.getDim();
      printf("[ERROR] Input dimension mismatch for layer '%s' input[%zu]: saved=%s, recomputed=%s\n",
             layer_name.c_str(), i, 
             saved_dim_str.str().c_str(),
             recomputed_dim_str.str().c_str());
      all_match = false;
      continue;
    }
    
    // Compute element-wise difference
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    size_t element_count = saved.size();
    
    const float *saved_data = saved.getData();
    const float *recomputed_data = recomputed.getData();
    
    for (size_t j = 0; j < element_count; ++j) {
      float diff = std::abs(saved_data[j] - recomputed_data[j]);
      sum_diff += diff;
      if (diff > max_diff) {
        max_diff = diff;
      }
    }
    
    float avg_diff = sum_diff / element_count;
    
    // Tolerance check (relative and absolute)
    const float rel_tolerance = 1e-5f;
    const float abs_tolerance = 1e-6f;
    
    bool matches = true;
    for (size_t j = 0; j < element_count; ++j) {
      float diff = std::abs(saved_data[j] - recomputed_data[j]);
      float threshold = abs_tolerance + rel_tolerance * std::abs(saved_data[j]);
      if (diff > threshold) {
        matches = false;
        break;
      }
    }
    
    if (!matches) {
      printf("[ERROR] Input mismatch for layer '%s' input[%zu] '%s': max_diff=%.10f, avg_diff=%.10f\n",
             layer_name.c_str(), i, saved.getName().c_str(), max_diff, avg_diff);
      all_match = false;
    } else {
      printf("[DEBUG] Input match for layer '%s' input[%zu] '%s': max_diff=%.10f, avg_diff=%.10f\n",
             layer_name.c_str(), i, saved.getName().c_str(), max_diff, avg_diff);
    }
  }
  
  if (all_match) {
    printf("[INFO] ✓ Input verification PASSED for layer '%s'\n", layer_name.c_str());
  } else {
    printf("[ERROR] ✗ Input verification FAILED for layer '%s'\n", layer_name.c_str());
  }
  
  return all_match;
}

bool NetworkGraph::verifyRecomputedOutputs(const std::string &layer_name,
                                           const std::vector<Tensor> &recomputed_outputs) {
  printf("Verifying recomputed outputs for layer '%s'\n", layer_name.c_str());
  if (!checkpoint_verification_enabled) {
    return true;
  }
  
  auto it = saved_forward_outputs.find(layer_name);
  if (it == saved_forward_outputs.end()) {
    printf("[WARNING] No saved outputs found for layer '%s'\n", layer_name.c_str());
    return false;
  }
  
  const auto &saved_outputs = it->second;
  
  if (saved_outputs.size() != recomputed_outputs.size()) {
    printf("[ERROR] Output count mismatch for layer '%s': saved=%zu, recomputed=%zu\n",
           layer_name.c_str(), saved_outputs.size(), recomputed_outputs.size());
    verification_stats.failed_verifications++;
    verification_stats.total_verifications++;
    return false;
  }
  
  bool all_match = true;
  float total_diff = 0.0f;
  int diff_count = 0;
  
  for (size_t i = 0; i < saved_outputs.size(); ++i) {
    const auto &saved = saved_outputs[i];
    const auto &recomputed = recomputed_outputs[i];
    
    // Check dimensions
    if (saved.getDim() != recomputed.getDim()) {
      std::ostringstream saved_dim_str, recomputed_dim_str;
      saved_dim_str << saved.getDim();
      recomputed_dim_str << recomputed.getDim();
      printf("[ERROR] Dimension mismatch for layer '%s' output[%zu]: saved=%s, recomputed=%s\n",
             layer_name.c_str(), i, 
             saved_dim_str.str().c_str(),
             recomputed_dim_str.str().c_str());
      all_match = false;
      continue;
    }
    
    // Compute element-wise difference
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    size_t element_count = saved.size();
    
    const float *saved_data = saved.getData();
    const float *recomputed_data = recomputed.getData();
    
    for (size_t j = 0; j < element_count; ++j) {
      float diff = std::abs(saved_data[j] - recomputed_data[j]);
      sum_diff += diff;
      if (diff > max_diff) {
        max_diff = diff;
      }
    }
    
    float avg_diff = sum_diff / element_count;
    total_diff += sum_diff;
    diff_count += element_count;
    
    // Tolerance check (relative and absolute)
    const float rel_tolerance = 1e-5f;
    const float abs_tolerance = 1e-6f;
    
    bool matches = true;
    for (size_t j = 0; j < element_count; ++j) {
      float diff = std::abs(saved_data[j] - recomputed_data[j]);
      float threshold = abs_tolerance + rel_tolerance * std::abs(saved_data[j]);
      if (diff > threshold) {
        matches = false;
        break;
      }
    }
    
    if (max_diff > verification_stats.max_diff) {
      verification_stats.max_diff = max_diff;
    }
    
    if (!matches) {
      printf("[ERROR] Output mismatch for layer '%s' output[%zu] '%s': max_diff=%.10f, avg_diff=%.10f\n",
             layer_name.c_str(), i, saved.getName().c_str(), max_diff, avg_diff);
      all_match = false;
    } else {
      printf("[DEBUG] Output match for layer '%s' output[%zu] '%s': max_diff=%.10f, avg_diff=%.10f\n",
             layer_name.c_str(), i, saved.getName().c_str(), max_diff, avg_diff);
    }
  }
  
  if (diff_count > 0) {
    verification_stats.avg_diff = total_diff / diff_count;
  }
  
  verification_stats.total_verifications++;
  if (all_match) {
    verification_stats.passed_verifications++;
    printf("[INFO] ✓ Verification PASSED for layer '%s'\n", layer_name.c_str());
  } else {
    verification_stats.failed_verifications++;
    printf("[ERROR] ✗ Verification FAILED for layer '%s'\n", layer_name.c_str());
  }
  
  return all_match;
}

bool NetworkGraph::verifyRecomputedTensors(const std::string &layer_name,
                                           const std::vector<Tensor> &recomputed_tensors,
                                           const std::vector<unsigned int> &forward_tensor_indices) {
  printf("Verifying recomputed tensors for layer '%s'\n", layer_name.c_str());
  if (!checkpoint_verification_enabled) {
    return true;
  }
  
  auto it = saved_forward_tensors.find(layer_name);
  if (it == saved_forward_tensors.end()) {
    printf("[WARNING] No saved tensors found for layer '%s'\n", layer_name.c_str());
    return false;
  }
  
  const auto &saved_tensors = it->second;
  
  // Filter out stateful tensors from recomputed tensors
  std::vector<Tensor> filtered_recomputed;
  std::vector<size_t> original_indices; // Track original indices for forward_tensor_indices check
  for (size_t idx = 0; idx < recomputed_tensors.size(); ++idx) {
    const auto &tensor = recomputed_tensors[idx];
    std::string tensor_name = tensor.getName();
    if (tensor_name.find("cache_key") != std::string::npos ||
        tensor_name.find("cache_value") != std::string::npos) {
      continue;
    }
    filtered_recomputed.push_back(tensor);
    original_indices.push_back(idx);
  }
  
  if (saved_tensors.size() != filtered_recomputed.size()) {
    printf("[ERROR] Tensor count mismatch for layer '%s': saved=%zu, recomputed=%zu\n",
           layer_name.c_str(), saved_tensors.size(), filtered_recomputed.size());
    return false;
  }
  
  // Determine which tensors to verify
  std::set<unsigned int> indices_to_verify;
  if (forward_tensor_indices.empty()) {
    // Verify all tensors
    for (size_t i = 0; i < saved_tensors.size(); ++i) {
      indices_to_verify.insert(static_cast<unsigned int>(i));
    }
  } else {
    // Only verify specified forward tensor indices
    for (auto idx : forward_tensor_indices) {
      indices_to_verify.insert(idx);
    }
    printf("[INFO] Layer '%s' - only verifying forward tensors: [", layer_name.c_str());
    for (auto idx : forward_tensor_indices) {
      printf("%u ", idx);
    }
    printf("]\n");
  }
  
  bool all_match = true;
  
  for (size_t i = 0; i < saved_tensors.size(); ++i) {
    // Skip tensors not in forward_tensor_indices (if specified)
    if (!indices_to_verify.empty() && 
        indices_to_verify.find(static_cast<unsigned int>(i)) == indices_to_verify.end()) {
      printf("[INFO] Skipping tensor[%zu] '%s' - not used in forward pass\n",
             i, saved_tensors[i].getName().c_str());
      continue;
    }
    
    const auto &saved = saved_tensors[i];
    const auto &recomputed = filtered_recomputed[i];
    
    // Check dimensions
    if (saved.getDim() != recomputed.getDim()) {
      std::ostringstream saved_dim_str, recomputed_dim_str;
      saved_dim_str << saved.getDim();
      recomputed_dim_str << recomputed.getDim();
      printf("[ERROR] Tensor dimension mismatch for layer '%s' tensor[%zu] '%s': saved=%s, recomputed=%s\n",
             layer_name.c_str(), i, saved.getName().c_str(),
             saved_dim_str.str().c_str(),
             recomputed_dim_str.str().c_str());
      all_match = false;
      continue;
    }
    
    // Compute element-wise difference
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    size_t element_count = saved.size();
    
    const float *saved_data = saved.getData();
    const float *recomputed_data = recomputed.getData();
    
    for (size_t j = 0; j < element_count; ++j) {
      float diff = std::abs(saved_data[j] - recomputed_data[j]);
      sum_diff += diff;
      if (diff > max_diff) {
        max_diff = diff;
      }
    }
    
    float avg_diff = sum_diff / element_count;
    
    // Tolerance check (relative and absolute)
    const float rel_tolerance = 1e-5f;
    const float abs_tolerance = 1e-6f;
    
    bool matches = true;
    for (size_t j = 0; j < element_count; ++j) {
      float diff = std::abs(saved_data[j] - recomputed_data[j]);
      float threshold = abs_tolerance + rel_tolerance * std::abs(saved_data[j]);
      if (diff > threshold) {
        matches = false;
        break;
      }
    }
    
    if (!matches) {
      printf("[ERROR] Tensor mismatch for layer '%s' tensor[%zu] '%s': max_diff=%.10f, avg_diff=%.10f\n",
             layer_name.c_str(), i, saved.getName().c_str(), max_diff, avg_diff);
      all_match = false;
    } else {
      printf("[DEBUG] Tensor match for layer '%s' tensor[%zu] '%s': max_diff=%.10f, avg_diff=%.10f\n",
             layer_name.c_str(), i, saved.getName().c_str(), max_diff, avg_diff);
    }
  }
  
  if (all_match) {
    printf("[INFO] ✓ Tensor verification PASSED for layer '%s'\n", layer_name.c_str());
  } else {
    printf("[ERROR] ✗ Tensor verification FAILED for layer '%s'\n", layer_name.c_str());
  }
  
  return all_match;
}

bool NetworkGraph::verifyRecomputedWeights(const std::string &layer_name,
                                           const std::vector<Tensor> &recomputed_weights) {
  printf("Verifying recomputed weights for layer '%s'\n", layer_name.c_str());
  if (!checkpoint_verification_enabled) {
    return true;
  }
  
  auto it = saved_forward_weights.find(layer_name);
  if (it == saved_forward_weights.end()) {
    printf("[WARNING] No saved weights found for layer '%s'\n", layer_name.c_str());
    return false;
  }
  
  const auto &saved_weights = it->second;
  
  if (saved_weights.size() != recomputed_weights.size()) {
    printf("[ERROR] Weight count mismatch for layer '%s': saved=%zu, recomputed=%zu\n",
           layer_name.c_str(), saved_weights.size(), recomputed_weights.size());
    return false;
  }
  
  bool all_match = true;
  
  for (size_t i = 0; i < saved_weights.size(); ++i) {
    const auto &saved = saved_weights[i];
    const auto &recomputed = recomputed_weights[i];
    
    // Check dimensions
    if (saved.getDim() != recomputed.getDim()) {
      std::ostringstream saved_dim_str, recomputed_dim_str;
      saved_dim_str << saved.getDim();
      recomputed_dim_str << recomputed.getDim();
      printf("[ERROR] Weight dimension mismatch for layer '%s' weight[%zu] '%s': saved=%s, recomputed=%s\n",
             layer_name.c_str(), i, saved.getName().c_str(),
             saved_dim_str.str().c_str(),
             recomputed_dim_str.str().c_str());
      all_match = false;
      continue;
    }
    
    // Compute element-wise difference
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    size_t element_count = saved.size();
    
    const float *saved_data = saved.getData();
    const float *recomputed_data = recomputed.getData();
    
    for (size_t j = 0; j < element_count; ++j) {
      float diff = std::abs(saved_data[j] - recomputed_data[j]);
      sum_diff += diff;
      if (diff > max_diff) {
        max_diff = diff;
      }
    }
    
    float avg_diff = sum_diff / element_count;
    
    // Weights should be exactly identical (no tolerance)
    bool matches = (max_diff == 0.0f);
    
    if (!matches) {
      printf("[ERROR] Weight mismatch for layer '%s' weight[%zu] '%s': max_diff=%.10f, avg_diff=%.10f\n",
             layer_name.c_str(), i, saved.getName().c_str(), max_diff, avg_diff);
      all_match = false;
    } else {
      printf("[DEBUG] Weight match for layer '%s' weight[%zu] '%s': EXACT MATCH\n",
             layer_name.c_str(), i, saved.getName().c_str());
    }
  }
  
  if (all_match) {
    printf("[INFO] ✓ Weight verification PASSED for layer '%s'\n", layer_name.c_str());
  } else {
    printf("[ERROR] ✗ Weight verification FAILED for layer '%s'\n", layer_name.c_str());
  }
  
  return all_match;
}

void NetworkGraph::clearSavedOutputs() {
  saved_forward_inputs.clear();
  saved_forward_outputs.clear();
  saved_forward_tensors.clear();
  saved_forward_weights.clear();
  printf("[DEBUG] Cleared all saved forward inputs, outputs, tensors, and weights\n");
}

void NetworkGraph::printVerificationStats() {
  printf("==========================================================================\n");
  printf("          Gradient Checkpointing Verification Statistics\n");
  printf("==========================================================================\n");
  printf("Total verifications: %u\n", verification_stats.total_verifications);
  printf("Passed: %u\n", verification_stats.passed_verifications);
  printf("Failed: %u\n", verification_stats.failed_verifications);
  
  if (verification_stats.total_verifications > 0) {
    float pass_rate = (float)verification_stats.passed_verifications / 
                      verification_stats.total_verifications * 100.0f;
    printf("Pass rate: %.2f%%\n", pass_rate);
  }
  
  printf("Max difference: %.10f\n", verification_stats.max_diff);
  printf("Avg difference: %.10f\n", verification_stats.avg_diff);
  printf("==========================================================================\n");
}

void NetworkGraph::enableTensorDump(bool enable, const std::string &path) {
  tensor_dump_enabled = enable;
  tensor_dump_path = path;
  tensor_dump_iteration = 0;
  
  if (enable) {
    // Create dump directory if it doesn't exist
    std::string mkdir_cmd = "mkdir -p " + path;
    int ret = system(mkdir_cmd.c_str());
    if (ret != 0) {
      printf("[WARNING] Failed to create tensor dump directory: %s\n", path.c_str());
    }
    printf("[INFO] Tensor dump enabled. Output path: %s\n", path.c_str());
  } else {
    printf("[INFO] Tensor dump disabled\n");
  }
}

void NetworkGraph::setTensorDumpIteration(unsigned int iteration) {
  tensor_dump_iteration = iteration;
}

void NetworkGraph::dumpTensor(const std::string &layer_name, 
                               const std::string &tensor_type,
                               unsigned int index, 
                               const Tensor &tensor) {
  if (!tensor_dump_enabled) {
    return;
  }
  
  // Create filename: iter{N}_{layer_name}_{tensor_type}_{index}.txt
  std::string safe_layer_name = layer_name;
  // Replace '/' with '_' in layer name for valid filename
  std::replace(safe_layer_name.begin(), safe_layer_name.end(), '/', '_');
  
  std::ostringstream filename;
  filename << tensor_dump_path << "/iter" << tensor_dump_iteration 
           << "_" << safe_layer_name 
           << "_" << tensor_type 
           << "_" << index << ".txt";
  
  std::ofstream file(filename.str());
  if (!file.is_open()) {
    printf("[ERROR] Failed to open file for tensor dump: %s\n", filename.str().c_str());
    return;
  }
  
  // Write tensor metadata
  file << "# Layer: " << layer_name << "\n";
  file << "# Type: " << tensor_type << "\n";
  file << "# Index: " << index << "\n";
  file << "# Iteration: " << tensor_dump_iteration << "\n";
  file << "# Shape: " << tensor.getDim() << "\n";
  file << "# Size: " << tensor.size() << "\n";
  
  // Compute statistics
  float sum = 0.0f, min_val = std::numeric_limits<float>::max();
  float max_val = std::numeric_limits<float>::lowest();
  
  const float *data = tensor.getData<float>();
  size_t size = tensor.size();
  
  for (size_t i = 0; i < size; ++i) {
    float val = data[i];
    sum += val;
    if (val < min_val) min_val = val;
    if (val > max_val) max_val = val;
  }
  
  float mean = size > 0 ? sum / size : 0.0f;
  
  file << "# Min: " << std::setprecision(10) << min_val << "\n";
  file << "# Max: " << std::setprecision(10) << max_val << "\n";
  file << "# Mean: " << std::setprecision(10) << mean << "\n";
  file << "# Sum: " << std::setprecision(10) << sum << "\n";
  file << "#\n";
  
  // Write first 100 values for quick comparison
  file << "# First 100 values:\n";
  size_t num_to_print = std::min(size, (size_t)100);
  for (size_t i = 0; i < num_to_print; ++i) {
    file << std::setprecision(10) << data[i];
    if (i < num_to_print - 1) file << ", ";
    if ((i + 1) % 10 == 0) file << "\n";
  }
  file << "\n";
  
  file.close();
}

} /* namespace nntrainer */
