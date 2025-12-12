// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   layer_context.cpp
 * @date   26 July 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @author Debadri Samaddar <s.debadri@samsung.com>
 * @author Niket Agarwal <niket.a@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the layer context for each layer
 */

#include "nntrainer_error.h"
#include <functional>
#include <memory>
#include <tensor_wrap_specs.h>

#include <fstream>
#include <iterator>
#include <layer_context.h>
#include <nntrainer_log.h>
#include <stdexcept>
#include <var_grad.h>

namespace nntrainer {

/**
 * @brief rename specification
 *
 * @param spec spec to rename
 * @param idx idx to suffix
 */
static void suffixSpec(VarGradSpecV2 &spec, unsigned int idx) {
  spec.variable_spec.name += std::to_string(idx);
  if (spec.gradient_spec) {
    spec.gradient_spec->name += std::to_string(idx) + Var_Grad::grad_suffix;
  }
}

InitLayerContext::InitLayerContext(
  const std::vector<TensorDim> &dim, const std::vector<bool> &req_out_connected,
  bool is_inplace_, const std::string &n, const std::string &prefix_,
  const float max_norm, std::array<std::string, 3> tensor_type_,
  const float loss_scale_, ml::train::ExecutionMode mode_,
  ml::train::LayerComputeEngine engine_, bool is_checkpointed_) :
  input_dim(dim),
  is_inplace(is_inplace_),
  clip_by_global_norm(max_norm),
  output_specs(),
  req_out_is_connected(req_out_connected),
  name(n),
  prefix(prefix_),
  tensor_type(tensor_type_),
  loss_scale(loss_scale_),
  mode(mode_),
  engine(engine_),
  is_checkpointed(is_checkpointed_) {
  NNTR_THROW_IF(!validate(), std::invalid_argument)
    << "Invalid init context name: " << name
    << " num inputs: " << getNumInputs();
  if (prefix.empty())
    prefix = name; // default prefix is the name
}

unsigned int InitLayerContext::getNumRequestedOutputs() const {
  return req_out_is_connected.size();
}

void InitLayerContext::setOutputDimensions(
  const std::vector<TensorDim> &out_dim) {
  std::vector<VarGradSpecV2> specs;
  specs.reserve(out_dim.size());

  for (unsigned i = 0u, sz = out_dim.size(); i < sz; ++i) {
    auto spec = outSpec(out_dim.at(i));

    specs.push_back(std::move(spec));
  }

  requestOutputs(std::move(specs));
}

VarGradSpecV2 InitLayerContext::outSpec(const TensorDim &dim,
                                        const std::string &name,
                                        TensorLifespan ls,
                                        TensorLifespan grad_ls) {
  VarGradSpecV2 spec;
  spec.variable_spec.dim = dim;
  spec.variable_spec.name = name;
  spec.variable_spec.ls = ls;
  spec.gradient_spec = std::make_unique<TensorSpecV2>(spec.variable_spec);
  spec.gradient_spec->ls = grad_ls;

  return spec;
}

void InitLayerContext::requestOutputs(std::vector<VarGradSpecV2> &&out_specs) {
  NNTR_THROW_IF(out_specs.size() < getNumRequestedOutputs(),
                std::invalid_argument)
    << "number of output dimension set is smaller than the number of out "
       "tensor slots requested, num output specification: "
    << out_specs.size() << " slots to fill: " << getNumRequestedOutputs()
    << " context name: " << name;
  NNTR_THROW_IF(output_specs.size(), std::invalid_argument)
    << "output specification already set, cannot set twice. Check if output is "
       "already requested elsewhere";
  output_specs.reserve(out_specs.size());

  auto is_dangled = [this](unsigned int idx) {
    return req_out_is_connected.size() <= idx || !req_out_is_connected[idx];
  };

  for (unsigned i = 0u, sz = out_specs.size(); i < sz; ++i) {
    auto &spec = out_specs.at(i);
    suffixSpec(spec, i);
    if (is_dangled(i)) {
      ml_logw("given output is being dangled: %s in context: %s",
              spec.variable_spec.name.c_str(), name.c_str());
      spec.gradient_spec = nullptr;
    }
    output_specs.push_back(std::move(spec));
  }
}

const std::vector<VarGradSpecV2> &InitLayerContext::getOutSpecs() const {
  return output_specs;
}

RunLayerContext::RunLayerContext(const std::string &name, bool trainable,
                                 float l, bool is_inplace_, float loss_scale_,
                                 std::shared_ptr<ContextData> ct_data_,
                                 bool restore_, const std::vector<Weight *> &w,
                                 const std::vector<Var_Grad *> &in,
                                 const std::vector<Var_Grad *> &out,
                                 const std::vector<Var_Grad *> &t) :
  ct_data(ct_data_),
  loss(l),
  is_inplace(is_inplace_),
  loss_scale(loss_scale_),
  restoreData(restore_),
  weights(w),
  inputs(in),
  outputs(out),
  tensors(t) {
  std::get<props::Name>(props).set(name);
  std::get<props::Trainable>(props).set(trainable);
  NNTR_THROW_IF(!readyToUse(), std::invalid_argument)
    << "run context is not ready to use upon creation";

  if (!validate())
    throw std::invalid_argument("Creating invalid run context");
}

/**
 * @brief Get the Weight tensor object
 *
 * @param idx Identifier of the weight
 * @return Tensor& Reference to the weight tensor
 */
Tensor &RunLayerContext::getWeight(unsigned int idx) const {
  return weights[idx]->getVariableRef();
}

/**
 * @brief Get the Weight Gradient tensor object
 *
 * @param idx Identifier of the weight
 * @return Tensor& Reference to the weight grad tensor
 */
Tensor &RunLayerContext::getWeightGrad(unsigned int idx) const {
  if (!weights[idx]->hasGradient())
    throw std::invalid_argument(
      "Requesting gradient for a non-trainable weight.");
  return weights[idx]->getGradientRef();
}

/**
 * @brief Get the Weight Gradient tensor object
 *
 * @param idx Identifier of the weight
 * @return Tensor& Reference to the weight grad tensor
 */
Tensor &RunLayerContext::getWeightFP32(unsigned int idx) const {
  return weights[idx]->getVariableFP32Ref();
}

/**
 * @brief Get the Weight Optimizer Variable tensor object
 *
 * @param idx Identifier of the weight
 * @param jdx Identifier of the optimizer variables
 * @return Tensor& Reference to the weight optimizer variable tensor
 */
Tensor &RunLayerContext::getWeightOptVar(unsigned int idx,
                                         unsigned int jdx) const {
  return weights[idx]->getOptimizerVariableRef(jdx);
}

/**
 * @brief Get the Number of Weight Optimizer Variable tensor object
 *
 * @param idx Identifier of the weight
 * @return int Number of the weight optimizer variable
 */
unsigned int RunLayerContext::getNumWeightOptVar(unsigned int idx) const {
  return weights[idx]->getNumOptVariable();
}

/**
 * @brief Get regularization loss for the weight
 *
 * @param idx Identifier of the weight
 * @return float Value of the loss
 */
float RunLayerContext::getWeightRegularizationLoss(unsigned int idx) const {
  return weights[idx]->getRegularizationLoss();
}

/**
 * @brief Get the Weight name
 *
 * @param idx Identifier of the weight
 * @return name of the weight
 */
const std::string &RunLayerContext::getWeightName(unsigned int idx) const {
  return weights[idx]->getName();
}

/**
 * @brief check if the weight has gradient
 *
 * @param idx Identifier of the weight
 * @return true if weight has gradient, else false
 */
bool RunLayerContext::weightHasGradient(unsigned int idx) const {
  return weights[idx]->hasGradient();
}

/**
 * @brief Get the Output tensor object
 *
 * @param idx Identifier of the output
 * @return Tensor& Reference to the output tensor
 * @note For gradient checkpointing:
 *   - Initial forward: returns initial_outputs (short-lived)
 *   - Recompute forward: returns outputs (default)
 *   - Backward: returns outputs (default)
 */
Tensor &RunLayerContext::getOutput(unsigned int idx) {
  // For in-place layers during initial forward, return input instead of output
  // This takes priority over checkpointed initial_outputs for in-place layers
  // For multi-output in-place layers, all outputs share initial_inputs[0]
  if (is_initial_forward){
    auto &tensor = initial_outputs[idx]->getVariableRef();
    // Also show what the normal input would be for comparison
    auto &normal_tensor = outputs[idx]->getVariableRef();
    printf("[Initial Forward_getOutput] Layer: %s, idx: %u\n", std::get<props::Name>(props).get().c_str(), idx);
    printf("  -> Using INITIAL_OUTPUT: name='%s', ptr=%p, data_ptr=%p\n",
            tensor.getName().c_str(), (void *)&tensor, (void *)tensor.getData());
    printf("  -> Normal output would be: name='%s', ptr=%p, data_ptr=%p\n",
            normal_tensor.getName().c_str(), (void *)&normal_tensor, (void *)normal_tensor.getData());
  }
  if (is_initial_forward && is_inplace && !initial_inputs.empty() && initial_inputs[0] != nullptr) {
    auto& tensor = initial_inputs[0]->getVariableRef();
    printf("[Initial Forward_getOutput][IN-PLACE] Layer name: %s, idx: %d (using initial_inputs[0]), tensor_name: %s, tensor_ptr: %p, data_ptr: %p\n", 
           std::get<props::Name>(props).get().c_str(), idx, tensor.getName().c_str(), (void*)&tensor, (void*)tensor.getData());
    return tensor;
  }
  if (is_checkpointed && is_initial_forward && !initial_outputs.empty()) {
    auto& tensor = initial_outputs[idx]->getVariableRef();
    printf("[Initial Forward_getOutput] Layer name: %s, idx: %d, tensor_name: %s, tensor_ptr: %p, data_ptr: %p\n", 
           std::get<props::Name>(props).get().c_str(), idx, tensor.getName().c_str(), (void*)&tensor, (void*)tensor.getData());
    return tensor;
  }
  auto& tensor = outputs[idx]->getVariableRef();
  printf("[Normal OR Recompute Forward_getOutput] Layer name: %s, idx: %d, tensor_name: %s, tensor_ptr: %p, data_ptr: %p\n", 
         std::get<props::Name>(props).get().c_str(), idx, tensor.getName().c_str(), (void*)&tensor, (void*)tensor.getData());
  
  // Debug: Print first few values for embedding layer to track value changes
  std::string layer_name = std::get<props::Name>(props).get();
  if (layer_name == "embedding" && tensor.getData() != nullptr && tensor.size() > 0) {
    printf("[EMBEDDING-OUTPUT-VALUES] embedding output[0..2] = [%f, %f, %f]\n",
           tensor.getData()[0], tensor.getData()[1], tensor.getData()[2]);
  }
  
  return tensor;
}

const Tensor &RunLayerContext::getOutput(unsigned int idx) const {
  // For in-place layers during initial forward, return input instead of output
  // This takes priority over checkpointed initial_outputs for in-place layers
  // For multi-output in-place layers, all outputs share initial_inputs[0]
  if (is_initial_forward && is_inplace && !initial_inputs.empty() && initial_inputs[0] != nullptr) {
    auto& tensor = initial_inputs[0]->getVariableRef();
    printf("[Initial Forward_getOutput][IN-PLACE] Layer name: %s, idx: %d (using initial_inputs[0]), tensor_name: %s, tensor_ptr: %p, data_ptr: %p\n", 
           std::get<props::Name>(props).get().c_str(), idx, tensor.getName().c_str(), (void*)&tensor, (void*)tensor.getData());
    return tensor;
  }
  if (is_checkpointed && is_initial_forward && !initial_outputs.empty()) {
    auto& tensor = initial_outputs[idx]->getVariableRef();
    printf("[Initial Forward_getOutput] Layer name: %s, idx: %d, tensor_name: %s, tensor_ptr: %p, data_ptr: %p\n", 
           std::get<props::Name>(props).get().c_str(), idx, tensor.getName().c_str(), (void*)&tensor, (void*)tensor.getData());
    return tensor;
  }
  auto& tensor = outputs[idx]->getVariableRef();
  printf("[Normal OR Recompute Forward_getOutput] Layer name: %s, idx: %d, tensor_name: %s, tensor_ptr: %p, data_ptr: %p\n", 
         std::get<props::Name>(props).get().c_str(), idx, tensor.getName().c_str(), (void*)&tensor, (void*)tensor.getData());
  return tensor;
}

/**
 * @brief Get the Output Grad tensor object
 *
 * @param idx Identifier of the output
 * @return Tensor Read-only output grad tensor
 */
const Tensor RunLayerContext::getOutputGrad(unsigned int idx) const {
  if (!outputs[idx]->hasGradient()) {
    return Tensor(outputs[idx]->getDim(), true, Initializer::ZEROS);
  }
  return const_cast<RunLayerContext *>(this)->getOutputGradUnsafe(idx);
}

/**
 * @brief check if the output has gradient
 *
 * @param idx Identifier of the output
 * @return true if output has gradient, else false
 */
bool RunLayerContext::outputHasGradient(unsigned int idx) const {
  return outputs[idx]->hasGradient();
}

/**
 * @brief Get the Output Grad tensor object
 *
 * @param idx Identifier of the output
 * @return Tensor& Reference to the output grad tensor
 *
 * @note recommended to NOT use this function as a layer developer but rather
 * use getOutputGrad().
 * @note For gradient checkpointing:
 *   - Backward phase: returns outputs gradient (from next layer's backprop)
 */
Tensor &RunLayerContext::getOutputGradUnsafe(unsigned int idx) {
  return outputs[idx]->getGradientRef();
}

/**
 * @brief Get the incoming Derivative tensor object
 *
 * @param idx Identifier of the output
 * @return Tensor tensor to incoming derivative. If
 */
const Tensor RunLayerContext::getIncomingDerivative(unsigned int idx) const {
  return getOutputGrad(idx);
}

/**
 * @brief Get the Input tensor object
 *
 * @param idx Identifier of the input
 * @return Tensor& Reference to the input tensor
 * @note For gradient checkpointing:
 *   - Initial forward: returns inputs (pointer to previous layer's output)
 *   - Recompute forward (first layer only): returns initial_inputs (saved copy)
 *   - Backward: returns inputs (recomputed values via pointer)
 */
Tensor &RunLayerContext::getInput(unsigned int idx) {
  // During initial forward, use initial_inputs if available (from checkpointed previous layers)
  if (is_initial_forward) {
    if (idx < initial_inputs.size() && initial_inputs[idx] != nullptr) {
      auto &tensor = initial_inputs[idx]->getVariableRef();
      // Also show what the normal input would be for comparison
      auto &normal_tensor = inputs[idx]->getVariableRef();
      printf("[Initial Forward_getInput] Layer: %s, idx: %u\n", std::get<props::Name>(props).get().c_str(), idx);
      printf("  -> Using INITIAL_INPUT: name='%s', ptr=%p, data_ptr=%p\n",
             tensor.getName().c_str(), (void *)&tensor, (void *)tensor.getData());
      printf("  -> Normal input would be: name='%s', ptr=%p, data_ptr=%p\n",
             normal_tensor.getName().c_str(), (void *)&normal_tensor, (void *)normal_tensor.getData());
      
      // Debug: Print first few values for layer0/add1 input[0] during initial forward
      std::string layer_name = std::get<props::Name>(props).get();
      if (layer_name == "layer0/add1" && idx == 0 && tensor.getData() != nullptr && tensor.size() > 0) {
        printf("[ADD1-INPUT0-VALUES-INITIAL] layer0/add1 input[0][0..2] = [%f, %f, %f]\n",
               tensor.getData()[0], tensor.getData()[1], tensor.getData()[2]);
      }
      
      return tensor; // Initial forward: use initial outputs from previous layer
    } else {
      printf("[Initial Forward_getInput][FALLBACK] Layer name: %s, idx: %u, initial_inputs missing (size=%zu). Using default input.\n",
             std::get<props::Name>(props).get().c_str(), idx, initial_inputs.size());
    }
  }
  auto &tensor = inputs[idx]->getVariableRef();
  printf("[Normal OR Recompute Forward_getInput] Layer name: %s, idx: %u, tensor_name: %s, tensor_ptr: %p, data_ptr: %p\n",
         std::get<props::Name>(props).get().c_str(), idx, tensor.getName().c_str(), (void *)&tensor, (void *)tensor.getData());
  
  // Debug: Print first few values for layer0/add1 input[0] to track value changes
  std::string layer_name = std::get<props::Name>(props).get();
  if (layer_name == "layer0/add1" && idx == 0 && tensor.getData() != nullptr && tensor.size() > 0) {
    printf("[ADD1-INPUT0-VALUES] layer0/add1 input[0][0..2] = [%f, %f, %f]\n",
           tensor.getData()[0], tensor.getData()[1], tensor.getData()[2]);
  }
  
  return tensor; // Normal/Recompute: pointer to recompute outputs
}

const Tensor &RunLayerContext::getInput(unsigned int idx) const {
  // During initial forward, use initial_inputs if available (from checkpointed previous layers)
  if (is_initial_forward) {
    if (idx < initial_inputs.size() && initial_inputs[idx] != nullptr) {
      auto &tensor = initial_inputs[idx]->getVariableRef();
      printf("[Initial Forward_getInput] Layer name: %s, idx: %u, tensor_name: %s, tensor_ptr: %p, data_ptr: %p\n",
             std::get<props::Name>(props).get().c_str(), idx, tensor.getName().c_str(), (void *)&tensor, (void *)tensor.getData());
      return tensor;
    } else {
      printf("[Initial Forward_getInput][FALLBACK] Layer name: %s, idx: %u, initial_inputs missing (size=%zu). Using default input.\n",
             std::get<props::Name>(props).get().c_str(), idx, initial_inputs.size());
    }
  }
  auto &tensor = inputs[idx]->getVariableRef();
  printf("[Normal_getInput] Layer name: %s, idx: %u, tensor_name: %s, tensor_ptr: %p, data_ptr: %p\n",
         std::get<props::Name>(props).get().c_str(), idx, tensor.getName().c_str(), (void *)&tensor, (void *)tensor.getData());
  return tensor;
}

/**
 * @brief Get the Input Grad tensor object
 *
 * @param idx Identifier of the input
 * @return Tensor& Reference to the input grad tensor
 * @note For gradient checkpointing:
 *   - Backward phase: returns inputs gradient (default)
 */
Tensor &RunLayerContext::getInputGrad(unsigned int idx) {
  if (!inputs[idx]->hasGradient()) {
    throw std::invalid_argument(
      "Requesting gradient for a non-trainable tensor.");
  }

  return inputs[idx]->getGradientRef();
}

/**
 * @brief check if the input has gradient
 *
 * @param idx Identifier of the input
 * @return true if output has gradient, else false
 */
bool RunLayerContext::inputHasGradient(unsigned int idx) const {
  return inputs[idx]->hasGradient();
}

/**
 * @brief Get the outgoing Derivative tensor object
 *
 * @param idx Identifier of the input
 * @return Tensor& Reference to the input derivative tensor
 */
Tensor &RunLayerContext::getOutgoingDerivative(unsigned int idx) {
  return getInputGrad(idx);
}

/**
 * @brief Get the Tensor object
 *
 * @param idx Identifier of the tensor
 * @return Tensor& Reference to the tensor
 * @note For gradient checkpointing:
 *   - Initial forward: returns initial_tensors (short-lived)
 *   - Recompute forward: returns tensors (default)
 *   - Backward phase: returns tensors (default)
 */
Tensor &RunLayerContext::getTensor(unsigned int idx) {
  if (is_initial_forward) {
    if (idx < initial_tensors.size() && initial_tensors[idx] != nullptr) {
      auto &tensor = initial_tensors[idx]->getVariableRef();
      printf("[Initial Forward_getTensor] Layer name: %s, idx: %u, tensor_name: %s, tensor_ptr: %p, data_ptr: %p\n",
             std::get<props::Name>(props).get().c_str(), idx, tensor.getName().c_str(), (void *)&tensor, (void *)tensor.getData());
      return tensor;
    } else {
      // Fallback to tensors if initial_tensors not available
      auto &tensor = tensors[idx]->getVariableRef();
      printf("[Initial Forward_getTensor][FALLBACK] Layer name: %s, idx: %u, tensor_name: %s, initial_tensors missing (size=%zu). Using tensors.\n",
             std::get<props::Name>(props).get().c_str(), idx, tensor.getName().c_str(), initial_tensors.size());
      return tensor;
    }
  }
  auto &tensor = tensors[idx]->getVariableRef();
  printf("[Normal_getTensor] Layer name: %s, idx: %u, tensor_name: %s, tensor_ptr: %p, data_ptr: %p\n",
         std::get<props::Name>(props).get().c_str(), idx, tensor.getName().c_str(), (void *)&tensor, (void *)tensor.getData());
  return tensors[idx]->getVariableRef();
}

/**
 * @brief Get the Tensor object
 *
 * @param idx Identifier of the tensor
 * @return Tensor& Reference to the tensor
 * @note For gradient checkpointing:
 *   - Initial forward: returns initial_tensors (short-lived)
 *   - Recompute forward: returns tensors (default)
 *   - Backward phase: returns tensors (default)
 */
const Tensor &RunLayerContext::getTensor(unsigned int idx) const {
  if (is_initial_forward) {
    if (idx < initial_tensors.size() && initial_tensors[idx] != nullptr) {
      auto &tensor = initial_tensors[idx]->getVariableRef();
      printf("[Initial Forward_getTensor] Layer name: %s, idx: %u, tensor_name: %s, tensor_ptr: %p, data_ptr: %p\n",
             std::get<props::Name>(props).get().c_str(), idx, tensor.getName().c_str(), (void *)&tensor, (void *)tensor.getData());
      return tensor;
    } else if (!initial_tensors.empty()) {
      printf("[Initial Forward_getTensor][FALLBACK] Layer name: %s, idx: %u, initial_tensors missing (size=%zu). Using default tensor.\n",
             std::get<props::Name>(props).get().c_str(), idx, initial_tensors.size());
    }
  }
  auto &tensor = tensors[idx]->getVariableRef();
  printf("[Normal_getTensor] Layer name: %s, idx: %u, tensor_name: %s, tensor_ptr: %p, data_ptr: %p\n",
         std::get<props::Name>(props).get().c_str(), idx, tensor.getName().c_str(), (void *)&tensor, (void *)tensor.getData());
  return tensors[idx]->getVariableRef();
}

/**
 * @brief Get the Tensor Grad object
 *
 * @param idx Identifier of the tensor
 * @return Tensor& Reference to the tensor grad tensor
 */
Tensor &RunLayerContext::getTensorGrad(unsigned int idx) {
  if (!tensors[idx]->hasGradient())
    throw std::invalid_argument(
      "Requesting gradient for a non-trainable tensor.");
  return tensors[idx]->getGradientRef();
}

/**
 * @brief Get the Tensor Grad object
 *
 * @param idx Identifier of the tensor
 * @return Tensor& Reference to the tensor grad tensor
 */
const Tensor &RunLayerContext::getTensorGrad(unsigned int idx) const {
  if (!tensors[idx]->hasGradient())
    throw std::invalid_argument(
      "Requesting gradient for a non-trainable tensor.");
  return tensors[idx]->getGradientRef();
}

/**
 * @brief check if the tensor has gradient
 *
 * @param idx Identifier of the tensor
 * @return true if tensor has gradient, else false
 */
bool RunLayerContext::tensorHasGradient(unsigned int idx) const {
  return tensors[idx]->hasGradient();
}

bool RunLayerContext::isWeightDependent(unsigned int idx) const {
  return weights[idx]->isDependent();
}

bool RunLayerContext::isGradientFirstAccess(unsigned int idx) const {
  return weights[idx]->isGradientFirstAccess();
}

bool RunLayerContext::isGradientLastAccess(unsigned int idx) const {
  return weights[idx]->isGradientLastAccess();
}

bool RunLayerContext::isGradientClipByGlobalNorm(unsigned int idx) const {
  return weights[idx]->isGradientClipByGlobalNorm();
}

bool RunLayerContext::isMixedPrecision(unsigned int idx) const {
  return weights[idx]->isMixedPrecision();
}

/**
 * @brief Get the tensor name
 *
 * @param idx Identifier of the tensor
 * @return name of the tensor
 */
const std::string &RunLayerContext::getTensorName(unsigned int idx) const {
  return tensors[idx]->getName();
}

/**
 * @brief Get the number of Outputs tensor objects
 *
 * @return unsigned int number of output tensors
 */
unsigned int RunLayerContext::getNumOutputs() const { return outputs.size(); }

/**
 * @brief Get the number of inputs tensor objects
 *
 * @return unsigned int number of input tensors
 */
unsigned int RunLayerContext::getNumInputs() const { return inputs.size(); }

/**
 * @brief Get the number of weights tensor objects
 *
 * @return unsigned int number of weight tensors
 */
unsigned int RunLayerContext::getNumWeights() const { return weights.size(); }

/**
 * @brief Get the number of requested tensors objects
 *
 * @return unsigned int number of requested tensors
 */
unsigned int RunLayerContext::getNumTensors() const { return tensors.size(); }

/**
 * @brief Set the batch for the run context
 *
 * @param batch Update batch size
 */
void RunLayerContext::setBatch(unsigned int batch) {
  for (auto &vg : inputs)
    vg->setBatchSize(batch);
  for (auto &vg : outputs)
    vg->setBatchSize(batch);
  for (auto &vg : initial_inputs)
    vg->setBatchSize(batch);
  for (auto &vg : initial_outputs)
    vg->setBatchSize(batch);
}

/**
 * @brief Update the dimensions for a requested tensor
 *
 * @param idx index of the tensor (identifier)
 * @param batch Updated batch size
 */
void RunLayerContext::updateTensor(unsigned int idx, unsigned int batch) {
  tensors[idx]->setBatchSize(batch);
  if (idx < initial_tensors.size() && initial_tensors[idx])
    initial_tensors[idx]->setBatchSize(batch);
}

void RunLayerContext::updateInput(unsigned int idx, TensorDim dimension) {
  inputs[idx]->updateDimension(dimension);
  if (idx < initial_inputs.size() && initial_inputs[idx])
    initial_inputs[idx]->updateDimension(dimension);
}

void RunLayerContext::updateOutput(unsigned int idx, TensorDim dimension) {
  outputs[idx]->updateDimension(dimension);
  if (idx < initial_outputs.size() && initial_outputs[idx])
    initial_outputs[idx]->updateDimension(dimension);
}

void RunLayerContext::updateTensor(unsigned int idx, TensorDim dimension) {
  tensors[idx]->updateDimension(dimension);
  if (idx < initial_tensors.size() && initial_tensors[idx])
    initial_tensors[idx]->updateDimension(dimension);
}

/**
 * @brief   Get weight object for the weights
 *
 * @param idx index of the weight (identifier)
 * @return weight object
 */
Weight &RunLayerContext::getWeightObject(unsigned int idx) {
  return *weights[idx];
}

/**
 * @brief   check if the label is available
 *
 * @param idx Identifier of the input
 * @return true if label is available else false
 */
bool RunLayerContext::isLabelAvailable(unsigned int idx) const {
  return outputs[idx]->getGradientRef().isAllocated();
}

/**
 * @brief   Get label tensor
 *
 * @param idx Identifier of the input
 * @return Tensor& Reference to the label tensor
 */
Tensor &RunLayerContext::getLabel(unsigned int idx) {
  if (isLabelAvailable(idx))
    return outputs[idx]->getGradientRef();
  else {
    std::stringstream ss;
    ss << "Requesting label of index: " << idx << "for " << getName()
       << " does not exist";
    throw std::invalid_argument(ss.str().c_str());
  }
}

/**
 * @brief   check if run context is set and is ready to use
 *
 * @return true if ready, else false
 */
bool RunLayerContext::readyToUse() const {
  /**
   * assumption:
   * 1. there must be at least 1 input
   * 2. the setter set everything at once
   */
  if (inputs.empty())
    return false;
  return !inputs[0]->getVariable().empty();
}

/**
 * @brief   validates the run context after run
 *
 * @return true if ready, else false
 */
bool RunLayerContext::validate(bool skip_input, bool skip_label) {
  /**
   * @note a common mistake when using run_context is re-assigning the tensor
   * references which leads to nasty bugs. This validation ensures that the
   * tensors are not set mistakenly by verifying their unique names
   */
  bool ret = true;
#ifdef DEBUG
  std::function<bool(const Var_Grad *, bool)> matcher;

  if (tensor_map.empty() || !tensor_map[inputs[0]->getName()]) {
    auto filler = [this](const auto &vec) {
      for (auto const &val : vec) {
        if (val->getVariableRef().getTensorType().data_type ==
            TensorDim::DataType::FP32) {
          tensor_map[val->getName()] = val->getVariableRef().getData();
          tensor_map[val->getGradientName()] = val->getGradientRef().getData();
        } else if (val->getVariableRef().getTensorType().data_type ==
                   TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
          tensor_map[val->getName()] =
            val->getVariableRef().template getData<_FP16>();
          tensor_map[val->getGradientName()] =
            val->getGradientRef().template getData<_FP16>();
#else
          throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
        }
      }
    };

    /** fill the tensor map for the first validation */
    filler(weights);
    filler(inputs);
    filler(outputs);
    filler(tensors);
  }

  matcher = [this](const Var_Grad *val, bool skip_grad) -> bool {
    if (val->getName().empty() ||
        (val->hasGradient() && val->getGradientName().empty()))
      return false;

    if (tensor_map.find(val->getName()) == tensor_map.end())
      /**
       * Disabled because of in-place input layer. Enable this later.
       * tensor_map[val->getName()] != val->getVariableRef().getData())
       */
      return false;

    if (skip_grad &&
        (tensor_map.find(val->getGradientName()) == tensor_map.end()))
      return false;

    return true;
  };

  auto matcher_w = [this, &matcher](const std::vector<Weight *> &vec) {
    return std::all_of(vec.begin(), vec.end(),
                       std::bind(matcher, std::placeholders::_1, false));
  };

  auto matcher_vw = [this, &matcher](const std::vector<Var_Grad *> &vec,
                                     bool skip_grad = false) {
    return std::all_of(vec.begin(), vec.end(),
                       std::bind(matcher, std::placeholders::_1, skip_grad));
  };

  /** match the tensor map from the next validations */
  ret =
    matcher_w(weights) & matcher_vw(tensors) & matcher_vw(outputs, skip_label);
  if (!skip_input)
    ret &= matcher_vw(inputs);
#endif

  return ret;
}

} // namespace nntrainer
