// SPDX-License-Identifier: Apache-2.0
/**
 * @file   lion.cpp
 * @date   17 October 2025
 * @brief  This is the Lion Optimizer.
 * @see    https://github.com/nnstreamer/nntrainer
 */

#include <cmath>
#include <fstream>

#include <lion.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace nntrainer {

Lion::Lion() : lion_props(PropsB1(), PropsB2(), PropsWeightDecay()) {
  auto &[beta1, beta2, weight_decay] = lion_props;
  beta1.set(0.9f);
  beta2.set(0.99f);
  weight_decay.set(0.0f);
}

Lion::~Lion() {}

enum LionParams { exp_avg };

std::vector<TensorDim> Lion::getOptimizerVariableDim(const TensorDim &dim) {
  return {dim};
}

void Lion::exportTo(Exporter &exporter,
                    const ml::train::ExportMethods &method) const {
  exporter.saveResult(lion_props, method, this);
  Optimizer::exportTo(exporter, method);
}

void Lion::setProperty(const std::vector<std::string> &values) {
  auto left = loadProperties(values, lion_props);
  Optimizer::setProperty(left);
}

void Lion::applyGradient(RunOptimizerContext &context) {
  Tensor &grad = context.getGradient();
  if (grad.empty())
    return;

  context.applyLossScale(grad);

  auto &beta1 = std::get<PropsB1>(lion_props).get();
  auto &beta2 = std::get<PropsB2>(lion_props).get();
  auto &weight_decay = std::get<PropsWeightDecay>(lion_props).get();

  Tensor &param = context.getWeight();
  Tensor &exp_avg = context.getOptimizerVariable(LionParams::exp_avg);

  float lr = context.getLearningRate();

  /** Weight decay */
  if (weight_decay > 0.0f) {
    param.multiply_i(1.0f - lr * weight_decay);
  }

  /** update = beta1 * exp_avg + (1 - beta1) * grad */
  Tensor update = exp_avg.multiply(beta1);
  update.add_i(grad, 1.0f - beta1);

  /** param = param - lr * sign(update) */
  Tensor sign_update = update.apply<float>([](float v) { return (v > 0.f) - (v < 0.f); });
  param.add_i(sign_update, -lr);

  /** exp_avg = beta2 * exp_avg + (1 - beta2) * grad */
  exp_avg.multiply_i(beta2);
  exp_avg.add_i(grad, 1.0f - beta2);
}

} // namespace nntrainer
