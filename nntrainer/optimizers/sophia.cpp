// SPDX-License-Identifier: Apache-2.0
/**
 * @file   sophia.cpp
 * @brief  Sophia Optimizer (skeleton)
 */

#include <sophia.h>

namespace nntrainer {

Sophia::Sophia() = default;

Sophia::~Sophia() = default;

std::vector<TensorDim> Sophia::getOptimizerVariableDim(const TensorDim & /*dim*/) {
  /** Skeleton: no extra optimizer variables (acts like plain SGD) */
  return {};
}

void Sophia::exportTo(Exporter &exporter,
                      const ml::train::ExportMethods &method) const {
  /** Skeleton: delegate to base */
  Optimizer::exportTo(exporter, method);
}

void Sophia::setProperty(const std::vector<std::string> &values) {
  /** Skeleton: no custom properties, forward to base */
  Optimizer::setProperty(values);
}

void Sophia::applyGradient(RunOptimizerContext &context) {
  /** Skeleton: plain gradient application with current learning rate */
  context.applyGradient(context.getLearningRate());
}

} // namespace nntrainer


