// SPDX-License-Identifier: Apache-2.0
/**
 * @file   sophia.cpp
 * @brief  Sophia Optimizer (skeleton)
 */

#include <algorithm>
#include <cmath>
#include <sophia.h>
#include <node_exporter.h>

namespace nntrainer {

Sophia::Sophia() : sophia_props(PropsB1(), PropsB2(), PropsEpsilon(), PropsRho(), PropsWeightDecay(), PropsK()) {
  auto &[beta1, beta2, eps, rho, wd, kprop] = sophia_props;
  beta1.set(0.9f);
  beta2.set(0.99f);
  eps.set(1.0e-12f);
  rho.set(0.03f);
  wd.set(0.0f);
  kprop.set(10u);
}

Sophia::~Sophia() = default;

enum SophiaParams { m, h };

std::vector<TensorDim> Sophia::getOptimizerVariableDim(const TensorDim &dim) {
  /**
   * m: first-moment (momentum), h: hessian diagonal estimate (EMA)
   * Keep in FP32 for stability, even under mixed precision.
   */
  TensorDim m_dim(dim);
  TensorDim h_dim(dim);
  m_dim.setDataType(ml::train::TensorDim::DataType::FP32);
  h_dim.setDataType(ml::train::TensorDim::DataType::FP32);
  return {m_dim, h_dim};
}

void Sophia::exportTo(Exporter &exporter,
                      const ml::train::ExportMethods &method) const {
  exporter.saveResult(sophia_props, method, this);
  Optimizer::exportTo(exporter, method);
}

void Sophia::setProperty(const std::vector<std::string> &values) {
  auto left = loadProperties(values, sophia_props);
  Optimizer::setProperty(left);
}

void Sophia::applyGradient(RunOptimizerContext &context) {
  // Prepare gradient in FP32 and apply loss scaling if any
  Tensor empty_tensor;
  Tensor &x_grad =
    context.getGradient().getDataType() == ml::train::TensorDim::DataType::FP32
      ? context.getGradient()
      : empty_tensor;

  if (x_grad.empty()) {
    x_grad = context.getGradient().clone(ml::train::TensorDim::DataType::FP32);
  }

  context.applyLossScale(x_grad);

  // State tensors
  Tensor &m_t = context.getOptimizerVariable(SophiaParams::m);
  Tensor &h_t = context.getOptimizerVariable(SophiaParams::h);

  // Hyper-parameters
  const float beta1 = std::get<PropsB1>(sophia_props).get();
  const float beta2 = std::get<PropsB2>(sophia_props).get();
  const float eps   = std::get<PropsEpsilon>(sophia_props).get();
  const float rho   = std::get<PropsRho>(sophia_props).get();
  const float wd    = std::get<PropsWeightDecay>(sophia_props).get();
  const unsigned int iter = context.getIteration();
  const unsigned int K = std::get<PropsK>(sophia_props).get();

  // 1) Hessian EMA update every K steps using empirical Fisher (grad^2)
  if (K > 0 && ((iter + 1) % K) == 0) {
    h_t.multiply_i(beta2);
    Tensor grad_sq = x_grad.multiply(x_grad);
    h_t.add_i(grad_sq, 1.0f - beta2);
  }

  // 2) First moment (momentum) update: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
  m_t.multiply_i(beta1);
  m_t.add_i(x_grad, 1.0f - beta1);

  // 3) ratio = m / (B * h + eps), where B is effective batch size from context
  Tensor denom = h_t.clone();
  {
    unsigned int B = context.getBatchSize();
    if (B > 1u) {
      denom.multiply_i(static_cast<float>(B));
    }
  }
  denom.add_i(eps);

  Tensor ratio;
  // Use out-parameter overload if available: ratio = m_t / denom
  // Fallback: create via divide returning a tensor
  ratio = m_t.divide(denom);

  // 4) Clamp per-coordinate to [-rho, rho]
  std::function<float(float)> clamp_func = [rho](float val) {
    if (val > rho) return rho;
    if (val < -rho) return -rho;
    return val;
  };
  ratio.apply_i<float>(clamp_func);

  // 5) Decoupled weight decay (optional)
  if (wd > 0.0f) {
    Tensor &decay_src =
      context.isMixedPrecision() ? context.getWeightFP32() : context.getWeight();
    ratio.add_i(decay_src, wd);
  }

  // 6) Apply update with lr scaled by 1/rho so that max step per element is lr
  const float lr = context.getLearningRate();
  const float scaled_lr = lr / rho;
  context.applyGradient(scaled_lr, ratio);
}

} // namespace nntrainer
