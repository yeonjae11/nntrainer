// SPDX-License-Identifier: Apache-2.0
/**
 * @file   sophia.h
 * @brief  Sophia Optimizer (skeleton)
 */

#ifndef __SOPHIA_H__
#define __SOPHIA_H__
#ifdef __cplusplus

#include <vector>

#include <adam.h>              // PropsB1, PropsB2, PropsEpsilon
#include <optimizer_devel.h>
#include <base_properties.h>
#include <lion.h>             // PropsWeightDecay

namespace nntrainer {

/**
 * @brief rho clipping threshold property
 */
class PropsRho : public Property<double> {
public:
  static constexpr const char *key = "rho";
  using prop_tag = double_prop_tag;
};

/**
 * @brief Hessian update period K (every K steps)
 */
class PropsK : public PositiveIntegerProperty {
public:
  static constexpr const char *key = "k";
  using prop_tag = uint_prop_tag;
};

/**
 * @class   Sophia Optimizer class (skeleton)
 * @brief   Clipped 2nd-moment with stochastic Hessian approximation optimizer
 */
class Sophia : public Optimizer {
public:
  /**
   * @brief Construct a new Sophia object
   */
  Sophia();

  /**
   * @brief Destroy the Sophia object
   */
  ~Sophia();

  /**
   * @copydoc Optimizer::getDefaultLearningRate()
   */
  double getDefaultLearningRate() const override { return 1e-4; }

  /**
   * @copydoc Optimizer::applyGradient(RunOptimizerContext &context)
   */
  void applyGradient(RunOptimizerContext &context) override;

  /**
   * @copydoc Optimizer::getType()
   */
  const std::string getType() const override { return Sophia::type; }

  /**
   * @copydoc Optimizer::getOptimizerVariableDim(const TensorDim &dim)
   */
  std::vector<TensorDim> getOptimizerVariableDim(const TensorDim &dim) override;

  /**
   * @copydoc Optimizer::exportTo(Exporter &exporter,
   * const ml::train::ExportMethods &method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Optimizer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  static constexpr const char *type = "sophia";

private:
  /**
   * @brief beta1, beta2, epsilon, rho, weight_decay
   */
  std::tuple<PropsB1, PropsB2, PropsEpsilon, PropsRho, PropsWeightDecay, PropsK>
    sophia_props;
};

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __SOPHIA_H__ */

