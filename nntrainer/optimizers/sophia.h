// SPDX-License-Identifier: Apache-2.0
/**
 * @file   sophia.h
 * @brief  Sophia Optimizer (skeleton)
 */

#ifndef __SOPHIA_H__
#define __SOPHIA_H__
#ifdef __cplusplus

#include <vector>

#include <optimizer_devel.h>

namespace nntrainer {

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
  double getDefaultLearningRate() const override { return 1e-3; }

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
};

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __SOPHIA_H__ */

