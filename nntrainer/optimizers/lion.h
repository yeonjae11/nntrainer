// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 
 *
 * @file   lion.h
 * @date  
 * @see    
 * @author 
 * @author 
 * @author 
 * @bug    
 * @brief  
 */


#ifndef __LION_H__
#define __LION_H__
#ifdef __cplusplus

#include <tuple>
#include <base_properties.h>
#include <optimizer_devel.h>

namespace nntrainer {

/**
 * @brief Beta 1 props
 *
 */
class PropsB1 : public Property<double> {
public:
  static constexpr const char *key = "beta1"; /**< unique key to access */
  using prop_tag = double_prop_tag;           /**< property type */
};

/**
 * @brief Beta 2 props
 *
 */
class PropsB2 : public Property<double> {
public:
  static constexpr const char *key = "beta2"; /**< unique key to access */
  using prop_tag = double_prop_tag;           /**< property type */
};

/**
 * @brief weight decay property
 *
 */
class PropsWeightDecay : public Property<double> {
public:
  static constexpr const char *key = "weight_decay"; /**< unique key to access */
  using prop_tag = double_prop_tag;                  /**< property type */
};

/**
 * @class   Lion Optimizer class
 * @brief   Lion Optimizer (E. Chen et al., 2023)
 */
class Lion : public Optimizer {
public:
  /**
   * @brief Construct a new Lion object
   */
  Lion();

  /**
   * @brief Destroy the Lion object
   */
  ~Lion();

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
  const std::string getType() const override { return Lion::type; }

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

  static constexpr const char *type = "lion";

private:
  std::tuple<PropsB1, PropsB2, PropsWeightDecay> lion_props;
};
} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __LION_H__ */
