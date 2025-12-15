// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Samsung Electronics Co., Ltd.
 *
 * @file   tensor_lifetime_report.h
 * @date   13 December 2024
 * @brief  Utility to report tensor lifetime information for debugging
 * @see    https://github.com/nnstreamer/nntrainer
 * @author NNTrainer Team
 * @bug    No known bugs except for NYI items
 */

#ifndef __TENSOR_LIFETIME_REPORT_H__
#define __TENSOR_LIFETIME_REPORT_H__

#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <tensor_wrap_specs.h>

namespace nntrainer {

/**
 * @brief Convert TensorLifespan enum to string
 * @note TensorLifespan is a bitmask, so we need to check each bit
 */
inline std::string lifespanToString(TensorLifespan ls) {
  // Check for exact matches first
  switch (ls) {
  case TensorLifespan::UNMANAGED:
    return "UNMANAGED";
  case TensorLifespan::FORWARD_FUNC_LIFESPAN:
    return "FORWARD_FUNC";
  case TensorLifespan::CALC_DERIV_LIFESPAN:
    return "CALC_DERIV";
  case TensorLifespan::CALC_GRAD_LIFESPAN:
    return "CALC_GRAD";
  case TensorLifespan::CALC_GRAD_DERIV_LIFESPAN:
    return "CALC_GRAD_DERIV";
  case TensorLifespan::CALC_AGRAD_LIFESPAN:
    return "CALC_AGRAD";
  case TensorLifespan::CALC_GRAD_DERIV_AGRAD_LIFESPAN:
    return "CALC_GRAD_DERIV_AGRAD";
  case TensorLifespan::ITERATION_LIFESPAN:
    return "ITERATION";
  case TensorLifespan::EPOCH_LIFESPAN:
    return "EPOCH";
  case TensorLifespan::MAX_LIFESPAN:
    return "MAX";
  default:
    break;
  }
  
  // Handle composite lifespans by checking individual bits
  std::string result;
  int ls_val = static_cast<int>(ls);
  
  if (ls_val & static_cast<int>(TensorLifespan::FORWARD_FUNC_LIFESPAN))
    result += "FWD|";
  if (ls_val & static_cast<int>(TensorLifespan::CALC_DERIV_LIFESPAN))
    result += "DERIV|";
  if (ls_val & static_cast<int>(TensorLifespan::CALC_GRAD_LIFESPAN))
    result += "GRAD|";
  if (ls_val & static_cast<int>(TensorLifespan::CALC_AGRAD_LIFESPAN))
    result += "AGRAD|";
  if (ls_val & static_cast<int>(TensorLifespan::FORWARD_RECOMPUTE_LIFESPAN))
    result += "RECOMPUTE|";
  
  if (!result.empty()) {
    result.pop_back(); // Remove trailing '|'
    return result;
  }
  
  return "UNKNOWN(" + std::to_string(ls_val) + ")";
}

/**
 * @brief Structure to hold tensor allocation information
 */
struct TensorAllocationInfo {
  std::string name;           /**< Tensor name */
  std::string layer_name;     /**< Layer that owns this tensor */
  std::string type;           /**< Type: INPUT, OUTPUT, WEIGHT, TENSOR, etc. */
  TensorLifespan lifespan;    /**< Tensor lifespan */
  std::vector<unsigned int> exec_order; /**< Execution orders */
  size_t size_bytes;          /**< Size in bytes */
  bool is_gradient;           /**< Whether this is a gradient tensor */
  std::string reference_name; /**< Reference tensor name (for views) */
  bool is_view;               /**< Whether this is a view of another tensor */
};

/**
 * @brief Class to collect and report tensor lifetime information
 */
class TensorLifetimeReport {
public:
  /**
   * @brief Get singleton instance
   */
  static TensorLifetimeReport &getInstance() {
    static TensorLifetimeReport instance;
    return instance;
  }

  /**
   * @brief Clear all collected information
   */
  void clear() {
    allocations.clear();
    layer_order.clear();
  }

  /**
   * @brief Add tensor allocation info
   */
  void addAllocation(const TensorAllocationInfo &info) {
    allocations.push_back(info);
    if (std::find(layer_order.begin(), layer_order.end(), info.layer_name) ==
        layer_order.end()) {
      layer_order.push_back(info.layer_name);
    }
  }

  /**
   * @brief Get all allocations (for updating with final values)
   */
  std::vector<TensorAllocationInfo> &getAllocations() {
    return allocations;
  }

  /**
   * @brief Generate report to file
   */
  void generateReport(const std::string &filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
      fprintf(stderr, "[TensorLifetimeReport] Failed to open file: %s\n",
              filename.c_str());
      return;
    }

    file << "================================================================================\n";
    file << "                        TENSOR LIFETIME REPORT\n";
    file << "================================================================================\n\n";

    // Summary
    file << "SUMMARY\n";
    file << "-------\n";
    file << "Total tensors: " << allocations.size() << "\n";
    
    size_t total_size = 0;
    std::map<std::string, int> type_count;
    std::map<TensorLifespan, int> lifespan_count;
    
    for (const auto &alloc : allocations) {
      total_size += alloc.size_bytes;
      type_count[alloc.type]++;
      lifespan_count[alloc.lifespan]++;
    }
    
    file << "Total size: " << (total_size / 1024.0 / 1024.0) << " MB\n\n";
    
    file << "By Type:\n";
    for (const auto &[type, count] : type_count) {
      file << "  " << std::setw(15) << type << ": " << count << "\n";
    }
    
    file << "\nBy Lifespan:\n";
    for (const auto &[ls, count] : lifespan_count) {
      file << "  " << std::setw(25) << lifespanToString(ls) << ": " << count << "\n";
    }
    
    file << "\n";

    // Detailed report by layer
    file << "================================================================================\n";
    file << "                        DETAILED REPORT BY LAYER\n";
    file << "================================================================================\n\n";

    for (const auto &layer_name : layer_order) {
      file << "--------------------------------------------------------------------------------\n";
      file << "LAYER: " << layer_name << "\n";
      file << "--------------------------------------------------------------------------------\n";

      for (const auto &alloc : allocations) {
        if (alloc.layer_name != layer_name)
          continue;

        file << "\n  [" << alloc.type << (alloc.is_gradient ? " GRAD" : "") << "] "
             << alloc.name << "\n";
        file << "    Lifespan: " << lifespanToString(alloc.lifespan) << "\n";
        file << "    Size: " << alloc.size_bytes << " bytes ("
             << (alloc.size_bytes / 1024.0) << " KB)\n";
        
        if (!alloc.exec_order.empty()) {
          file << "    Exec Order: [";
          for (size_t i = 0; i < alloc.exec_order.size(); ++i) {
            if (i > 0) file << ", ";
            file << alloc.exec_order[i];
          }
          file << "]\n";
          
          if (alloc.exec_order.size() >= 2) {
            auto minmax = std::minmax_element(alloc.exec_order.begin(), 
                                               alloc.exec_order.end());
            file << "    Exec Range: " << *minmax.first << " - " << *minmax.second << "\n";
          }
        }
        
        if (alloc.is_view) {
          file << "    View of: " << alloc.reference_name << "\n";
        }
      }
      file << "\n";
    }

    // Gradient tensors specific report
    file << "\n================================================================================\n";
    file << "                        GRADIENT TENSORS REPORT\n";
    file << "================================================================================\n\n";

    file << std::setw(50) << std::left << "Gradient Name" 
         << std::setw(25) << "Lifespan"
         << std::setw(20) << "Exec Range"
         << "View Of\n";
    file << std::string(120, '-') << "\n";

    for (const auto &alloc : allocations) {
      if (!alloc.is_gradient) continue;
      
      std::string exec_range = "-";
      if (!alloc.exec_order.empty()) {
        auto minmax = std::minmax_element(alloc.exec_order.begin(), 
                                           alloc.exec_order.end());
        exec_range = std::to_string(*minmax.first) + "-" + std::to_string(*minmax.second);
      }
      
      file << std::setw(50) << std::left << alloc.name
           << std::setw(25) << lifespanToString(alloc.lifespan)
           << std::setw(20) << exec_range
           << (alloc.is_view ? alloc.reference_name : "-")
           << "\n";
    }

    file << "\n[Report generated successfully]\n";
    file.close();
    
    printf("[TensorLifetimeReport] Report saved to: %s\n", filename.c_str());
  }

  /**
   * @brief Print summary to stdout
   */
  void printSummary() {
    printf("\n=== Tensor Lifetime Summary ===\n");
    printf("Total tensors: %zu\n", allocations.size());
    
    std::map<std::string, int> type_count;
    std::map<TensorLifespan, int> lifespan_count;
    
    for (const auto &alloc : allocations) {
      type_count[alloc.type]++;
      lifespan_count[alloc.lifespan]++;
    }
    
    printf("\nBy Type:\n");
    for (const auto &[type, count] : type_count) {
      printf("  %15s: %d\n", type.c_str(), count);
    }
    
    printf("\nBy Lifespan:\n");
    for (const auto &[ls, count] : lifespan_count) {
      printf("  %25s: %d\n", lifespanToString(ls).c_str(), count);
    }
    printf("\n");
  }

private:
  TensorLifetimeReport() = default;
  std::vector<TensorAllocationInfo> allocations;
  std::vector<std::string> layer_order;
};

} // namespace nntrainer

#endif // __TENSOR_LIFETIME_REPORT_H__
