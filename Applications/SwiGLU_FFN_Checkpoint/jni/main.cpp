#include <layer.h>
#include <model.h>
#include <optimizer.h>
#include <cifar_dataloader.h>
#include <profiler.h>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <sys/resource.h>
#include <unistd.h>
#include <iomanip>
#include <iteration_profiler.h>

// Default values (increased for better performance)
int number_of_db = 32;
int batch_size = 8;
int epochs = 10;
float learning_rate = 0.0001;
int seq_len = 128;
int vocab_size = 1000;
int model_dim = 256;
int num_heads = 8;
int num_layers = 6;
int ffn_dim = 1024;
bool enable_profile = false;
bool enable_checkpointing = true;  // NEW: Enable gradient checkpointing by default

// Performance measurement utilities
class PerformanceMonitor {
public:
    static size_t getMemoryUsage() {
        std::ifstream status_file("/proc/self/status");
        std::string line;
        while (std::getline(status_file, line)) {
            if (line.substr(0, 6) == "VmRSS:") {
                std::istringstream iss(line);
                std::string label;
                size_t memory_kb;
                iss >> label >> memory_kb;
                return memory_kb * 1024; // Convert to bytes
            }
        }
        return 0;
    }
    
    static size_t getPeakMemoryUsage() {
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        return usage.ru_maxrss * 1024; // Convert to bytes (Linux uses KB)
    }
    
    static void printMemoryInfo(const std::string& stage) {
        size_t current_mem = getMemoryUsage();
        size_t peak_mem = getPeakMemoryUsage();
        std::cout << "[MEMORY] " << stage << " - Current: " 
                  << (current_mem / 1024 / 1024) << " MB, Peak: " 
                  << (peak_mem / 1024 / 1024) << " MB" << std::endl;
    }
    
    static auto startTimer() {
        return std::chrono::high_resolution_clock::now();
    }
    
    static void printElapsed(const std::string& stage, 
                           const std::chrono::high_resolution_clock::time_point& start) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "[TIMING] " << stage << " took: " << duration.count() << " ms" << std::endl;
    }
    
    static long getElapsedMicroseconds(const std::chrono::high_resolution_clock::time_point& start) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return duration.count();
    }
};

void parseArguments(int argc, char* argv[]) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--seq_len" && i + 1 < argc) {
            seq_len = std::atoi(argv[++i]);
        } else if (arg == "--vocab_size" && i + 1 < argc) {
            vocab_size = std::atoi(argv[++i]);
        } else if (arg == "--model_dim" && i + 1 < argc) {
            model_dim = std::atoi(argv[++i]);
        } else if (arg == "--num_heads" && i + 1 < argc) {
            num_heads = std::atoi(argv[++i]);
        } else if (arg == "--num_layers" && i + 1 < argc) {
            num_layers = std::atoi(argv[++i]);
        } else if (arg == "--ffn_dim" && i + 1 < argc) {
            ffn_dim = std::atoi(argv[++i]);
        } else if (arg == "--batch_size" && i + 1 < argc) {
            batch_size = std::atoi(argv[++i]);
        } else if (arg == "--epochs" && i + 1 < argc) {
            epochs = std::atoi(argv[++i]);
        } else if (arg == "--learning_rate" && i + 1 < argc) {
            learning_rate = std::atof(argv[++i]);
        } else if (arg == "--enable-profile") {
            enable_profile = true;
            nntrainer::g_enable_iteration_profile = true;
            printf("profile activate\n");
        } else if (arg == "--disable-checkpointing") {
            enable_checkpointing = false;
            std::cout << "[INFO] Gradient checkpointing DISABLED" << std::endl;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --seq_len <int>        Sequence length (default: 128)\n"
                      << "  --vocab_size <int>     Vocabulary size (default: 1000)\n"
                      << "  --model_dim <int>      Model dimension (default: 256)\n"
                      << "  --num_heads <int>      Number of attention heads (default: 8)\n"
                      << "  --num_layers <int>     Number of transformer layers (default: 6)\n"
                      << "  --ffn_dim <int>        FFN dimension (default: 1024)\n"
                      << "  --batch_size <int>     Batch size (default: 8)\n"
                      << "  --epochs <int>         Number of epochs (default: 10)\n"
                      << "  --learning_rate <float> Learning rate (default: 0.0001)\n"
                      << "  --enable-profile       Enable detailed profiling output\n"
                      << "  --disable-checkpointing Disable gradient checkpointing\n"
                      << "  --help, -h             Show this help message\n";
            exit(0);
        }
    }
    
    // Validation
    if (model_dim % num_heads != 0) {
        std::cerr << "Error: model_dim (" << model_dim << ") must be divisible by num_heads (" << num_heads << ")" << std::endl;
        exit(1);
    }
    
    std::cout << "[CONFIG] Parsed arguments:" << std::endl;
    std::cout << "  seq_len=" << seq_len << ", vocab_size=" << vocab_size << ", model_dim=" << model_dim << std::endl;
    std::cout << "  num_heads=" << num_heads << ", num_layers=" << num_layers << ", ffn_dim=" << ffn_dim << std::endl;
    std::cout << "  batch_size=" << batch_size << ", epochs=" << epochs << ", learning_rate=" << learning_rate << std::endl;
    std::cout << "  gradient_checkpointing=" << (enable_checkpointing ? "ENABLED" : "DISABLED") << std::endl;
}

std::unique_ptr<ml::train::Model> create_model() {
    std::cout << "[DEBUG] Starting model creation..." << std::endl;
    
    std::unique_ptr<ml::train::Model> model = 
        ml::train::createModel(ml::train::ModelType::NEURAL_NET, {"loss=mse"});
    std::cout << "[DEBUG] Model created successfully" << std::endl;

    try {
        // Input layer for token sequences
        std::cout << "[DEBUG] Adding input layer..." << std::endl;
        model->addLayer(
            ml::train::createLayer(
            "input", {"input_shape=1:" + std::to_string(seq_len) + ":1", "name=input_tokens"})
        );
        std::cout << "[DEBUG] Input layer added successfully" << std::endl;

        // Simple embedding layer (just one FC layer)
        std::cout << "[DEBUG] Adding simple embedding layer..." << std::endl;
        model->addLayer(
            ml::train::createLayer("fully_connected", {
                "unit=" + std::to_string(model_dim),
                "name=simple_embedding"
            })
        );
        
        std::cout << "[DEBUG] Embedding layer (FC) added successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "[ERROR] Exception in basic layers: " << e.what() << std::endl;
        throw;
    }

    // Create transformer decoder layers (pre-norm + MHA + FFN with residuals)
    for (int i = 0; i < num_layers; i++) {
        std::cout << "[DEBUG] Creating transformer layer " << i << std::endl;

        try {
            // Split for residual: ln_multiout1 -> [residual tap, stream]
            std::cout << "[DEBUG] Adding ln_multiout1 for layer " << i << std::endl;
            model->addLayer(
                ml::train::createLayer("multiout", {
                    "name=layer" + std::to_string(i) + "/ln_multiout1"
                })
            );

            // LayerNorm before attention
            std::cout << "[DEBUG] Adding ln1 for layer " << i << std::endl;
            model->addLayer(
                ml::train::createLayer("layer_normalization", {
                    "axis=3",
                    "name=layer" + std::to_string(i) + "/ln1"
                })
            );

            // Split into Q, K, V for MHA
            std::cout << "[DEBUG] Adding multi_out1 (Q,K,V) for layer " << i << std::endl;
            model->addLayer(
                ml::train::createLayer("multiout", {
                    "name=layer" + std::to_string(i) + "/multi_out1"
                })
            );

            // Multi-Head Attention (decoder self-attention)
            std::cout << "[DEBUG] Adding multi_head_attention for layer " << i << std::endl;
            model->addLayer(
                ml::train::createLayer("multi_head_attention", {
                    "name=layer" + std::to_string(i) + "/multi_head_attention",
                    "input_layers=layer" + std::to_string(i) + "/multi_out1(0), layer" + std::to_string(i) + "/multi_out1(1), layer" + std::to_string(i) + "/multi_out1(2)",
                    "num_heads=" + std::to_string(num_heads)
                })
            );

            // Residual add after attention
            std::cout << "[DEBUG] Adding add1 (residual after attention) for layer " << i << std::endl;
            model->addLayer(
                ml::train::createLayer("addition", {
                    "name=layer" + std::to_string(i) + "/add1",
                    "input_layers=layer" + std::to_string(i) + "/ln_multiout1(1), layer" + std::to_string(i) + "/multi_head_attention"
                })
            );

            // Split for residual: ln_multiout2 -> [residual tap, stream]
            std::cout << "[DEBUG] Adding ln_multiout2 for layer " << i << std::endl;
            model->addLayer(
                ml::train::createLayer("multiout", {
                    "name=layer" + std::to_string(i) + "/ln_multiout2"
                })
            );

            // LayerNorm before FFN
            std::cout << "[DEBUG] Adding ln2 for layer " << i << std::endl;
            model->addLayer(
                ml::train::createLayer("layer_normalization", {
                    "axis=3",
                    "name=layer" + std::to_string(i) + "/ln2"
                })
            );

            // Split for FFN input (gate and up projections)
            std::cout << "[DEBUG] Adding multi_out3 for layer " << i << std::endl;
            model->addLayer(
                ml::train::createLayer("multiout", {
                    "name=layer" + std::to_string(i) + "/multi_out3"
                })
            );

            // SwiGLU FFN: gate projection with swish activation
            std::cout << "[DEBUG] Adding SwiGLU gate projection for layer " << i << std::endl;
            model->addLayer(
                ml::train::createLayer("fully_connected", {
                    "name=layer" + std::to_string(i) + "/fc_gate",
                    "input_layers=layer" + std::to_string(i) + "/multi_out3(0)",
                    "unit=" + std::to_string(ffn_dim),
                    "activation=swish"
                })
            );

            // SwiGLU FFN: up projection (no activation)
            std::cout << "[DEBUG] Adding SwiGLU up projection for layer " << i << std::endl;
            model->addLayer(
                ml::train::createLayer("fully_connected", {
                    "name=layer" + std::to_string(i) + "/fc_up",
                    "input_layers=layer" + std::to_string(i) + "/multi_out3(1)",
                    "unit=" + std::to_string(ffn_dim)
                })
            );

            // Element-wise multiply (gating)
            std::cout << "[DEBUG] Adding multiply (gating) for layer " << i << std::endl;
            model->addLayer(
                ml::train::createLayer("multiply", {
                    "name=layer" + std::to_string(i) + "/gating",
                    "input_layers=layer" + std::to_string(i) + "/fc_gate, layer" + std::to_string(i) + "/fc_up"
                })
            );

            // Down projection
            std::cout << "[DEBUG] Adding SwiGLU down projection for layer " << i << std::endl;
            model->addLayer(
                ml::train::createLayer("fully_connected", {
                    "name=layer" + std::to_string(i) + "/fc_down",
                    "unit=" + std::to_string(model_dim)
                })
            );

            // Residual add after FFN
            std::cout << "[DEBUG] Adding add2 (residual after SwiGLU FFN) for layer " << i << std::endl;
            model->addLayer(
                ml::train::createLayer("addition", {
                    "name=layer" + std::to_string(i) + "/add2",
                    "input_layers=layer" + std::to_string(i) + "/ln_multiout2(1), layer" + std::to_string(i) + "/fc_down"
                })
            );

        } catch (const std::exception& e) {
            std::cout << "[ERROR] Exception in transformer layer " << i << ": " << e.what() << std::endl;
            throw;
        }
    }

    try {
        // Final layer normalization
        std::cout << "[DEBUG] Adding final layer normalization..." << std::endl;
        model->addLayer(
            ml::train::createLayer("layer_normalization", {
                "axis=3",
                "name=final_ln"
            })
        );
        std::cout << "[DEBUG] Final layer normalization added successfully" << std::endl;

        // Output projection (simple regression for debugging)
        std::cout << "[DEBUG] Adding simple output projection..." << std::endl;
        model->addLayer(
            ml::train::createLayer("fully_connected", {
                "unit=1",
                "name=output_projection"
            })
        );
        std::cout << "[DEBUG] Output projection added successfully" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "[ERROR] Exception in final layers: " << e.what() << std::endl;
        throw;
    }

    // ===== GRADIENT CHECKPOINTING SETUP =====
    if (enable_checkpointing) {
        std::cout << "\n[CHECKPOINT] Setting up gradient checkpointing for transformer blocks..." << std::endl;
        
        // Create checkpoint blocks for each transformer layer
        // Each block contains all layers within one transformer block
        for (int i = 0; i < num_layers; i++) {
            std::vector<std::string> block_layers = {
                "layer" + std::to_string(i) + "/ln_multiout1",
                "layer" + std::to_string(i) + "/ln1",
                "layer" + std::to_string(i) + "/multi_out1",
                "layer" + std::to_string(i) + "/multi_head_attention",
                "layer" + std::to_string(i) + "/add1",
                "layer" + std::to_string(i) + "/ln_multiout2",
                "layer" + std::to_string(i) + "/ln2",
                "layer" + std::to_string(i) + "/multi_out3",
                "layer" + std::to_string(i) + "/fc_gate",
                "layer" + std::to_string(i) + "/fc_up",
                "layer" + std::to_string(i) + "/gating",
                "layer" + std::to_string(i) + "/fc_down",
                "layer" + std::to_string(i) + "/add2"
            };
            
            model->addCheckpointBlock(block_layers);
            std::cout << "[CHECKPOINT] Added checkpoint block for transformer layer " << i 
                      << " (" << block_layers.size() << " layers)" << std::endl;
        }
        
        std::cout << "[CHECKPOINT] Gradient checkpointing setup complete!" << std::endl;
        std::cout << "[CHECKPOINT] Total checkpoint blocks: " << num_layers << std::endl;
        std::cout << "[CHECKPOINT] Expected memory savings: ~" << (num_layers - 1) * 100 / num_layers << "%" << std::endl;
    } else {
        std::cout << "\n[INFO] Gradient checkpointing is DISABLED" << std::endl;
    }

    std::cout << "[DEBUG] Model creation completed successfully!" << std::endl;
    return model;
}

std::unique_ptr<nntrainer::util::DataLoader> getRandomDataGenerator() {
    std::cout << "[DEBUG] Data shapes - Input: [" << batch_size << ", 1, " << seq_len << ", 1]" << std::endl;
    std::cout << "[DEBUG] Data shapes - Label: [" << batch_size << ", 1, " << seq_len << ", 1]" << std::endl;
    std::cout << "[DEBUG] Using safer RandomDataLoader with post-processing..." << std::endl;
    
    std::unique_ptr<nntrainer::util::DataLoader> random_db(
        new nntrainer::util::RandomDataLoader(
            {{static_cast<unsigned int>(batch_size), 1, static_cast<unsigned int>(seq_len), 1}}, 
            {{static_cast<unsigned int>(batch_size), 1, static_cast<unsigned int>(seq_len), 1}}, 
            static_cast<unsigned int>(number_of_db)
        )
    );

    return random_db;
}

// Dataset callback
int dataset_cb(float **input, float **label, bool *last, void *user_data) {
    auto data = reinterpret_cast<nntrainer::util::DataLoader *>(user_data);
    data->next(input, label, last);
    return 0;
}

int main(int argc, char *argv[]) {
    // Parse command line arguments
    parseArguments(argc, argv);
    
    std::cout << "[DEBUG] Starting main function..." << std::endl;
    PerformanceMonitor::printMemoryInfo("Program Start");
    
    // Initialize profiler if enabled
    std::shared_ptr<nntrainer::profile::GenericProfileListener> profile_listener = nullptr;
    if (enable_profile) {
        std::cout << "[INFO] Profiling enabled - detailed statistics will be shown at the end" << std::endl;
        profile_listener = std::make_shared<nntrainer::profile::GenericProfileListener>(1);
        PROFILE_BEGIN(profile_listener);
    }
    
    try {
        // Model Creation
        auto timer_start = PerformanceMonitor::startTimer();
        std::cout << "[DEBUG] Creating model..." << std::endl;
        auto model = create_model();
        std::cout << "[DEBUG] Model created successfully!" << std::endl;
        PerformanceMonitor::printElapsed("Model Creation", timer_start);
        PerformanceMonitor::printMemoryInfo("After Model Creation");

        // Model Configuration
        timer_start = PerformanceMonitor::startTimer();
        std::cout << "[DEBUG] Setting model properties..." << std::endl;
        model->setProperty({"batch_size=" + std::to_string(batch_size),
                            "epochs=" + std::to_string(epochs)});
        std::cout << "[DEBUG] Model properties set successfully" << std::endl;

        std::cout << "[DEBUG] Creating optimizer..." << std::endl;
        auto optimizer = ml::train::createOptimizer("adam", {"learning_rate=" + std::to_string(learning_rate)});
        model->setOptimizer(std::move(optimizer));
        std::cout << "[DEBUG] Optimizer set successfully" << std::endl;
        PerformanceMonitor::printElapsed("Model Configuration", timer_start);

        // Model Compilation
        timer_start = PerformanceMonitor::startTimer();
        std::cout << "[DEBUG] Compiling model..." << std::endl;
        int status = model->compile();
        std::cout << "[DEBUG] Model compiled with status: " << status << std::endl;
        PerformanceMonitor::printElapsed("Model Compilation", timer_start);
        PerformanceMonitor::printMemoryInfo("After Compilation");
        
        // Model Initialization
        timer_start = PerformanceMonitor::startTimer();
        std::cout << "[DEBUG] Initializing model..." << std::endl;
        status = model->initialize();
        std::cout << "[DEBUG] Model initialized with status: " << status << std::endl;
        PerformanceMonitor::printElapsed("Model Initialization", timer_start);
        PerformanceMonitor::printMemoryInfo("After Initialization");

        // Data Setup
        timer_start = PerformanceMonitor::startTimer();
        std::cout << "[DEBUG] Creating data generator..." << std::endl;
        auto random_generator = getRandomDataGenerator();
        auto train_dataset = ml::train::createDataset(
        ml::train::DatasetType::GENERATOR, dataset_cb, random_generator.get());
        std::cout << "[DEBUG] Data generator created successfully" << std::endl;

        std::cout << "[DEBUG] Setting dataset..." << std::endl;
        model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, std::move(train_dataset));
        std::cout << "[DEBUG] Dataset set successfully" << std::endl;
        PerformanceMonitor::printElapsed("Data Setup", timer_start);

        // Training
        timer_start = PerformanceMonitor::startTimer();
        std::cout << "[DEBUG] Starting training..." << std::endl;
        std::cout << "[INFO] Iteration stats are printed from neuralnet.cpp (once per batch)" << std::endl;
        
        model->train();
        std::cout << "[DEBUG] Training completed successfully" << std::endl;
        PerformanceMonitor::printElapsed("Training", timer_start);
        PerformanceMonitor::printMemoryInfo("After Training");

        // Model Summary
        std::cout << "\n=== PERFORMANCE SUMMARY ===" << std::endl;
        std::cout << "Model Configuration:" << std::endl;
        std::cout << "  - Sequence Length: " << seq_len << std::endl;
        std::cout << "  - Vocab Size: " << vocab_size << std::endl;
        std::cout << "  - Model Dimension: " << model_dim << std::endl;
        std::cout << "  - Number of Heads: " << num_heads << std::endl;
        std::cout << "  - Number of Layers: " << num_layers << std::endl;
        std::cout << "  - FFN Dimension: " << ffn_dim << std::endl;
        std::cout << "  - Batch Size: " << batch_size << std::endl;
        std::cout << "  - Epochs: " << epochs << std::endl;
        std::cout << "  - Gradient Checkpointing: " << (enable_checkpointing ? "ENABLED" : "DISABLED") << std::endl;
        
        // Calculate approximate model parameters
        size_t embedding_params = seq_len * model_dim;
        size_t attention_params = num_layers * (4 * model_dim * model_dim);
        size_t ffn_params = num_layers * (2 * model_dim * ffn_dim + ffn_dim * model_dim);
        size_t total_params = embedding_params + attention_params + ffn_params;
        
        std::cout << "\nApproximate Model Parameters (SwiGLU FFN):" << std::endl;
        std::cout << "  - Embedding: " << embedding_params << std::endl;
        std::cout << "  - Attention: " << attention_params << std::endl;
        std::cout << "  - SwiGLU FFN: " << ffn_params << " (3 projections: gate+up+down)" << std::endl;
        std::cout << "  - Total: " << total_params << " (~" << (total_params * 4 / 1024 / 1024) << " MB)" << std::endl;

        // Print profiler results if enabled
        if (enable_profile && profile_listener) {
            std::cout << "\n=== PROFILER RESULTS ===" << std::endl;
            PROFILE_END(profile_listener);
        }
        
        return status;
        
    } catch (const std::exception& e) {
        std::cout << "[ERROR] Exception in main: " << e.what() << std::endl;
        PerformanceMonitor::printMemoryInfo("Error State");
        return -1;
    }
}
