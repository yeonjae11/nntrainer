/**
 * @file main.cpp
 * @brief Gradient Checkpointing Verification Test Application
 * 
 * This application tests the gradient checkpointing verification feature
 * by comparing forward pass outputs with recomputed outputs during backpropagation.
 */

#include <iostream>
#include <memory>
#include <vector>
#include <cstdlib>
#include <layer.h>
#include <model.h>
#include <optimizer.h>
#include <neuralnet.h>

// Test configuration
int batch_size = 4;
int seq_len = 32;
int model_dim = 64;
int num_heads = 4;
int ffn_dim = 256;
int num_layers = 2;
int epochs = 2;
int num_batches = 8;
bool enable_verification = true;
std::string model_type = "relu";  // "relu", "gelu", "swiglu"

void parseArguments(int argc, char* argv[]) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--seq_len" && i + 1 < argc) {
            seq_len = std::atoi(argv[++i]);
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
        } else if (arg == "--num_batches" && i + 1 < argc) {
            num_batches = std::atoi(argv[++i]);
        } else if (arg == "--disable-verification") {
            enable_verification = false;
        } else if (arg == "--model" && i + 1 < argc) {
            model_type = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --seq_len <int>        Sequence length (default: 32)\n"
                      << "  --model_dim <int>      Model dimension (default: 64)\n"
                      << "  --num_heads <int>      Number of attention heads (default: 4)\n"
                      << "  --num_layers <int>     Number of transformer layers (default: 2)\n"
                      << "  --ffn_dim <int>        FFN dimension (default: 256)\n"
                      << "  --batch_size <int>     Batch size (default: 4)\n"
                      << "  --epochs <int>         Number of epochs (default: 2)\n"
                      << "  --num_batches <int>    Number of batches (default: 8)\n"
                      << "  --model <type>         Model type: relu, gelu, swiglu (default: relu)\n"
                      << "  --disable-verification Disable checkpoint verification\n"
                      << "  --help, -h             Show this help message\n";
            exit(0);
        }
    }
    
    // Validation
    if (model_dim % num_heads != 0) {
        std::cerr << "Error: model_dim (" << model_dim << ") must be divisible by num_heads (" 
                  << num_heads << ")" << std::endl;
        exit(1);
    }
}

std::unique_ptr<ml::train::Model> createTestModel() {
    std::cout << "\n[TEST] Creating test model with gradient checkpointing..." << std::endl;
    std::cout << "[TEST] Model type: " << model_type << std::endl;
    
    auto model = ml::train::createModel(ml::train::ModelType::NEURAL_NET, {"loss=mse"});
    
    // Input layer
    model->addLayer(
        ml::train::createLayer(
            "input", {"input_shape=1:" + std::to_string(seq_len) + ":1", "name=input_tokens"})
    );
    
    // Simple embedding
    model->addLayer(
        ml::train::createLayer("fully_connected", {
            "unit=" + std::to_string(model_dim),
            "name=embedding"
        })
    );
    
    // Create transformer layers with checkpointing
    for (int i = 0; i < num_layers; i++) {
        std::string prefix = "layer" + std::to_string(i);
        
        // Pre-norm + MHA + Residual
        model->addLayer(ml::train::createLayer("multiout", {"name=" + prefix + "/ln_multiout1"}));
        model->addLayer(ml::train::createLayer("layer_normalization", {"axis=3", "name=" + prefix + "/ln1"}));
        model->addLayer(ml::train::createLayer("multiout", {"name=" + prefix + "/multi_out1"}));
        model->addLayer(ml::train::createLayer("multi_head_attention", {
            "name=" + prefix + "/mha",
            "input_layers=" + prefix + "/multi_out1(0)," + prefix + "/multi_out1(1)," + prefix + "/multi_out1(2)",
            "num_heads=" + std::to_string(num_heads)
        }));
        model->addLayer(ml::train::createLayer("addition", {
            "name=" + prefix + "/add1",
            "input_layers=" + prefix + "/ln_multiout1(1)," + prefix + "/mha"
        }));
        
        // Pre-norm + FFN + Residual (different based on model_type)
        model->addLayer(ml::train::createLayer("multiout", {"name=" + prefix + "/ln_multiout2"}));
        model->addLayer(ml::train::createLayer("layer_normalization", {"axis=3", "name=" + prefix + "/ln2"}));
        
        if (model_type == "swiglu") {
            // SwiGLU FFN: gate_proj and up_proj with swish activation
            // gate = swish(x @ W_gate)
            // up = x @ W_up
            // output = (gate * up) @ W_down
            model->addLayer(ml::train::createLayer("fully_connected", {
                "unit=" + std::to_string(ffn_dim),
                "name=" + prefix + "/gate_proj"
            }));
            model->addLayer(ml::train::createLayer("activation", {
                "activation=swish",
                "name=" + prefix + "/gate_act"
            }));
            model->addLayer(ml::train::createLayer("fully_connected", {
                "unit=" + std::to_string(ffn_dim),
                "input_layers=" + prefix + "/ln2",
                "name=" + prefix + "/up_proj"
            }));
            model->addLayer(ml::train::createLayer("multiply", {
                "name=" + prefix + "/glu_mul",
                "input_layers=" + prefix + "/gate_act," + prefix + "/up_proj"
            }));
            model->addLayer(ml::train::createLayer("fully_connected", {
                "unit=" + std::to_string(model_dim),
                "name=" + prefix + "/down_proj"
            }));
            model->addLayer(ml::train::createLayer("addition", {
                "name=" + prefix + "/add2",
                "input_layers=" + prefix + "/ln_multiout2(1)," + prefix + "/down_proj"
            }));
        } else {
            // Standard FFN with relu or gelu
            std::string activation = (model_type == "gelu") ? "gelu" : "relu";
            model->addLayer(ml::train::createLayer("fully_connected", {
                "unit=" + std::to_string(ffn_dim),
                "activation=" + activation,
                "name=" + prefix + "/fc1"
            }));
            model->addLayer(ml::train::createLayer("fully_connected", {
                "unit=" + std::to_string(model_dim),
                "name=" + prefix + "/fc2"
            }));
            model->addLayer(ml::train::createLayer("addition", {
                "name=" + prefix + "/add2",
                "input_layers=" + prefix + "/ln_multiout2(1)," + prefix + "/fc2"
            }));
        }
    }
    
    // Output layer
    model->addLayer(ml::train::createLayer("layer_normalization", {"axis=3", "name=final_ln"}));
    model->addLayer(ml::train::createLayer("fully_connected", {"unit=1", "name=output"}));
    
    // Add checkpoint blocks
    std::cout << "[TEST] Adding checkpoint blocks..." << std::endl;
    for (int i = 0; i < num_layers; i++) {
        std::string prefix = "layer" + std::to_string(i);
        std::vector<std::string> block_layers;
        
        if (model_type == "swiglu") {
            block_layers = {
                prefix + "/ln_multiout1",
                prefix + "/ln1",
                prefix + "/multi_out1",
                prefix + "/mha",
                prefix + "/add1",
                prefix + "/ln_multiout2",
                prefix + "/ln2",
                prefix + "/gate_proj",
                prefix + "/gate_act",
                prefix + "/up_proj",
                prefix + "/glu_mul",
                prefix + "/down_proj",
                prefix + "/add2"
            };
        } else {
            block_layers = {
                prefix + "/ln_multiout1",
                prefix + "/ln1",
                prefix + "/multi_out1",
                prefix + "/mha",
                prefix + "/add1",
                prefix + "/ln_multiout2",
                prefix + "/ln2",
                prefix + "/fc1",
                prefix + "/fc1/activation_realized",  // Add auto-generated activation layer
                prefix + "/fc2",
                prefix + "/add2"
            };
        }
        model->addCheckpointBlock(block_layers);
        std::cout << "[TEST] Added checkpoint block for " << prefix 
                  << " (" << block_layers.size() << " layers)" << std::endl;
    }
    
    return model;
}

class SimpleDataGenerator {
public:
    SimpleDataGenerator(int batch_size, int seq_len, int num_batches)
        : batch_size_(batch_size), seq_len_(seq_len), num_batches_(num_batches), current_batch_(0) {
        
        // Allocate memory for input and label
        input_data_.resize(batch_size * seq_len);
        label_data_.resize(batch_size * seq_len);
        
        // Initialize with simple pattern
        for (int i = 0; i < batch_size * seq_len; i++) {
            input_data_[i] = static_cast<float>(i % 10) / 10.0f;
            label_data_[i] = static_cast<float>((i + 1) % 10) / 10.0f;
        }
    }
    
    void next(float **input, float **label, bool *last) {
        // Always provide data
        *input = input_data_.data();
        *label = label_data_.data();
        
        // Set last flag when we reach the end
        *last = (current_batch_ >= num_batches_ - 1);
        
        current_batch_++;
        
        // Reset for next epoch
        if (current_batch_ >= num_batches_) {
            current_batch_ = 0;
        }
    }
    
    void reset() {
        current_batch_ = 0;
    }
    
private:
    int batch_size_;
    int seq_len_;
    int num_batches_;
    int current_batch_;
    std::vector<float> input_data_;
    std::vector<float> label_data_;
};

int dataset_cb(float **input, float **label, bool *last, void *user_data) {
    auto* generator = reinterpret_cast<SimpleDataGenerator*>(user_data);
    generator->next(input, label, last);
    return 0;  // Always return success
}

int main(int argc, char* argv[]) {
    // Parse arguments
    parseArguments(argc, argv);
    
    std::cout << "========================================" << std::endl;
    std::cout << "Gradient Checkpointing Verification Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "[CONFIG] Test Configuration:" << std::endl;
    std::cout << "  - Model type: " << model_type << std::endl;
    std::cout << "  - Batch size: " << batch_size << std::endl;
    std::cout << "  - Sequence length: " << seq_len << std::endl;
    std::cout << "  - Model dimension: " << model_dim << std::endl;
    std::cout << "  - Number of heads: " << num_heads << std::endl;
    std::cout << "  - Number of layers: " << num_layers << std::endl;
    std::cout << "  - FFN dimension: " << ffn_dim << std::endl;
    std::cout << "  - Epochs: " << epochs << std::endl;
    std::cout << "  - Number of batches: " << num_batches << std::endl;
    std::cout << "  - Verification: " << (enable_verification ? "ENABLED" : "DISABLED") << std::endl;
    
    try {
        // Create model
        auto model = createTestModel();
        
        // Configure model
        std::cout << "\n[TEST] Configuring model..." << std::endl;
        model->setProperty({
            "batch_size=" + std::to_string(batch_size),
            "epochs=" + std::to_string(epochs)
        });
        
        auto optimizer = ml::train::createOptimizer("adam", {"learning_rate=0.001"});
        model->setOptimizer(std::move(optimizer));
        
        // Compile and initialize
        std::cout << "[TEST] Compiling model..." << std::endl;
        int status = model->compile();
        if (status != 0) {
            std::cerr << "[ERROR] Model compilation failed with status: " << status << std::endl;
            return -1;
        }
        
        std::cout << "[TEST] Initializing model..." << std::endl;
        status = model->initialize();
        if (status != 0) {
            std::cerr << "[ERROR] Model initialization failed with status: " << status << std::endl;
            return -1;
        }
        
        // Enable checkpoint verification
        if (enable_verification) {
            std::cout << "\n[TEST] Enabling checkpoint verification..." << std::endl;
            auto* neural_net = dynamic_cast<nntrainer::NeuralNetwork*>(model.get());
            if (neural_net) {
                auto& network_graph = neural_net->getNetworkGraph();
                network_graph.enableCheckpointVerification(true);
                
                std::cout << "[TEST] Checkpoint verification ENABLED" << std::endl;
            } else {
                std::cerr << "[WARNING] Could not access NetworkGraph - verification disabled" << std::endl;
            }
        }
        
        // Setup dataset
        std::cout << "\n[TEST] Setting up dataset..." << std::endl;
        SimpleDataGenerator data_gen(batch_size, seq_len, num_batches);
        
        auto train_dataset = ml::train::createDataset(
            ml::train::DatasetType::GENERATOR, dataset_cb, &data_gen);
        model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, std::move(train_dataset));
        
        // Train
        std::cout << "\n[TEST] Starting training..." << std::endl;
        std::cout << "========================================" << std::endl;
        
        model->train();
        
        std::cout << "========================================" << std::endl;
        std::cout << "[TEST] Training completed successfully!" << std::endl;
        
        // Print verification statistics
        if (enable_verification) {
            std::cout << "\n[TEST] Printing verification statistics..." << std::endl;
            auto* neural_net = dynamic_cast<nntrainer::NeuralNetwork*>(model.get());
            if (neural_net) {
                auto network_graph = neural_net->getNetworkGraph();
                network_graph.printVerificationStats();
                network_graph.clearSavedOutputs();
            }
        }
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "Test completed successfully!" << std::endl;
        std::cout << "========================================" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception: " << e.what() << std::endl;
        return -1;
    }
}
