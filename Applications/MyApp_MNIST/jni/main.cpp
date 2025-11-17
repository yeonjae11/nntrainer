#include <dataset.h>
#include <layer.h>
#include <model.h>
#include <optimizer.h>
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// Default: empty (resolved at runtime or via CLI flags)
static std::string g_config_path;
static std::string g_data_path;

// Common training/batch/optimizer settings (for controlled comparisons)
static unsigned int total_train_data_size = 100;
static unsigned int total_val_data_size   = 100;
static unsigned int batch_size = 32;
static unsigned int epochs     = 5;
static std::string  optimizer_type = "lion"; // lion|adam|adamw|sgd|sophia
static float        learning_rate  = 1e-3f;
static float        weight_decay   = 0.0f;

constexpr unsigned int SEED = 0;
constexpr unsigned int feature_size = 784;
constexpr unsigned int total_label_size = 10;

class DataInformation {
public:
  DataInformation(unsigned int num_samples, const std::string &filename) :
    count(0),
    num_samples(num_samples),
    file(filename, std::ios::in | std::ios::binary),
    idxes(num_samples) {
    std::iota(idxes.begin(), idxes.end(), 0);
    rng.seed(SEED);
    std::shuffle(idxes.begin(), idxes.end(), rng);
    if (!file.good()) {
      throw std::invalid_argument("Data file is not readable: " + filename);
    }
  }
  unsigned int count;
  unsigned int num_samples;
  std::ifstream file;
  std::vector<unsigned int> idxes;
  std::mt19937 rng;
};

static bool getData(std::ifstream &F, float *input, float *label, unsigned int id) {
  F.clear();
  F.seekg(0, std::ios_base::end);
  uint64_t file_length = F.tellg();
  uint64_t position =
    static_cast<uint64_t>((feature_size + total_label_size) * static_cast<uint64_t>(id) * sizeof(float));

  if (position > file_length) {
    return false;
  }
  F.seekg(position, std::ios::beg);
  F.read(reinterpret_cast<char *>(input), sizeof(float) * feature_size);
  F.read(reinterpret_cast<char *>(label), sizeof(float) * total_label_size);
  return true;
}

static int getSample(float **outVec, float **outLabel, bool *last, void *user_data) {
  auto data = reinterpret_cast<DataInformation *>(user_data);
  getData(data->file, *outVec, *outLabel, data->idxes.at(data->count));
  data->count++;
  if (data->count < data->num_samples) {
    *last = false;
  } else {
    *last = true;
    data->count = 0;
    std::shuffle(data->idxes.begin(), data->idxes.end(), data->rng);
  }
  return 0;
}

static std::string try_find_in_ancestors(const std::filesystem::path &start_base,
                                         const std::filesystem::path &relative_subpath,
                                         int max_up = 6) {
  std::filesystem::path base = start_base;
  for (int i = 0; i <= max_up; ++i) {
    std::filesystem::path candidate = base / relative_subpath;
    std::error_code ec;
    if (std::filesystem::exists(candidate, ec)) {
      return candidate.string();
    }
    base = base.parent_path();
  }
  return {};
}

static void parse_args(int argc, char *argv[]) {
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    auto starts_with = [&](const char *prefix) { return arg.rfind(prefix, 0) == 0; };
    if (starts_with("--config=")) {
      g_config_path = arg.substr(9);
    } else if (starts_with("--data=")) {
      g_data_path = arg.substr(7);
    } else if (starts_with("--opt=")) {
      optimizer_type = arg.substr(6);
    } else if (starts_with("--lr=")) {
      learning_rate = std::stof(arg.substr(5));
    } else if (starts_with("--wd=")) {
      weight_decay = std::stof(arg.substr(5));
    } else if (starts_with("--epochs=")) {
      epochs = static_cast<unsigned int>(std::stoul(arg.substr(9)));
    } else if (starts_with("--bs=")) {
      batch_size = static_cast<unsigned int>(std::stoul(arg.substr(5)));
    } else if (starts_with("--train_size=")) {
      total_train_data_size = static_cast<unsigned int>(std::stoul(arg.substr(13)));
    } else if (starts_with("--val_size=")) {
      total_val_data_size = static_cast<unsigned int>(std::stoul(arg.substr(11)));
    } else if (arg == "--help" || arg == "-h") {
      std::cout
        << "Usage: nntrainer_myapp_mnist [--config=<ini>] [--data=<dat>] [--opt=lion|adam|adamw|sgd|sophia]\n"
           "                               [--lr=<float>] [--wd=<float>] [--epochs=<uint>] [--bs=<uint>]\n"
           "                               [--train_size=<uint>] [--val_size=<uint>]\n";
      std::exit(0);
    }
  }
}

struct WeightStats {
  double l2 = 0.0;
  size_t num_elems = 0;
};

static WeightStats measure_weight_l2(ml::train::Model &model) {
  WeightStats ws{};
  model.forEachLayer([&](ml::train::Layer &layer, nntrainer::RunLayerContext & /*rc*/, void * /*ud*/) {
    std::vector<float *> weights;
    std::vector<ml::train::TensorDim> dims;
    layer.getWeights(weights, dims);
    for (size_t wi = 0; wi < weights.size(); ++wi) {
      float *p = weights[wi];
      size_t n = dims[wi].getDataLen();
      ws.num_elems += n;
      double acc = 0.0;
      for (size_t k = 0; k < n; ++k) {
        double v = static_cast<double>(p[k]);
        acc += v * v;
      }
      ws.l2 += acc;
    }
  });
  return ws;
}

static std::vector<std::vector<float>> snapshot_weights(ml::train::Model &model) {
  std::vector<std::vector<float>> snap;
  model.forEachLayer([&](ml::train::Layer &layer, nntrainer::RunLayerContext & /*rc*/, void * /*ud*/) {
    std::vector<float *> weights;
    std::vector<ml::train::TensorDim> dims;
    layer.getWeights(weights, dims);
    for (size_t wi = 0; wi < weights.size(); ++wi) {
      float *p = weights[wi];
      size_t n = dims[wi].getDataLen();
      std::vector<float> buf(n);
      std::copy(p, p + n, buf.begin());
      snap.emplace_back(std::move(buf));
    }
  });
  return snap;
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);

  // Resolve default paths if not provided
  {
    std::filesystem::path cwd = std::filesystem::current_path();
    // prefer local copies (placed next to executable by Meson)
    if (g_config_path.empty()) {
      std::filesystem::path local_ini = cwd / "mnist.ini";
      if (std::filesystem::exists(local_ini)) g_config_path = local_ini.string();
    }
    if (g_data_path.empty()) {
      std::filesystem::path local_dat = cwd / "mnist_trainingSet.dat";
      if (std::filesystem::exists(local_dat)) g_data_path = local_dat.string();
    }
    // fallback: search ancestors for Applications/MNIST/res/*
    if (g_config_path.empty() || !std::filesystem::exists(g_config_path)) {
      std::string found = try_find_in_ancestors(
        cwd, std::filesystem::path("Applications/MNIST/res/mnist.ini"));
      if (!found.empty()) g_config_path = found;
    }
    if (g_data_path.empty() || !std::filesystem::exists(g_data_path)) {
      std::string found = try_find_in_ancestors(
        cwd, std::filesystem::path("Applications/MNIST/res/mnist_trainingSet.dat"));
      if (!found.empty()) g_data_path = found;
    }
    if (g_config_path.empty() || !std::filesystem::exists(g_config_path) ||
        g_data_path.empty() || !std::filesystem::exists(g_data_path)) {
      std::cerr << "MNIST config/data not found.\n"
                << "Provide paths explicitly:\n"
                << "  --config=/path/to/mnist.ini --data=/path/to/mnist_trainingSet.dat\n";
      return 1;
    }
  }

  // Prepare user data holders
  std::unique_ptr<DataInformation> train_user_data;
  std::unique_ptr<DataInformation> valid_user_data;
  try {
    train_user_data = std::make_unique<DataInformation>(total_train_data_size, g_data_path);
    valid_user_data = std::make_unique<DataInformation>(total_val_data_size, g_data_path);
  } catch (const std::exception &e) {
    std::cerr << "Error creating userdata: " << e.what() << std::endl;
    return 1;
  }

  // Create datasets via generator callback
  std::shared_ptr<ml::train::Dataset> dataset_train, dataset_val;
  try {
    dataset_train = ml::train::createDataset(ml::train::DatasetType::GENERATOR, getSample,
                                             train_user_data.get());
    dataset_val = ml::train::createDataset(ml::train::DatasetType::GENERATOR, getSample,
                                           valid_user_data.get());
  } catch (const std::exception &e) {
    std::cerr << "Error creating dataset: " << e.what() << std::endl;
    return 1;
  }

  // Load model from INI and override optimizer
  std::unique_ptr<ml::train::Model> model;
  try {
    model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
    model->load(g_config_path, ml::train::ModelFormat::MODEL_FORMAT_INI_WITH_BIN);
  } catch (const std::exception &e) {
    std::cerr << "Error loading config: " << e.what() << std::endl;
    return 1;
  }

  try {
    std::vector<std::string> opt_props = {"learning_rate=" + std::to_string(learning_rate)};
    if ((optimizer_type == "lion" || optimizer_type == "adamw" || optimizer_type == "sophia") &&
        weight_decay > 0.0f) {
      opt_props.emplace_back("weight_decay=" + std::to_string(weight_decay));
    }
    auto optimizer = ml::train::createOptimizer(optimizer_type, opt_props);
    model->setOptimizer(std::move(optimizer));
  } catch (const std::exception &e) {
    std::cerr << "Error setting optimizer: " << e.what() << std::endl;
    return 1;
  }

  // Set common training properties
  try {
    model->setProperty({"batch_size=" + std::to_string(batch_size),
                        "epochs=" + std::to_string(epochs)});
  } catch (const std::exception &e) {
    std::cerr << "Error setting model properties: " << e.what() << std::endl;
    return 1;
  }

  // Compile / initialize / attach datasets
  try {
    model->compile();
    model->initialize();
    model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset_train);
    model->setDataset(ml::train::DatasetModeType::MODE_VALID, dataset_val);
  } catch (const std::exception &e) {
    std::cerr << "Error initializing model: " << e.what() << std::endl;
    return 1;
  }

  // Measure L2 norm of weights before training
  auto snap_before = snapshot_weights(*model);
  auto ws_before = measure_weight_l2(*model);
  std::cout << "[MNIST] before_l2=" << ws_before.l2
            << " elems=" << ws_before.num_elems
            << " opt=" << optimizer_type
            << " lr=" << learning_rate
            << " wd=" << weight_decay
            << " epochs=" << epochs
            << " bs=" << batch_size
            << std::endl;

  // Train
  try {
    model->train();
  } catch (const std::exception &e) {
    std::cerr << "Error during train: " << e.what() << std::endl;
    return 1;
  }

  // Measure L2 norm and delta after training
  auto ws_after = measure_weight_l2(*model);
  // Compute delta L2
  size_t idx = 0;
  double delta_l2 = 0.0;
  model->forEachLayer([&](ml::train::Layer &layer, nntrainer::RunLayerContext & /*rc*/, void * /*ud*/) {
    std::vector<float *> weights;
    std::vector<ml::train::TensorDim> dims;
    layer.getWeights(weights, dims);
    for (size_t wi = 0; wi < weights.size(); ++wi) {
      float *p = weights[wi];
      size_t n = dims[wi].getDataLen();
      double acc = 0.0;
      for (size_t k = 0; k < n; ++k) {
        double d = static_cast<double>(p[k]) - static_cast<double>(snap_before[idx][k]);
        acc += d * d;
      }
      delta_l2 += acc;
      ++idx;
    }
  });

  std::cout << "[MNIST] after_l2=" << ws_after.l2
            << " delta_l2=" << delta_l2
            << " elems=" << ws_after.num_elems
            << std::endl;

  return 0;
}


