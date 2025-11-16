#include <layer.h>
#include <model.h>
#include <optimizer.h>
#include <cifar_dataloader.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static int number_of_db = 16;
static int batch_size = 4;
static int epochs = 1;
static float learning_rate = 0.001f;
static std::string optimizer_type = "lion";
static float weight_decay = 0.0f;

std::unique_ptr<ml::train::Model> create_model() {
    std::vector<std::string> model_props = {"loss=mse", "model_tensor_type=FP32-FP32"};
    std::unique_ptr<ml::train::Model> model =
        ml::train::createModel(ml::train::ModelType::NEURAL_NET, model_props);

    model->addLayer(
        ml::train::createLayer(
        "input", {"input_shape=1:1:10"})
    );

    model->addLayer(
        ml::train::createLayer("fully_connected", {"unit=5", "activation=softmax"}));

    return model;
}

std::unique_ptr<nntrainer::util::DataLoader> getRandomDataGenerator() {
    std::unique_ptr<nntrainer::util::DataLoader> random_db(
        new nntrainer::util::RandomDataLoader(
            {{static_cast<unsigned int>(batch_size), 1u, 1u, 10u}},
            {{static_cast<unsigned int>(batch_size), 1u, 1u, 1u}},
            static_cast<unsigned int>(number_of_db)));

    return random_db;
}

int dataset_cb(float **input, float **label, bool *last, void *user_data) {
    auto data = reinterpret_cast<nntrainer::util::DataLoader *>(user_data);

    data->next(input, label, last);
    return 0;
}

static void parse_args(int argc, char *argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        auto starts_with = [&](const char *prefix) {
            return arg.rfind(prefix, 0) == 0;
        };
        if (starts_with("--wd=")) {
            weight_decay = std::stof(arg.substr(5));
        } else if (starts_with("--epochs=")) {
            epochs = std::stoi(arg.substr(9));
        } else if (starts_with("--db=")) {
            number_of_db = std::stoi(arg.substr(5));
        } else if (starts_with("--bs=")) {
            batch_size = std::stoi(arg.substr(5));
        } else if (starts_with("--lr=")) {
            learning_rate = std::stof(arg.substr(5));
        } else if (starts_with("--opt=")) {
            optimizer_type = arg.substr(6);
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: nntrainer_myapp [--wd=<float>] [--opt=lion|adam|adamw|sgd]\n"
                   "                        [--epochs=<int>] [--db=<int>] [--bs=<int>] [--lr=<float>]\n";
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
    model.forEachLayer(
        [&](ml::train::Layer &layer, nntrainer::RunLayerContext & /*rc*/, void * /*ud*/) {
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

static WeightStats measure_delta_l2(ml::train::Model &model,
                                    const std::vector<std::vector<float>> &snapshot) {
    WeightStats ws{};
    size_t idx = 0;
    model.forEachLayer(
        [&](ml::train::Layer &layer, nntrainer::RunLayerContext & /*rc*/, void * /*ud*/) {
            std::vector<float *> weights;
            std::vector<ml::train::TensorDim> dims;
            layer.getWeights(weights, dims);
            for (size_t wi = 0; wi < weights.size(); ++wi) {
                float *p = weights[wi];
                size_t n = dims[wi].getDataLen();
                ws.num_elems += n;
                double acc = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    double d = static_cast<double>(p[k]) - static_cast<double>(snapshot[idx][k]);
                    acc += d * d;
                }
                ws.l2 += acc;
                ++idx;
            }
        });
    return ws;
}

static std::vector<std::vector<float>> snapshot_weights(ml::train::Model &model) {
    std::vector<std::vector<float>> snap;
    model.forEachLayer(
        [&](ml::train::Layer &layer, nntrainer::RunLayerContext & /*rc*/, void * /*ud*/) {
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

    auto model = create_model();

    model->setProperty({"batch_size=" + std::to_string(batch_size),
                        "epochs=" + std::to_string(epochs)});

    std::vector<std::string> opt_props = {"learning_rate=" + std::to_string(learning_rate)};
    if (optimizer_type == "lion") {
        if (weight_decay > 0.0f) {
            opt_props.emplace_back("weight_decay=" + std::to_string(weight_decay));
        }
    }
    auto optimizer = ml::train::createOptimizer(optimizer_type, opt_props);
    model->setOptimizer(std::move(optimizer));

    int status = model->compile();
    status = model->initialize();

    auto random_generator = getRandomDataGenerator();
    auto train_dataset = ml::train::createDataset(
    ml::train::DatasetType::GENERATOR, dataset_cb, random_generator.get());

    model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, std::move(train_dataset));

    // Baseline weight norms and snapshot before training
    auto snap_before = snapshot_weights(*model);
    auto ws_before = measure_weight_l2(*model);
    std::cout << "[DEBUG] weights_l2_before=" << ws_before.l2
              << " elems=" << ws_before.num_elems
              << " opt=" << optimizer_type
              << " wd=" << weight_decay
              << " lr=" << learning_rate
              << std::endl;

    model->train();

    // Measure deltas after training
    auto ws_after = measure_weight_l2(*model);
    auto delta = measure_delta_l2(*model, snap_before);
    std::cout << "[DEBUG] weights_l2_after=" << ws_after.l2
              << " delta_l2=" << delta.l2
              << " elems=" << delta.num_elems
              << std::endl;

    return status;
}