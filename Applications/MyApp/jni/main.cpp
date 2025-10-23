#include <layer.h>
#include <model.h>
#include <optimizer.h>
#include <cifar_dataloader.h>

const int number_of_db = 16;
const int batch_size = 4;
const int epochs = 10;
const float learning_rate = 0.001;

std::unique_ptr<ml::train::Model> create_model() {
    std::unique_ptr<ml::train::Model> model = 
        ml::train::createModel(ml::train::ModelType::NEURAL_NET, {"loss=mse"});

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
        new nntrainer::util::RandomDataLoader({{batch_size, 1, 1, 10}}, {{batch_size, 1, 1, 1}}, number_of_db));

    return random_db;
}

int dataset_cb(float **input, float **label, bool *last, void *user_data) {
    auto data = reinterpret_cast<nntrainer::util::DataLoader *>(user_data);

    data->next(input, label, last);
    return 0;
}

int main(int argc, char *argv[]) {
    auto model = create_model();

    model->setProperty({"batch_size=" + std::to_string(batch_size),
                        "epochs=" + std::to_string(epochs),
                        "save_path=my_app.bin"});

    auto optimizer = ml::train::createOptimizer("SGD", {"learning_rate=" + std::to_string(learning_rate)});
    model->setOptimizer(std::move(optimizer));

    int status = model->compile();
    status = model->initialize();

    auto random_generator = getRandomDataGenerator();
    auto train_dataset = ml::train::createDataset(
    ml::train::DatasetType::GENERATOR, dataset_cb, random_generator.get());

    model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, std::move(train_dataset));

    model->train();

    return status;
}