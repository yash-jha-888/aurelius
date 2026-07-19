#include "Model.h"

Model::Model(
    const std::vector<int>& sizes,
    std::unique_ptr<Activation> act,
    std::unique_ptr<Initializer> init,
    std::unique_ptr<Optimizer> opt
)
    : activation(std::move(act)), initializer(std::move(init)), optimizer(std::move(opt))
{
    for (int i = 0; i < sizes.size() - 1; i++)
    {
        layers.emplace_back(sizes[i], sizes[i + 1], *initializer);
    }
}
Model::Model(const Model& other)
: layers(other.layers),
  activation(other.activation ? other.activation->clone() : nullptr),
  initializer(other.initializer ? other.initializer->clone() : nullptr),
  optimizer(other.optimizer ? other.optimizer->clone() : nullptr)
{
}

Model& Model::operator=(const Model& other)
{
    if (this == &other)
    {
        return *this;
    }

    layers = other.layers;
    initializer = other.initializer ? other.initializer->clone() : nullptr;
    optimizer = other.optimizer ? other.optimizer->clone() : nullptr;
    if (other.activation)
    {
        activation = other.activation->clone();
    }
    else
    {
        activation.reset();
    }

    return *this;
}

Eigen::MatrixXd Model::forward(const Eigen::MatrixXd& X) {
    Eigen::MatrixXd current = X;
    for (int i = 0; i < layers.size(); i++) {
        current = layers[i].forward(current);         
        if (i < layers.size() - 1) {
            current = activation->forward(current);
        }
    }
    return current;
}

void Model::backward(const Eigen::MatrixXd& output_delta)
{
     const double n = output_delta.rows();
     Eigen::MatrixXd delta = output_delta;
     for (int i = layers.size() - 1; i >= 0; i--) {
         if (i < layers.size() - 1) {
             Eigen::MatrixXd mask =
             activation->derivative(layers[i].cached_Z);

             delta =
             (delta * layers[i + 1].W.transpose()).array()
             *
             mask.array();  
         }
         layers[i].gradW = layers[i].cached_input.transpose() * delta;
         layers[i].gradb = delta.colwise().sum();
     }
}

void Model::update(double learning_rate)
{
for (auto& layer : layers)
{
    optimizer->update(
        layer.W,
        layer.b,
        layer.gradW,
        layer.gradb,
        learning_rate
    );
}
}