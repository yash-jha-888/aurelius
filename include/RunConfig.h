#pragma once

#include <string>
#include <vector>

struct RunConfig
{
    std::vector<int> architecture;

    std::string activation;
    std::string initializer;
    std::string optimizer;

    double learning_rate;

    int batch_size;
    int epochs;
    int patience;

    int random_seed;
};