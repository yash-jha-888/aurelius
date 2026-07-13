#pragma once

#include <string>
#include <fstream>

struct EpochMetrics
{
    int epoch;

    double train_loss;
    double train_acc;
    double val_loss;
    double val_acc;

    double learning_rate;
    long long epoch_time_ms;
};

class Logger
{
public:
    Logger();

    void log_epoch(const EpochMetrics& metrics);

private:
    std::ofstream csv_file;
};