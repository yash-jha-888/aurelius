#include"Logger.h"

#include<filesystem>
#include<iomanip>
#include<iostream>
#include<chrono>

Logger::Logger()
{
    std::filesystem::create_directories("logs");
    csv_file.open("logs/metrics.csv");

    if (!csv_file.is_open())
    {
        throw std::runtime_error("Failed to open metrics.csv ");
    }

    csv_file << std::fixed << std::setprecision(6);

    csv_file << "epoch,"
             << "train_loss,"
             << "train_acc,"
             << "val_loss,"
             << "val_acc,"
             << "learning_rate,"
             << "epoch_time_ms\n";
}

void Logger::log_epoch(const EpochMetrics& metrics)
{
    // Terminal output

    std::cout << "Epoch " << metrics.epoch
              << ", Train Loss: " << metrics.train_loss
              << ", Train Acc: " << metrics.train_acc * 100 << "%"
              << ", Val Loss: " << metrics.val_loss
              << ", Val Acc: " << metrics.val_acc * 100 << "%"
              << ", Learning Rate: " << metrics.learning_rate
              << ", Epoch Time: " << metrics.epoch_time_ms / 1000.0 << "s"
              << std::endl;
    
    // CSV output
    csv_file << metrics.epoch << ","
             << metrics.train_loss << ","
             << metrics.train_acc << ","
             << metrics.val_loss << ","
             << metrics.val_acc << ","
             << metrics.learning_rate << ","
             << metrics.epoch_time_ms << "\n";
    csv_file.flush();
}
