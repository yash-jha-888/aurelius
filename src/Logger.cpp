#include"Logger.h"

#include<filesystem>
#include<iomanip>
#include<iostream>
#include<chrono>
#include<ctime>
#include<sstream>

Logger::Logger(const std::string& log_directory)
{
    std::filesystem::create_directories(log_directory);

    run_directory = std::filesystem::path(log_directory) / generate_timestamp();

    std::filesystem::create_directories(run_directory);

    csv_file.open(run_directory / "metrics.csv");

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

void Logger::write_run_config(const RunConfig& config)
{
    std::ofstream json(run_directory / "run_config.json");
    if (!json.is_open())
    {
        throw std::runtime_error("Failed to open run_config.json");
    }
    json << "{\n";
    json << "  \"architecture\": [";
    for (size_t i = 0; i < config.architecture.size(); i++)
    {
    json << config.architecture[i];
    if (i + 1 != config.architecture.size())
        json << ", ";
    }
    json << "],\n";
    json << "  \"activation\": \"" << config.activation << "\",\n";
    json << "  \"initializer\": \"" << config.initializer << "\",\n";
    json << "  \"optimizer\": \"" << config.optimizer << "\",\n";
    json << "  \"learning_rate\": " << config.learning_rate << ",\n";
    json << "  \"batch_size\": " << config.batch_size << ",\n";
    json << "  \"max_epochs\": " << config.epochs << ",\n";
    json << "  \"patience\": " << config.patience << ",\n";
    json << "  \"random_seed\": " << config.random_seed << "\n";
    json << "}\n";
    json.flush();
}
std::string Logger::generate_timestamp() const
{
    auto now = std::chrono::system_clock::now();
    std::time_t current_time = std::chrono::system_clock::to_time_t(now);
    
    std::chrono::system_clock::to_time_t(now);
    std::tm local_time = *std::localtime(&current_time);

    std::ostringstream oss;
    oss << std::put_time(&local_time, "%Y%m%d_%H%M%S");
    return oss.str();
}
