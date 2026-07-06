#include "Dataset.h"
#include <fstream>
#include <cstdint>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

static uint32_t swap_endian(uint32_t val) {
    return ((val >> 24) & 0xff)     |
           ((val >> 8)  & 0xff00)   |
           ((val << 8)  & 0xff0000) |
           ((val << 24) & 0xff000000);
}

Eigen::MatrixXd load_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    uint32_t magic = 0;
    file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    uint32_t num_images = 0, rows = 0, cols = 0;
    file.read(reinterpret_cast<char*>(&num_images), 4);      num_images = swap_endian(num_images);
    file.read(reinterpret_cast<char*>(&rows), 4);            rows = swap_endian(rows);
    file.read(reinterpret_cast<char*>(&cols), 4);            cols = swap_endian(cols);
    Eigen::MatrixXd images(num_images, 784);
    for (int i = 0; i < num_images; i++) {
        for (int j = 0; j < 784; j++) {
            unsigned char pixel = 0;
            file.read(reinterpret_cast<char*>(&pixel), 1);   // read ONE byte
            images(i, j) = pixel / 255.0;                     // normalize to [0,1]
         }
     }
    return images;
}

Eigen::MatrixXd load_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    uint32_t magic = 0;
    file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    
    uint32_t num_labels = 0;
    file.read(reinterpret_cast<char*>(&num_labels), 4);
    num_labels = swap_endian(num_labels);   // will be 60000
    Eigen::MatrixXd labels = Eigen::MatrixXd::Zero(num_labels, 10);   // start all-zero
    for (int i = 0; i < num_labels; i++) {
        unsigned char label = 0;
        file.read(reinterpret_cast<char*>(&label), 1);  // read the digit (0–9)
        labels(i, label) = 1.0;                          // set the hot position
    }
    return labels;
}

DataSplit split_data(const Eigen::MatrixXd& X,
                     const Eigen::MatrixXd& Y,
                     double fraction_b)
{
    int n = X.rows();

    int n_b = static_cast<int>(n * fraction_b);
    int n_a = n - n_b;

    DataSplit result;

    result.X_a = X.topRows(n_a);
    result.Y_a = Y.topRows(n_a);

    result.X_b = X.bottomRows(n_b);
    result.Y_b = Y.bottomRows(n_b);

    return result;
}

ShuffledData shuffle_data(const Eigen::MatrixXd& X,
                          const Eigen::MatrixXd& Y,
                          unsigned int seed)
{
    int n = X.rows();

    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);

    ShuffledData result;

    result.X.resizeLike(X);
    result.Y.resizeLike(Y);

    for (int i = 0; i < n; i++)
    {
        result.X.row(i) = X.row(indices[i]);
        result.Y.row(i) = Y.row(indices[i]);
    }

    return result;
}