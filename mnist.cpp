#include <fstream>
#include <cstdint>
#include <iostream>
#include <Eigen/Dense>
uint32_t swap_endian(uint32_t val) {
    return ((val >> 24) & 0xff)     |
           ((val >> 8)  & 0xff00)   |
           ((val << 8)  & 0xff0000) |
           ((val << 24) & 0xff000000);
}
int main(){
    std::ifstream file("data/train-images-idx3-ubyte", std::ios::binary);
    uint32_t magic = 0;
    file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    std::cout << "magic: " << magic << "\n";   // want 2051
    uint32_t num_images = 0, rows = 0, cols = 0;
    file.read(reinterpret_cast<char*>(&num_images), 4);      num_images = swap_endian(num_images);
    file.read(reinterpret_cast<char*>(&rows), 4);            rows = swap_endian(rows);
    file.read(reinterpret_cast<char*>(&cols), 4);            cols = swap_endian(cols);
    std::cout << num_images << " images, " << rows << "x" << cols << "\n";

    Eigen::MatrixXd images(num_images, 784);
    for (int i = 0; i < num_images; i++) {
        for (int j = 0; j < 784; j++) {
            unsigned char pixel = 0;
            file.read(reinterpret_cast<char*>(&pixel), 1);   // read ONE byte
            images(i, j) = pixel / 255.0;                     // normalize to [0,1]
         }
     } 
    for (int r = 0; r < 28; r++) {
         for (int c = 0; c < 28; c++) {
             double v = images(0, r*28 + c);   // pixel (r,c) of image 0
             std::cout << (v > 0.5 ? '#' : ' ');
         }
         std::cout << "\n"; 
        }

    return 0;
}