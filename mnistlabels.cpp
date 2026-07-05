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
    std::ifstream file("data/train-labels-idx1-ubyte", std::ios::binary);
    uint32_t magic = 0;
    file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    std::cout << "magic: " << magic << "\n";   // want 2051
    
    uint32_t num_labels = 0;
    file.read(reinterpret_cast<char*>(&num_labels), 4);
    num_labels = swap_endian(num_labels);   // will be 60000
    Eigen::MatrixXd labels = Eigen::MatrixXd::Zero(num_labels, 10);   // start all-zero
    for (int i = 0; i < num_labels; i++) {
        unsigned char label = 0;
        file.read(reinterpret_cast<char*>(&label), 1);  // read the digit (0–9)
        labels(i, label) = 1.0;                          // set the hot position
    }
    std::cout <<labels.row(0)<<"\n"; 
    return 0;
}