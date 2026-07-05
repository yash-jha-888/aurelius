#include <fstream>
#include <cstdint>
#include <iostream>
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
    return 0;
}