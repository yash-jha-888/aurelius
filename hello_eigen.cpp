#include<iostream>
#include<Eigen/Dense>

using namespace std; 
int main() {
    Eigen::MatrixXd A(2,2);
    A << 1, 2,
         3, 4;
    Eigen::MatrixXd B(2,2);
    B << 5, 6,
         7, 8;

    Eigen::MatrixXd C = A * B;

    cout << "A x B =\n" << C << endl;

    return 0;
}