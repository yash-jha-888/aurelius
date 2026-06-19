#include<iostream>
#include<Eigen/Dense>

using namespace std;
using namespace Eigen;

int main(){

    VectorXd v(3);

    v << 1,2,3;

    cout << "v = \n" << v << "\n\n";
    
    
    
    MatrixXd A(2,3);
    A << 1,2,3,
         4,5, 6;

    cout << "A = \n" << A << "\n\n";

    MatrixXd B(3,2);

    B << 7,8,
         9, 10,
         11, 12;

    cout << "B = \n" << B << "\n\n";

    MatrixXd C=A*B;
    cout << "A x B =\n" << C << "\n\n";

    VectorXd a(3);

    a << 1, 0, -1;

    VectorXd b(3);

    b << 2, 3, 4;

    double d = a.dot(b);

    cout << "a.b = " << d << "\n\n";

    MatrixXd R = MatrixXd::Random(3, 3);

    cout << "R =\n" << R << "\n";
    cout << "R is " << R.rows() << "x" << R.cols() << "\n";
    
    return 0;
}
