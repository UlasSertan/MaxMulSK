#include <iostream>
#include "Matrix.hpp"

int main() {
    std::cout << "Hello, World!" << std::endl;
    const Matrix<float> A(1, 1);

    const Matrix<float> B(1, 1);

    Matrix <float> C = A + B;
    return 0;
}