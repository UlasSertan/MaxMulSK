// 24.01.2016
#ifndef MATRIX_HPP
#define MATRIX_HPP
#include <array>
#if defined(_MSC_VER)
    #define RESTRICT __restrict
#else
    #define RESTRICT __restrict__
#endif

template<typename T>
class Matrix {

    size_t rows;
    size_t cols;
    T* data; // Just mention the types, not constraints

public:
    Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
        data = new T[rows * cols];
    }

    Matrix(Matrix&& other) noexcept : rows(other.rows), cols(other.cols), data(other.data) {
        other.data = nullptr;
        other.rows = 0;
        other.cols = 0;
    }

    ~Matrix() {delete[] data;}

    static void add(const Matrix& A, const Matrix& B, Matrix& C ) {
        if (A.rows != B.rows || A.cols != B.cols) {
            throw std::invalid_argument("Matrix sizes do not match");
        }
        if (C.rows != A.rows || C.cols != A.cols) {
            throw std::invalid_argument("Matrix sizes do not match");
        }

        // RESTRICT tells compiler that pointers do not overlap in memory
        const T* RESTRICT pA = A.data;
        const T* RESTRICT pB = B.data;
        T* RESTRICT pC = C.data;

        const size_t totalData = A.rows * A.cols;

        for (size_t i = 0; i < totalData; i++) {
            pC[i] = pA[i] + pB[i];
        }
    }

    static void subtract(const Matrix& A, const Matrix& B, Matrix& C ) {
        if (A.rows != B.rows || A.cols != B.cols) {
            throw std::invalid_argument("Matrix sizes do not match");
        }
        if (C.rows != A.rows || C.cols != A.cols) {
            throw std::invalid_argument("Matrix sizes do not match");
        }

        // RESTRICT tells compiler that pointers do not overlap in memory
        const T* RESTRICT pA = A.data;
        const T* RESTRICT pB = B.data;
        T* RESTRICT pC = C.data;

        const size_t totalData = A.rows * A.cols;

        for (size_t i = 0; i < totalData; i++) {
            pC[i] = pA[i] - pB[i];
        }
    }



    // Operator overloading that enables usage of m(i, j) directly

    // Read-Write
    T& operator()(size_t const row, size_t const col) {
        return data[row * cols + col];
    }

    // Read-only
    const T& operator()(size_t const row, size_t const col) const {
        return data[row * cols + col];
    }

    // Matrix addition
    Matrix operator+(const Matrix& matrixToAdd) const {
        if (matrixToAdd.rows != rows || matrixToAdd.cols != cols) {
            throw std::invalid_argument("Matrix dimensions do not match");
        }
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows * cols; i++) {
            result.data[i] = this->data[i] + matrixToAdd.data[i];
        }
        return result;
    }


};



#endif //MATRIX_HPP
