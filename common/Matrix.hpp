#pragma once

#include <cstddef>
#include <stdexcept>

#if defined(_MSC_VER)
    #define RESTRICT __restrict
#else
    #define RESTRICT __restrict__
#endif

template<typename T>
class Matrix {

    size_t rows;
    size_t cols;
    T* data;

public:
    Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
        data = new T[rows * cols];
    }

    Matrix(Matrix&& other) noexcept : rows(other.rows), cols(other.cols), data(other.data) {
        other.data = nullptr;
        other.rows = 0;
        other.cols = 0;
    }

    ~Matrix() { delete[] data; }

    static void add(const Matrix& A, const Matrix& B, Matrix& C) {
        if (A.rows != B.rows || A.cols != B.cols)
            throw std::invalid_argument("Matrix sizes do not match");
        if (C.rows != A.rows || C.cols != A.cols)
            throw std::invalid_argument("Matrix sizes do not match");

        const T* RESTRICT pA = A.data;
        const T* RESTRICT pB = B.data;
        T* RESTRICT pC = C.data;
        const size_t total = A.rows * A.cols;

        for (size_t i = 0; i < total; i++)
            pC[i] = pA[i] + pB[i];
    }

    static void subtract(const Matrix& A, const Matrix& B, Matrix& C) {
        if (A.rows != B.rows || A.cols != B.cols)
            throw std::invalid_argument("Matrix sizes do not match");
        if (C.rows != A.rows || C.cols != A.cols)
            throw std::invalid_argument("Matrix sizes do not match");

        const T* RESTRICT pA = A.data;
        const T* RESTRICT pB = B.data;
        T* RESTRICT pC = C.data;
        const size_t total = A.rows * A.cols;

        for (size_t i = 0; i < total; i++)
            pC[i] = pA[i] - pB[i];
    }

    T& operator()(size_t row, size_t col) {
        return data[row * cols + col];
    }

    const T& operator()(size_t row, size_t col) const {
        return data[row * cols + col];
    }

    Matrix operator+(const Matrix& other) const {
        if (other.rows != rows || other.cols != cols)
            throw std::invalid_argument("Matrix dimensions do not match");
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows * cols; i++)
            result.data[i] = data[i] + other.data[i];
        return result;
    }
};
