//
// Created by Ulaş Sertan KEMEÇ on 24.01.2026.
//

#ifndef GEMMKERNELS_HPP
#define GEMMKERNELS_HPP



namespace  GEMMKernels {

    // A: MxK , B: KxN , C: MxN
    void multiply(const float* A, const float* B, float* C,
        size_t N, size_t M, size_t K);

};



#endif //GEMMKERNELS_HPP
