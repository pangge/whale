#include <cstring>
#include "operation/gemm.h"

namespace whale {

template<typename T>
Gemm<T>::Gemm():Op("gemm") {
    set<bool>("transa", false, ""); 
    set<bool>("transb", false, ""); 
    set<int>("m", 0, ""); 
    set<int>("n", 0, ""); 
    set<int>("k", 0, ""); 
    set<T>("alpha", 0, ""); 
    set<T>("beta", 0, ""); 
    set<int>("lda", 0, ""); 
    set<int>("ldb", 0, ""); 
    set<int>("ldc", 0, "");
}

template<typename T>
void Gemm<T>::prepare() {
}

template<typename T>
int Gemm<T>::operator()(Cell<T, 4> A, Cell<T, 4> B, Cell<T, 4>C) {
    switch(A.target()) {
#ifdef BUILD_X86
        case Target::X86: { 
            for(int i=0; i<get<int>("m"); i++) {
                for(int j=0; j<get<int>("n"); j++) {
                    T sum = 0;
                    for(int index=0; index<get<int>("k")/64; index++) {
                        #pragma omp parallel num_threads(3)
                        #pragma omp for
                        for(int m =0; m < 64; m++) {
                            int idx = index*64 + m;
                            sum += A.data()[i*get<int>("k") + idx] * B.data()[idx*get<int>("n") + j];
                        }
                    }
                    C.data()[i*get<int>("n") + j] = sum;
                }
            }
        } break;
#endif
#ifdef WITH_CUDA 
        case Target::CUDA: { 
        } break;
#endif
#ifdef WITH_ARM
        case Target::ARM: { 
        } break;
#endif 
        default: { 
            fprintf(stderr, "ERROR: Target{%d} not support yet!", Target::type(A.target())); 
            exit(1);
        } 
    }
    return 1;
}

template class Gemm<float>;

}
