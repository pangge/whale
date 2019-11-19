#include "operation/gemm.h"

using namespace whale;

void test_gemm() {
    int m = 100;
    int n = 100;
    int k = 1024;
    Cell<float,4> A;
    A.Alloc({1, 1, m, k});
    A.map_cpu(fill_value<float>, 1);
    printf("Creat A shape(%d, %d, %d, %d) value: \n", A.shape()[0], A.shape()[1], A.shape()[2], A.shape()[3]);


    Cell<float,4> B;
    B.Alloc({1, 1, k, n});
    B.map_cpu(fill_value<float>, 1);
    printf("Creat B shape(%d, %d, %d, %d) value: \n", B.shape()[0], B.shape()[1], B.shape()[2], B.shape()[3]);

    Cell<float,4> C;
    C.Alloc({1, 1, m, n});

    printf("Creat C shape(%d, %d, %d, %d) value: \n", C.shape()[0], C.shape()[1], C.shape()[2], C.shape()[3]);

    // create gemm operator
    Gemm<float> gemm;
    auto transa = gemm.get<bool>("transa");
    auto alpha = gemm.get<float>("alpha");
    gemm.get<int>("m") = m;
    gemm.get<int>("n") = n;
    gemm.get<int>("k") = k;

    for(int i=0; i<1; i++) {
        gemm(A, B, C);
    }
    for(int i=0; i < 5; i++) {
        printf("%f ", C.data()[i]);
    }
    printf("\n");
}

int main(int argc, const char** argv) {
    test_gemm();
    return 1;
}
