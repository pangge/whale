#include <thread>
#include <iostream>
#include "operation/gemm.h"

using namespace whale;

void test_gemm() {
    printf("Create new threads(%ld): \n", std::this_thread::get_id());
    int m = 100;
    int n = 100;
    int k = 10240;
    Cell<float,4> A;
    A.Alloc({1, 1, m, k});
    A.map_cpu(fill_value<float>, 1);
    printf("Creat A shape(%d, %d, %d, %d)  \n", A.shape()[0], A.shape()[1], A.shape()[2], A.shape()[3]);


    Cell<float,4> B;
    B.Alloc({1, 1, k, n});
    B.map_cpu(fill_value<float>, 1);
    printf("Creat B shape(%d, %d, %d, %d)  \n", B.shape()[0], B.shape()[1], B.shape()[2], B.shape()[3]);

    Cell<float,4> C;
    C.Alloc({1, 1, m, n});

    printf("Creat C shape(%d, %d, %d, %d)  \n", C.shape()[0], C.shape()[1], C.shape()[2], C.shape()[3]);

    // create gemm operator
    Gemm<float> gemm;
    auto transa = gemm.get<bool>("transa");
    auto alpha = gemm.get<float>("alpha");
    gemm.get<int>("m") = m;
    gemm.get<int>("n") = n;
    gemm.get<int>("k") = k;
    std::cout<<"m, n, k = " << m <<", "<<n<<", "<<k<<"\n";

    gemm(A, B, C);

    for(int i=0; i < 5; i++) {
        printf("%f ", C.data()[i]);
    }
    printf("\n");
}

void test_multi_thread() {
    printf("multi_thread test start");
    int worker_num = 1;
    std::vector<std::thread> workders;
    for(int i =0; i<worker_num; i++) {
        workders.emplace_back(std::thread(test_gemm));
    }
    for(int i=0; i<worker_num; i++) {
        workders[i].join();
    }
    printf("multi_thread test end!\n");
}

int main(int argc, const char** argv) {
    test_multi_thread();
    return 1;
}
