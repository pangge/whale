#include "core/cell.h"

using namespace whale;

void test_cell_basic() {
    Cell<float,4> A;
    A.Alloc({1, 2, 3, 10});
    A.map_cpu(fill_value<float>, 42);
    printf("Creat A shape(%d, %d, %d, %d) value: \n", A.shape()[0], A.shape()[1], A.shape()[2], A.shape()[3]);
    for(int i=0; i < 8; i++) {
        printf("%f ", A.data()[i]);
    }
    printf("\n");
}

void test_cell_level1() {
    Cell<float,4> A;
    A.Alloc({1, 1, 1, 10});
    printf("A.bytes : %ldB real_bytes: %ldB\n", A.bytes(), A.rel_bytes());
    A.Alloc({1, 1, 1, 5});
    printf("A.bytes : %ldB real_bytes: %ldB\n", A.bytes(), A.rel_bytes());
    A.Alloc({1, 1, 1, 20});
    printf("A.bytes : %ldB real_bytes: %ldB\n", A.bytes(), A.rel_bytes());
}

void test_cell_level2() {
    Cell<float,4> B;
    B.Alloc({1, 1, 1, 5});
    B.map_cpu(fill_value<float>, 42);
    printf("Creat B shape(%d, %d, %d, %d) value: \n", B.shape()[0], B.shape()[1], B.shape()[2], B.shape()[3]);
    for(int i=0; i < 5; i++) {
        printf("%f ", B.data()[i]);
    }
    Cell<float,4> A;
    printf("\n Creat A shape(0)\n");
    printf("copy B to A...\n");
    A.CopyFrom(B);
    printf("Check A shape(%d, %d, %d, %d) value: \n", A.shape()[0], A.shape()[1], A.shape()[2], A.shape()[3]);
    for(int i=0; i < 5; i++) {
        printf("%f ", A.data()[i]);
    }
    printf("\n");
}


int main(int argc, const char** argv) {
    test_cell_basic();
    test_cell_level1();
    test_cell_level2();
    return 1;
}
