#include "core/cell.h"

using namespace whale;

int main(int argc, const char** argv) {
    Cell<float,4> A;
    std::vector<int> test{1, 2, 3, 10};
    Shape<4> shape(test);
    A.Alloc(shape);
    A.map_cpu(random_gen<float>, 0, 1);
    printf("A.shape(%d, %d, %d, %d)\n", A.shape()[0], A.shape()[1], A.shape()[2], A.shape()[3]);
    for(int i=0; i < A.size(); i++) {
        printf("%f ", A.data()[i]);
    }
    return 1;
}
