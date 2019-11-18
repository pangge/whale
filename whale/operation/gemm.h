#ifndef WHALE_GEMM_H
#define WHALE_GEMM_H

#include <vector>
#include "core/cell.h"
#include "operation/operator.h"

namespace whale {

template<typename T>
class Gemm: public Op {
public:
    Gemm();

    virtual void prepare() override;

    virtual int operator() (Cell<T, 4> A, Cell<T, 4> B, Cell<T, 4>C); 
};

}

#endif
