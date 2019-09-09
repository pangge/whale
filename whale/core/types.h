#ifndef WHALE_TYPES_H
#define WHALE_TYPES_H

#include <unistd.h>
#include <assert.h>
#include <vector>
#include <chrono>
#include <random>

//#include "cudnn.h"
//#include "cublas_v2.h"

namespace whale {

struct Target {
    enum Type {
        X86,
        CUDA,
        ARM
    };

    Target() {};

    Target(Type type):_type(type) {}
    Target(const Target& target):_type(target._type) {}

    operator Type {
        return _type;
    }
    Type _type;
};

struct Layout {
    enum Type {
        NCHW, 
        NHWC,
	    NCHWCx
    };
    Layout(){};

    Layout(Type type):_type(type) {}
    Layout(const Layout& layout):_type(layout._type) {}

    operator Type() {
       return _type;
    }
#ifndef WITH_CUDA
    operator cudnnTensorFormat_t() {
        switch(_type) {
            case NCHW: return CUDNN_TENSOR_NCHW;
            case NHWC: return CUDNN_TENSOR_NHWC;
            default: return CUDNN_TENSOR_NCHW_VECT_C;
        }
    }
#endif
    Type _type;
};

#ifdef WITH_CUDA
template<typename T>
struct DataTraits {
    static constexpr cudnnDataType_t cudnn_type = CUDNN_DATA_FLOAT;
    static constexpr int size = sizeof(float);
};

template<>
struct DataTraits<double> {
    static constexpr cudnnDataType_t cudnn_type = CUDNN_DATA_DOUBLE;
    static constexpr int size = sizeof(double);
};

template<>
struct DataTraits<int> {
    static constexpr cudnnDataType_t cudnn_type = CUDNN_DATA_INT32;
    static constexpr int size = sizeof(int);
};

// int8 * 4
/*template<>
struct DataTraits<int32_t> {
    static constexpr cudnnDataType_t cudnn_type = CUDNN_DATA_INT8x4;
    static constexpr int size = sizeof(int32_t);
};*/

template<>
struct DataTraits<int16_t> {
    static constexpr cudnnDataType_t cudnn_type = CUDNN_DATA_HALF;
    static constexpr int size = sizeof(int16_t);
};

template<>
struct DataTraits<int8_t> {
    static constexpr cudnnDataType_t cudnn_type = CUDNN_DATA_INT8;
    static constexpr int size = sizeof(int8_t);
};
#endif

struct transform_t  {
    enum op_t {
        OP_N,       ///< the non-transpose operation is selected
        OP_T,       ///< the transpose operation is selected
        OP_C,       ///< the conjugate transpose operation is selected
    };

    op_t _type;

    transform_t():_type(OP_N) {} 

    transform_t(const op_t& type):_type(type) {}

    operator op_t() {
        return _type;
    }

#ifndef WITH_CUDA
    operator cublasOperation_t() {
        switch(_type) {
            case OP_N: return CUBLAS_OP_N;    
            case OP_T: return CUBLAS_OP_T;
            case OP_C: return CUBLAS_OP_C;  
            default: return CUBLAS_OP_N; // default CUBLAS_OP_N
        }
    }
#endif

};

}

#endif
