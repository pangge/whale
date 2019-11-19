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
    enum type {
        X86,
        CUDA,
        ARM
    };
#ifdef WITH_ARM
    const static type Default = Target::ARM;
#else
    const static type Default = Target::X86;
#endif
    Target() {}

    Target(type t):_type(t) {}
    Target(const Target& target):_type(target._type) {}

    operator type() { return _type; };
    type Type() { return _type; }
    type _type;
};

struct Layout {
    enum type {
        Unknown = -1,
        W = 0,
        HW,
        WH,
        CHW,
        HWC,
        NCHW, 
        NHWC,
	    NCHWCx
    };
    Layout(){}

    Layout(type t):_type(t) {}
    Layout(const Layout& layout):_type(layout._type) {}

    bool operator==(Layout& layout) {
        return this->_type == layout;
    }

    Layout& operator=(const Layout& layout) {
        this->_type = layout._type;
        return *this;
    }

    operator type() { return _type; }
#ifdef WITH_CUDA
    operator cudnnTensorFormat_t() {
        switch(_type) {
            case NCHW: return CUDNN_TENSOR_NCHW;
            case NHWC: return CUDNN_TENSOR_NHWC;
            default: return CUDNN_TENSOR_NCHW_VECT_C;
        }
    }
#endif
    type _type;
};

template<int Dim>
struct LayoutTraits {
    static constexpr Layout::type Default = Layout::Unknown;
};

template<>
struct LayoutTraits<4> {
    static constexpr Layout::type Default = Layout::NCHW;
};

template<>
struct LayoutTraits<3> {
    static constexpr Layout::type Default = Layout::CHW;
};

template<>
struct LayoutTraits<2> {
    static constexpr Layout::type Default = Layout::HW;
};

template<>
struct LayoutTraits<1> {
    static constexpr Layout::type Default = Layout::W;
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

#ifdef WITH_CUDA
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
