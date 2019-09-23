#include "core/buffer.h"

namespace whale {

// ref: https://stackoverflow.com/questions/38088732/explanation-to-aligned-malloc-implementation 
void* aligned_malloc(size_t required_bytes, size_t alignment) {
    void* p1; // original block
    void** p2; // aligned block
    int offset = alignment - 1 + sizeof(void*);
    if ((p1 = (void*)malloc(required_bytes + offset)) == NULL) {
       return NULL;
    }
    p2 = (void**)(((size_t)(p1) + offset) & ~(alignment - 1));
    p2[-1] = p1;
    return p2;
}

void aligned_free(void *p) {
    free(((void**)p)[-1]);
}


template<typename T, Target::type TargetType>
struct Deleter {
    void operator() (T* ptr){
        aligned_free((void*)ptr);
    }
};

#ifdef WITH_CUDA
template<typename T>
struct Deleter<T, Target::CUDA> {
    void operator() (T* ptr){
        // TODO
    }
};
#endif

template<typename T>
Buffer<T>::Buffer(Target target, size_t len) {
    _target=target;
    this->realloc(len);
}

template<typename T>
int Buffer<T>::realloc(size_t size) {
    _bytes = size * sizeof(T);
    if(_bytes > _real_bytes) {
        _real_bytes = _bytes;
        switch(_target) {
#ifdef BUILD_X86
            case Target::X86: {
                aligned_malloc(_real_bytes, 64) {
                _ptr.reset(new T(size), Deleter<T, Target::X86>);
            } break;
#endif
#ifdef WITH_CUDA
            case Target::CUDA: {
                // TODO
            } break;
#endif
#ifdef WITH_ARM
            case Target::ARM: {
                _ptr.reset(new T(size), Deleter<T, Target::ARM>);
            } break;
#endif
            default: {
                fprintf(stderr, "ERROR: Target{%d} not support yet!", Target::type(_target));
                exit(1);
            } 
        }
        return _bytes;
    }
    return 0;
}

template<typename T>
void Buffer<T>::switch_to(Target target) {
    if(_target != target) {
        _target=target;
        this->realloc(size());
    }
}

template class Buffer<float>;

int mem_cpy(Buffer& buf_dst, Buffer& buf_src) {
    if(buf_dst.target() == buf_src.target()) {
        buf_dst.realloc(buf_src.size());
        switch(buf_dst.target()) {
#ifdef BUILD_X86
            case Target::X86: {
            } break;
#endif
#ifdef WITH_CUDA
            case Target::CUDA: {
                // TODO
            } break;
#endif
#ifdef WITH_ARM
            case Target::ARM: {
            } break;
#endif
            default: {
                fprintf(stderr, "ERROR: Target{%d} to Target{%d} not support yet!", 
                        buf_dst.target(), buf_src.target());
                exit(1);
            } 
        }
    } else {
        switch(buf_dst.target()) {
#ifdef BUILD_X86
            case Target::X86: {
                if(buf_src.target() == Target::CUDA) {
                    // D2H
                } else {
                }
            } break;
#endif
#ifdef WITH_CUDA
            case Target::CUDA: {
                if(buf_src.target() == Target::X86k) {
                    // H2D
                } else {
                }
            } break;
#endif
            default: {
                fprintf(stderr, "ERROR: Target{%d} to Target{%d} not support yet!", 
                        buf_dst.target(), buf_src.target());
                exit(1);
            } 
        }
    }
}

// deep copy from buf_src
Buffer slice(const Buffer& buf_src, int start, int len) {
}

}
