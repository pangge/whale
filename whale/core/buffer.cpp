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

template<typename T>
void DeleterCPU(T* ptr) { 
    aligned_free((void*)ptr);
}

#ifdef WITH_CUDA
template<typename T>
void DeleterCUDA(T* ptr) { 
    cuda(Free(ptr));
}
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
        switch(_target.Type()) {
#ifdef BUILD_X86
            case Target::X86: {
                void* ptr = aligned_malloc(_real_bytes, 64);
                _ptr.reset((T*)ptr, DeleterCPU);
            } break;
#endif
#ifdef WITH_CUDA
            case Target::CUDA: {
                void* ptr=nullptr;
                cuda(Malloc(&ptr, _real_bytes);
                _ptr.reset((T*)ptr, DeleterCUDA);
            } break;
#endif
#ifdef WITH_ARM
            case Target::ARM: {
                void* ptr = aligned_malloc(_real_bytes, 64);
                _ptr.reset((T*)ptr, DeleterCPU);
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

template<typename T>
int mem_cpy(Buffer<T>& buf_dst, Buffer<T>& buf_src) {
    if(buf_dst.target() == buf_src.target()) {
        buf_dst.realloc(buf_src.size());
        switch(buf_dst.target()) {
#ifdef BUILD_X86
            case Target::X86: {
                memcpy(buf_dst.get(), buf_src.get(), buf_src.bytes());
            } break;
#endif
#ifdef WITH_CUDA
            case Target::CUDA: {
                // D2D
                cuda(Memcpy(buf_dst.get(), buf_src.get(), buf_src.bytes(), cudaMemcpyDeviceToDevice));
            } break;
#endif
#ifdef WITH_ARM
            case Target::ARM: {
                memcpy(buf_dst.get(), buf_src.get(), buf_src.bytes());
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
#if (defined BUILD_X86) and (defined WITH_CUDA)
            case Target::X86 && (buf_src.target() == Target::CUDA): {
                // D2H
                cuda(Memcpy(buf_dst.get(), buf_src.get(), buf_src.bytes(), cudaMemcpyDeviceToHost));
            } break;
            case Target::CUDA && (buf_src.target() == Target::X86k): {
                // H2D
                cuda(Memcpy(buf_dst.get(), buf_src.get(), buf_src.bytes(), cudaMemcpyHostToDevice));
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

template<>
int mem_cpy(Buffer<float>& buf_dst, Buffer<float>& buf_src);

// deep copy from buf_src
template<typename T>
Buffer<T> slice(const Buffer<T>& buf_src, int start, int len) { 
}

template<>
Buffer<float> slice(const Buffer<float>& buf_src, int start, int len);

}
