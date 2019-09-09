#include <whale/core/types.h>

namespace whale {

template<typename T>
struct Deleter {
    void operator()(T* ptr) const {
#ifdef WITH_CUDA
#else
        delete ptr;
        ptr=nullptr;
#endif
    }
};

template<typename T>
Buffer<T>::Buffer(Target target, size_t len) {
    _target=target;
    this->realloc(len);
}

template<typename T>
int Buffer<T>::realloc(size_t size) {
    _bytes = size * sizeof(T);
    if(_bytes > _real_bytes) {
        _ptr->reset(nullptr);
        _real_bytes = _bytes;
        if(_target == Target::X86) {
            _ptr = _ptr.reset(new T(size), Deleter);
        } 
#ifdef WITH_CUDA
        else if(_target == Target::CUDA) {
            // TODO
        } 
#endif
        else if(_target == Target::ARM) {
            _ptr = _ptr.reset(new T(size), Deleter);
        } else {
            fprintf(stderr, "ERROR: Target{%d} not support yet!", Target::_type);
            exit(1);
        }
        return _bytes;
    }
    return 0;
}

template<typename T>
void Buffer<T>::swith_to(Target target) {
    if(_target != target) {
        _target=target;
        this->realloc(size());
    }
}

template class Buffer<float>;

}
