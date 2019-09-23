#ifndef WHALE_BUFFER_H
#define WHALE_BUFFER_H

#include <unistd.h>
#include <vector>
#include "core/types.h"
#include "core/check.h"

namespace whale {

template<typename T>
struct WhalePointer {
    typedef void (*deleter) (T*);

    WhalePointer(T* data, deleter del=nullptr):_data(data), _del(del) {}

    ~WhalePointer() { _del(_data); }

    T* get() { return _data; }
    void reset(T* data, deleter del=nullptr) {
        if(!del) { _del(this->_data); }
        _data = data;
        _del = del;
    }
private:
    T* _data;
    deleter _del;
};

template<typename T>
class Buffer {
public:
    Buffer(Target target, size_t len = 0);
    ~Buffer(){}

    int realloc(size_t size);

    // swith to target
    void switch_to(Target target);

    // get size with type T
    size_t size() { return _bytes / sizeof(T); }
    size_t bytes() { return _bytes; }
    Target target() { return _target; }

    // get raw pointer
    T* get() { return _ptr.get(); }
	const T* get() { return _ptr.get(); }

    Buffer(const Buffer& buffer) = delete;
    Buffer& operator=(const Buffer& buffer) = delete;

private:
    Target _target;
    size_t _bytes{0};
    WhalePointer<T> _ptr{nullptr};

private:
    size_t _real_bytes{0};
};

int mem_cpy(Buffer& buf_dst, Buffer& buf_src);

// deep copy from buf_src
Buffer slice(const Buffer& buf_src, int start, int len);

}

#endif
