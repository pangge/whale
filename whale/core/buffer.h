#ifndef WHALE_BUFFER_H
#define WHALE_BUFFER_H

#include <unistd.h>
#include <vector>
#include "core/types.h"
#include "core/check.h"

namespace whale {

template<typename T>
struct WhalePointer {
    typedef void(*deleter) (T* ptr);
    WhalePointer(T* data):_data(data) {}

    ~WhalePointer() { _del(_data); }

    T* get() { return _data; }

    const T* get() const { return _data; }

    void reset(T* data, deleter del) {
        if(!_data) { _del(this->_data); }
        _data = data;
        _del = del;
    }
private:
    T* _data;
    deleter _del;
};

class MemBase {
public:
    MemBase() {}
    virtual ~MemBase() = 0;
};

template<typename T>
class Buffer : public MemBase {
public:
    Buffer(Target target, size_t len = 0);
    virtual ~Buffer() {}

    int realloc(size_t size);

    // swith to target
    void switch_to(Target target);

    // get size with type T
    inline size_t size() { return _bytes / sizeof(T); }
    inline size_t bytes() { return _bytes; }
    inline size_t rel_bytes() { return _real_bytes; }
    inline Target::type target() { return _target; }

    // get raw pointer
    inline T* get() { return _ptr.get(); }
	inline const T* get() const { return _ptr.get(); }

    Buffer(const Buffer& buffer) = delete;
    Buffer& operator=(const Buffer& buffer) = delete;

private:
    Target _target;
    size_t _bytes{0};
    WhalePointer<T> _ptr{nullptr};

private:
    size_t _real_bytes{0};
};

template<typename T>
int mem_cpy(Buffer<T>& buf_dst, Buffer<T>& buf_src);

// deep copy from buf_src
template<typename T>
Buffer<T> slice(const Buffer<T>& buf_src, int start, int len);

}

#endif
