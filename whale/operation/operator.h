#ifndef WHALE_OPERATION_OPERATOR_H
#define WHALE_OPERATION_OPERATOR_H

#include <vector>
#include <string>
#include <unordered_map>
#include "core/types.h"
#include "core/check.h"
#include "core/buffer.h"

namespace whale {

struct OpParam {
    const char* name = "Unknown"; ///< parameter name
    const char* doc = "None";     ///< parameter doc.

    size_t size{0};              ///< parameter element number.
                                  ///< Judge if the operation is commutative.
    MemBase* buf{nullptr};
};

class Op {
public:
    Op(const char* name);
    ~Op();

    template<typename T>
    Op& set(const char* name, size_t size, const char* doc) {
        OpParam* op_tmp = (struct OpParam*) malloc (sizeof(OpParam));
        op_tmp->name = name;
        op_tmp->size= size;
        op_tmp->doc = doc;
        op_tmp->buf = new Buffer<T>(Target::Default, size);
        this->append(op_tmp);
        return *this;
    }
size
    template<typename T>
    Buffer<T>* get_buf(const char* name){
        if(args.count(name)) {
            return static_cast<Buffer<T>*>(args[name]->buf);
        } else {
            fprintf(stderr, "ERROR: target arg name{%s} not found!", name);
            exit(1);
            return nullptr;
        }
    }

    template<typename T>
    T& get_val(const char* name) {
        if(T* ret = get_val_ptr(name)) {
            return ret[0];
        }
        return T();
    }

    template<typename T>
    T* get_val_ptr(const char* name) {
        if(Buffer<T>* buf_p = get_buf(name)) {
            return buf_p.get();
        }
        return nullptr;
    }


    std::string type() { return name; }

    virtual void prepare() = 0;

    void finalizer() {}

private:  

    void append(OpParam* op_param);

protected:
    std::string name;
    std::unordered_map<std::string, OpParam*> args; 
};

}

#endif
