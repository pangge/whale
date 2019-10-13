#ifndef WHALE_OPERATION_OPERATOR_H
#define WHALE_OPERATION_OPERATOR_H

#include <vector>
#include "core/types.h"
#include "core/check.h"

namespace whale {

struct OpParam {
    const char* name = "Unknown"; ///< parameter name
    std::string doc;              ///< parameter doc.

    bool is_commutative{true};    ///< true default.
                                  ///< Judge if the operation is commutative.
    unsigned char buffer[0];
};

class Op {
public:
    Op(const char* name);
    ~Op();

    template<typename T>
    Op& set(const char* name, const T& default_val) {
        OpParam* param_ptr = new OpParam();
        param_ptr->name = name;
        this->append(param_ptr, &default_val, sizeof(T));
    }

    template<typename T>
    T& get(const char* name){
    }

    void finalizer();

private:  

    void append(OpParam* param_ptr, const char* data, size_t bytes);

private:
    std::unordered_map<std::string, OpParam*> _args_map; 
    OpParam _param;
};

}

#endif
