#ifndef WHALE_OPERATION_OPERATOR_H
#define WHALE_OPERATION_OPERATOR_H

#include <vector>
#include "core/types.h"
#include "core/check.h"

namespace whale {

struct OpParam {
    const char* name = "Unknown"; ///< op name
    std::string doc;              ///< operator doc.

    bool is_commutative{true};    ///< true default.
                                  ///< Judge if the operation is commutative.
    ///< Operator paremeter map:  parameter name ---> arguments.
    ///std::unordered_map<std::string, Argument> Args_map;
    unsigned char buffer[0];
};

class Op {
public:
    Op(const char* name);
    ~Op();

    template<typename T>
    Op& set(const char* name, T default_val) {
    }

    template<typename T>
    T& get(const char* name){
    }

    void finalizer();

private:  

private:
    std::unordered_map<std::string, size_t> _args_map; 
    OpParam _param;
};

}

#endif
