#ifndef WHALE_OPERATION_PARAMS_H
#define WHALE_OPERATION_PARAMS_H

#include <vector>
#include "core/types.h"
#include "core/check.h"

namespace whale {

struct OpParam {
    const char* name = "Unknown";

};

struct OperatorParam {
public:
    const char* name = "Unknown"; ///< op name
    std::string doc;              ///< operator doc.
    size_t num_in;                ///< io number of operator.
    size_t num_out;
    bool is_commutative{true};    ///< true default.
                                  ///< Judge if the operation is commutative.
    ///< Operator paremeter map:  parameter name ---> arguments.
    std::unordered_map<std::string, Argument> Args_map;
};

}

#endif
