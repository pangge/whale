#include <cstring>
#include "operation/operator.h"

namespace whale {

Op::Op(const char* name) {
    name = name;
}

Op::~Op() {
    for(auto it = args.begin(); it!=args.end();it++) {
        free(it->second);
    }
}

void Op::append(OpParam* op_param, 
            size_t bytes,
            const char* doc) {
    if(!args.count(op_param->name)) {
        args[op_param->name] = op_param;
    }
}

}
