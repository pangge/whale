#ifndef WHALE_OPERATION_OPERATOR_H
#define WHALE_OPERATION_OPERATOR_H

#include <vector>
#include <string>
#include <unordered_map>
#include "core/types.h"
#include "core/check.h"

namespace whale {

struct OpParam {
    const char* name = "Unknown"; ///< parameter name
    const char* doc = "None";     ///< parameter doc.

    size_t bytes{0};              ///< parameter bytes.
                                  ///< Judge if the operation is commutative.
    char buffer[0];
};

class Op {
public:
    Op(const char* name);
    ~Op();

    template<typename T>
    Op& set(const char* name, const T& default_val, const char* doc) {
        OpParam* op_tmp = (struct OpParam*) malloc (sizeof(OpParam) + sizeof(T));
        op_tmp->name = name;
        op_tmp->bytes = sizeof(T);
        op_tmp->doc = doc;
        *((T*)(op_tmp->buffer)) = default_val;
        this->append(op_tmp, sizeof(T), doc);
    }

    template<typename T>
    T& get(const char* name){
        if(args.count(name)) {
            return *((T*)(args[name]->buffer));
        } else {
            fprintf(stderr, "ERROR: target arg name{%s} not found!", name);
            exit(1);
            T ret;
            return ret;
        }
    }

    std::string type() { return name; }

    virtual void prepare() = 0;

    void finalizer(){}

private:  

    void append(OpParam* op_param, 
                size_t bytes, 
                const char* doc);

protected:
    std::string name;
    std::unordered_map<std::string, OpParam*> args; 
};

}

#endif
