#pragma once

#include "sys/bits.h"

#include <unistd.h>
#include <vector>
#include <ostream>

namespace whale {
namespace sys {

template<class DevT>
class DevInfo {
public:
    enum ValType {
        Boolean,
        Integer,
        String
    };

    DevInfo() noexcept {}
    ~DevInfo() = default;

    static const char* type() { return DevT::type; }

    // create global instance, note that it is not thread safe
    static DevT make() {
        static DevT ins;
        return ins;
    }

    std::string _vendor;
    std::string _brand;
};

}
} 

