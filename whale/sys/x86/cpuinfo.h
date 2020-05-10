#ifndef WHALE_CPUINFO_H
#define WHALE_CPUINFO_H

#include <stdio.h>
#include <unistd.h>
#include <vector>
#include <algorithm>
#include <type_traits>
#include <sstream>
#include <unordered_map>

#include "sys/device.h"

namespace whale {
namespace sys {
namespace x86 {

struct CPUID {
    bool operator() (unsigned int leaf, 
                     Bits<32>& eax, 
                     Bits<32>& ebx, 
                     Bits<32>& ecx, 
                     Bits<32>& edx,
                     unsigned int sub_leaf=-1);

private:
    void check(unsigned int leaf, 
               Bits<32>& eax, 
               Bits<32>& ebx, 
               Bits<32>& ecx, 
               Bits<32>& edx);

    void check(unsigned int leaf, 
               unsigned int sub_leaf,
               Bits<32>& eax, 
               Bits<32>& ebx, 
               Bits<32>& ecx,
               Bits<32>& edx);
};

class X86Info: public DevInfo<X86Info> {
public:
    static constexpr const char* type = "X86 CPU";

    X86Info() noexcept;
    ~X86Info() = default;

    virtual std::ostream& operator<<(std::ostream& os) override {
        os << DebugStr();
        return os;
    }

    std::string DebugStr() {
        std::stringstream info;
        info << "Vendor: " << _vendor << "\r\n";
        info << "   \t" << _brand << "\r\n";
        for(auto& feature : _cpu_features) {
            switch(_cpu_features_val_type[feature]) {
                case Boolean: {
                    info << feature << ":" << (support(feature) ? "yes":"no") << "\r\n";
                } break;
                case Integer: {
                    info << feature << ":" << get_val(feature) << "\r\n";
                } break;
                default:break;
            }
        }
        return info.str();
    }

    bool support(const std::string& feature) {
        if(has(feature)) {
            return _cpu_features_support[feature];
        }
        return false;
    }

    int get_val(const std::string& feature) {
        if(has(feature)) {
            return _cpu_features_val[feature];
        }
        fprintf(stderr, "X86 CPUID query %s error!\n", feature.c_str());
        return -1;
    }

    std::string get_str(const std::string& feature) {
        if(has(feature)) {
            return _cpu_features_str[feature];
        }
        fprintf(stderr, "X86 CPUID query %s error!\n", feature.c_str());
        return "Not Known!";
    }

    bool has(const std::string& feature) {
        if ( std::find(_cpu_features.begin(), _cpu_features.end(), feature) 
                == _cpu_features.end() ) {
            return false;
        }
        return true;
    }


private:
    int extract_leaf(const std::string& leaf_name) {
        size_t pos = leaf_name.find_first_of('@');
        if(leaf_name[1] != 'x') {
            std::string num_str  = std::string(leaf_name.begin(), leaf_name.begin() + pos);
            return std::stoi(num_str);
        } else {
            std::string num_str  = std::string(leaf_name.begin()+2, leaf_name.begin() + pos);
            return std::stoi(num_str);
        }
    }

    template<typename T, 
        typename = typename std::enable_if<std::is_same<T, bool>::value | 
                                           std::is_same<T, int>::value | 
                                           std::is_same<T, std::string>::value>::type>
    void insert(const std::string& feature, T& val);

private:
    std::vector<std::string> _cpu_features;
    std::unordered_map<std::string, ValType> _cpu_features_val_type;
    std::unordered_map<std::string, bool> _cpu_features_support;
    std::unordered_map<std::string, int> _cpu_features_val;
    std::unordered_map<std::string, std::string> _cpu_features_str;
};

template<>
void X86Info::insert<bool>(const std::string& feature, bool& val) {
    if(has(feature)) {
        return;
    } else {
         _cpu_features.push_back(feature);
    }
    _cpu_features_support[feature] = val;
    _cpu_features_val_type[feature] = Boolean;
}

template<>
void X86Info::insert<int>(const std::string& feature, int& val) {
    if(has(feature)) {
        return;
    } else {
         _cpu_features.push_back(feature);
    }
    _cpu_features_val[feature] = val;
    _cpu_features_val_type[feature] = Integer;
}

template<>
void X86Info::insert<std::string>(const std::string& feature, std::string& val) {
    if(has(feature)) {
        return;
    } else {
         _cpu_features.push_back(feature);
    }
    _cpu_features_str[feature] = val;
    _cpu_features_val_type[feature] = String;
}

}
}
} 

#endif
