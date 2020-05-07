#pragma once

#include <type_traits>

namespace whale {
namespace sys {

template<size_t N>
struct StdSize {
    static_assert(false, "Template parameter N is not standard size! ");
    const static bool value = false;
    typedef void type;
};

template<>
struct StdSize<8> {
    const static bool value = true;
    typedef unsigned char type;
};

template<>
struct StdSize<16> {
    const static bool value = true;
    typedef unsigned short type;
};

template<>
struct StdSize<32> {
    const static bool value = true;
    typedef unsigned int type;
};

template<>
struct StdSize<64> {
    const static bool value = true;
    typedef unsigned long type;
};


template<size_t N, std::enable_if_t<StdSize<N>::value> >
class BitBase {
    typedef StdSize<N>::type __WordT;
    __WordT __data{0};
    const unsigned char __char_onehot[8]={
        1<<0, // 0000 0001
        1<<1, // 0000 0010
        1<<2, // 0000 0100
        1<<3, // 0000 1000
        1<<4, // 0001 0000
        1<<5, // 0010 0000
        1<<6, // 0100 0000
        1<<7, // 1000 0000
    };

    size_t count_char(unsigned char ch) {
        size_t result = 0;
        for(int i=0; i<sizeof(unsigned char); i++) {
            if(__char_onehot[i] & ch) {
                result++;
            }
        }
        return result;
    }

    template<size_t> friend class Bits;
public:
    size_t count() const {
        size_t result = 0;
        const unsigned char* begin_p = (const unsigned char*)__data;
        const unsigned char* end_p = (const unsigned char*)__data + sizeof(__WordT);
        while(begin_p < end_p) {
            result += count_char[*begin_p];
            begin_p++;
        }
        return result;
    }

    size_t size() const { return sizeof(__WordT); }

    void set() { __data = ~(__data ^__data); }

    void set(size_t pos) {
    }
};

// class Bits only support standard base data type bit width
template<size_t N>
class Bits : public BitBase<N> {
public:

};

}
} 
