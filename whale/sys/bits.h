#pragma once

#include <stdio.h>
#include <type_traits>

namespace whale {
namespace sys {

template<std::size_t N>
struct StdSize {
    //static_assert(false, "Template parameter N is not standard size! ");
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

template<std::size_t N, typename = typename std::enable_if<StdSize<N>::value>::type >
class BitBase {
    typedef typename StdSize<N>::type __WordT;
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

    std::size_t count_char(unsigned char ch) const {
        std::size_t result = 0;
        for(int i=0; i<8; i++) {
            if(__char_onehot[i] & ch) {
                result++;
            }
        }
        return result;
    }

    template<std::size_t> friend class Bits;
public:
    std::size_t count() const {
        std::size_t result = 0;
        __WordT base = 0x1;
        for(int i =0; i < N; i++) {
            if(__data & (base << i)) {
                result += 1;
            }
        }
        return result;
    }

    std::size_t size() const { return N; }

    void set() { __data = ~(__data ^__data); }

    void set(std::size_t pos, bool positive = true) {
        // assert 0 <= pos < size()
        if(positive) {
            __data = __data | (__WordT)0x1 << pos;
        } else {
            if((*this)[pos]) {
                __data = __data - ((__WordT)0x1 << pos);
            }
        }
    }

    bool operator[](std::size_t pos) {
        __WordT base = 0x1;
        base <<= pos;
        if(base & __data) {
            return true;
        }
        return false;
    }

    void reset() { __data = (__WordT)0; }

    void flip() { __data = ~__data; }
};

// class Bits only support standard base data type bit width
template<std::size_t N>
class Bits : public BitBase<N> {
public:
    Bits() noexcept { this->reset(); }
    Bits(unsigned long val) noexcept { this->__data = val; }
    ~Bits() = default;

    Bits(const Bits<N>& other) {
        this->__data = other.__data;
    }

    // get slice bits's value
    typename BitBase<N>::__WordT get_val(int start, int end) {
        Bits<N> result;
        for(int i = start; i<=end; i++) {
            result.set(i);
        }
        result &= (*this);
        result >>= start;
        return result.to_data();
    }

    // get slice bits's ascii string

    std::string get_str(int start, int end) {
        typename BitBase<N>::__WordT result = get_val(start, end);
        char result_str[N/8];
        *reinterpret_cast<int*>(result_str) = result;
        return std::string(result_str);
    }

    Bits<N>& operator=(const Bits<N>& other) {
        this->__data = other.__data;
        return *this;
    }

    // get data of bits
    typename BitBase<N>::__WordT& to_data() { return this->__data; }

    Bits<N>& operator&=(const Bits<N>& other ) noexcept {
        this->__data = this->__data & other.__data;
        return *this;
    }

    Bits<N>& operator|=(const Bits<N>& other ) noexcept {
        this->__data = this->__data | other.__data;
        return *this;
    }
    Bits<N>& operator^=(const Bits<N>& other ) noexcept {
        this->__data = this->__data ^ other.__data;
        return *this;
    }
    Bits<N> operator~() const noexcept {
        Bits<N> result;
        result.flip();
        return result;
    }

    Bits<N> operator<<(std::size_t pos ) const noexcept {
        Bits<N> result(*this);
        result <<= pos;
        return result;
    }
    Bits<N>& operator<<=(std::size_t pos ) noexcept {
        this->__data = (this->__data << pos);
        return *this;
    }
    Bits<N> operator>>(std::size_t pos ) const noexcept {
        Bits<N> result(*this);
        result >>= pos;
        return result;
    }
    Bits<N>& operator>>=(std::size_t pos ) noexcept {
        this->__data = (this->__data >> pos);
        return *this;
    }
};

}
} 
