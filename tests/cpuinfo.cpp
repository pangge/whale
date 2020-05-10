#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "sys/x86/cpuinfo.h"

using namespace whale::sys;

void bits_test() {
    // assert
    Bits<8> a32;
    // base
    Bits<32> b32;
    b32 = 0x01;
    printf("b32 = 0x2 : %x\n", b32.to_data());
    b32 <<= 1;
    b32 <<= 1;
    b32 >>= 1;
    printf("b32 = 0x2 : %x\n", b32.to_data()); 
    // count
    Bits<32> c32;
    c32 = 0x03030303;
    assert(c32.count() == 8);
    c32.set(2, true);
    assert(c32.count() == 9);
    // test operator[]
    assert(c32[0] == true);
    assert(c32[1] == true);
    assert(c32[2] == true);
    assert(c32[3] == false);
    assert(c32.count() == 9);
    c32.set(2, false);
    assert(c32.count() == 8);
    assert(c32[2] == false);
    // more
    Bits<16> dest;
    std::string pattern_str = "1001";
    Bits<16> pattern(0x9); // binary 1001 = 0x9
    for (size_t i = 0, ie = dest.size() / pattern_str.size(); i != ie; ++i) {
        dest <<= pattern_str.size();
        dest |= pattern;
    }
    assert(dest.to_data() == 39321);
    return 0;
}

int main(void) {
    x86::X86Info info;
    std::cout<< info;
}
