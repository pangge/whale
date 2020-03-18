#include <stdlib.h>
#include <x86intrin.h>

static inline bool before(uint64_t a, uint64_t b) { 
    return ((int64_t)b - (int64_t)a) > 0; 
} 

void pollDelay(uint32_t clocks) { 
    uint64_t endTime = _rdtsc()+ clocks; 
    for (; before(_rdtsc(), endTime); ) 
        _mm_pause();
}

int main(void) {
  return 0;
}

