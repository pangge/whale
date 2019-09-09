#ifndef CUBASE_CUDA_TIMER_H
#define CUBASE_CUDA_TIMER_H

#include "cubase_err.h"

enum DeviceT{
    GPU,
    CPU
};

enum Precision {
    ns = 0,
    us = 1,
    ms = 2,
    s = 3
};

template<int value, int count, bool lt_0>
struct Power;

template<int base, int count>
struct Power<base, count, false> {
    static const decltype(base) value = base * Power<base, count-1, count-1 < 0>::value;
};

template<int base, int count>
struct Power<base, count, true> {
    static const decltype(base) value = (1.0 / base) * Power<base, count+1, count+1 < 0>::value;
};


template<int base>
struct Power<base, 0, false> {
    static const int value = 1;
};

template<int base>
struct Power<base, 0, true> {
    static const int value = 1;
};

template<Precision dst, Precision src>
struct Precision_step {
    static const int value = Power<1000, src-dst, (src > dst)>::value;
};

template<Precision dst, Precision src>
struct time_cast {
    time_cast(float src_time) {
        dst_time = Precision_step<dst, src>::value * src_time;
    }

    operator float() {
        return dst_time;
    }

    float dst_time;
};

template<Precision dst>
struct ms_cast : time_cast<dst, ms> {
    ms_cast(float src_time): time_cast<dst, ms>(src_time) {}
};

template<DeviceT dev, Precision pres = ms>
struct Timer {
    Timer();
    ~Timer();

    void start();

    void end();

    float elapsed_time();
};

template<Precision pres>
struct Timer<GPU, pres> {
    cudaEvent_t _start, _stop;

    Timer() {
        cuda(EventCreate(&_start)); 
        cuda(EventCreate(&_stop));
    }
    ~Timer() {
        cuda(EventDestroy(_start));
        cuda(EventDestroy(_stop));
    }

    void start() {
        cuda(EventRecord(_start, NULL));
    }

    void end() {
        cuda(EventRecord(_stop, NULL)); 
        cuda(EventSynchronize(_stop));
    }

    float elapsed_time() {
        float elapsedTimeMs;
        cuda(EventElapsedTime(&elapsedTimeMs, _start, _stop));
        return ms_cast<pres>(elapsedTimeMs);
    }
};

#endif
