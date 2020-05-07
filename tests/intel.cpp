#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>

#include <ctime>
#include <iostream>

class timer {
    struct date {
        int year;
        int mon;
        int day;
        bool isLeap(int y) { return (y%4 ==0 && y%100 != 0) || y%400 == 0; }
        int daysOfMonth(int m) {
            int days[12]={31,28,31,30,31,30,31,31,30,31,30,31};
			if(m!=2) {
                return days[m-1];
            } else {
                return 28+isLeap(year);
            }
        }
        int TotalDays() {
            int days=day;
            for(int y=1; y<year; y++) {
                days += 365 + isLeap(y);
            }
            for(int m=1; m<mon; m++) {
                days += daysOfMonth(m);
            }
            return days;
        }
        int operator-(date& date_other) {
            return this->TotalDays() - date_other.TotalDays();
        }
    };
public:
    explicit timer(int year, int mon, int day, int valid_day):
        _valid_day(valid_day), _start_date({year, mon, day}) {}
    ~timer() {}

    operator bool() {
        std::cout<<_start_date.year<<":"<<_start_date.mon<<":"<<_start_date.day<<"\n";
        date now_d = now();
        int delta = now_d - _start_date;
        std::cout<<"\\___delta: "<<delta<<"\n";
        if(delta  > _valid_day) {
            return true;
        }
        return false; 
    }

private:
	date now() {
        date ret;
        std::time_t t = std::time(0);   // get time now
        std::tm* now = std::localtime(&t);
        ret.year = now->tm_year + 1900;    
        ret.mon = now->tm_mon + 1;
        ret.day = now->tm_mday;
        return ret;
    }
private:
    date _start_date;
    int _valid_day{0};
};

// more inline asm ref: https://gcc.gnu.org/onlinedocs/gcc/Extended-Asm.html

static inline bool before(uint64_t a, uint64_t b) { 
    return ((int64_t)b - (int64_t)a) > 0; 
} 

void pollDelay(uint32_t clocks) { 
    uint64_t endTime = _rdtsc()+ clocks; 
    for (; before(_rdtsc(), endTime); ) 
        _mm_pause();
}

void DoCheck(uint32_t dwSomeValue)
{
   uint32_t dwRes;

   // Assumes dwSomeValue is not zero.
   asm ("bsfl %1,%0"
     : "=r" (dwRes)
     : "r" (dwSomeValue)
     : "cc");
   printf("%d\n", dwRes);
   //assert(dwRes > 3);
}


#include <iostream>
#include <array>
 
struct S {
         const unsigned char a[4] = {0x01, 0x03, 0x04, 0x10};
};

int main(void) {
	uint32_t dwSomeValue = 2;
	DoCheck(dwSomeValue);
    
    printf("hello intel %d \n", sizeof(unsigned long long));
    //timer ti(2020, 4, 0, 7);
    /*while(1) {
        if(ti) {
            printf("continue\n");
        }
    }*/
    S ll;
    for(int i=0; i<4; i++) {
        printf("a[%d]: %d\n",i,ll.a[i]);
    }
    return 0;
}

