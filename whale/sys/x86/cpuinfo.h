#ifndef WHALE_CPUINFO_H
#define WHALE_CPUINFO_H

#include <unistd.h>
#include <vector>
#include <bitset>

namespace whale {
namespace sys {
namespace x86 {

class CPUInfo : public DeviceBase<CPUInfo> {
public:

private:
    std::bitset<32>;
    char cpu_num;
};

}
}
} 

#endif
