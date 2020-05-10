#include "sys/x86/cpuinfo.h"

namespace whale {
namespace sys {
namespace x86 {

struct CPUFeatureQuery {
    typedef std::function<bool(void)>  query_bits_t;
    typedef std::function<int(void)> query_val_t;
    typedef std::function<std::string(void)> query_str_t;
    typedef std::unordered_map<std::string, query_bits_t>  query_bits_map;
    typedef std::unordered_map<std::string, query_val_t> query_val_map;
    typedef std::unordered_map<std::string, query_str_t> query_str_map;
    typedef std::function< bool(void) > query_leaf_t;

    struct Pack {
        query_leaf_t query_leaf_func;
        query_bits_map query_bits_func;
        query_val_map query_val_func;
        query_str_map query_str_func;

        Pack& register_query_bits(std::string key_of_leaf, query_bits_t func) {
            if(query_bits_func.count(key_of_leaf)) {
                query_bits_func[key_of_leaf] = func;
            }
            return *this;
        }

        Pack& register_query_val(std::string key_of_leaf, query_val_t func) {
            if(query_val_func.count(key_of_leaf)) {
                query_val_func[key_of_leaf] = func;
            }
            return *this;
        }

        Pack& register_query_str(std::string key_of_leaf, query_str_t func) {
            if(query_str_func.count(key_of_leaf)) {
                query_str_func[key_of_leaf] = func;
            }
            return *this;
        }
    };


    static CPUFeatureQuery& Global() {
        static CPUFeatureQuery ins;
        return ins;
    }

    // reset all reg's bit contents to zero
    void reset() {
        EAX.reset();
        EBX.reset();
        ECX.reset();
        EDX.reset();
    }

    Pack& register_query_leaf(std::string key_of_q_leaf, query_leaf_t func) {
        Pack pack;
        pack.query_leaf_func = func;
        CPUFeatureQuery::Global().cpu_features[key_of_q_leaf] = pack;
        return CPUFeatureQuery::Global().cpu_features[key_of_q_leaf];
    }

    std::unordered_map<std::string, Pack> cpu_features;
    Bits<32> EAX;
    Bits<32> EBX;
    Bits<32> ECX;
    Bits<32> EDX;
    CPUID CMD;
};

// those register only use after DECLARE_FEATURE_LEAF and DECLARE_FEATURE_LEAF_WITH_SUB
#define REG_FEATURE(REG, FEATURE_NAME, ID)\
    register_query_bits(#FEATURE_NAME, []() -> bool {\
                return CPUFeatureQuery::Global().REG[ID];\
            })

#define REG_FEATURE_VAL(REG, FEATURE_NAME, START, END)\
    register_query_val(#FEATURE_NAME, []() -> int {\
                return CPUFeatureQuery::Global().REG.get_val(START, END);\
            })

#define REG_FEATURE_STR(REG, FEATURE_NAME, START, END)\
    register_query_str(#FEATURE_NAME, []() -> std::string {\
                return CPUFeatureQuery::Global().REG.get_str(START, END);\
            })

#define CONCAT(X, Y) X##Y
#define NAME_CONCAT(X, Y) CONCAT(X, Y)
#define NAME_SOLE(X) NAME_CONCAT(X, __COUNTER__)

#define DECLARE_FEATURE_LEAF(LEAF) \
    static CPUFeatureQuery::Pack& NAME_SOLE(LEAF_PACK) = \
        CPUFeatureQuery::Global().register_query_leaf(#LEAF"@NULL", \
            []() -> bool {\
                CPUFeatureQuery::Global().reset();\
                return CPUFeatureQuery::Global().CMD(LEAF,\
                                              CPUFeatureQuery::Global().EAX,\
                                              CPUFeatureQuery::Global().EBX,\
                                              CPUFeatureQuery::Global().ECX,\
                                              CPUFeatureQuery::Global().EDX,\
                                              -1);\
            }\
        )

#define DECLARE_FEATURE_LEAF_WITH_SUB(LEAF, SUB_LEAF) \
    static CPUFeatureQuery::Pack& NAME_SOLE(SUB_LEAF_PACK) = \
        CPUFeatureQuery::Global().register_query_leaf(#LEAF"@"#SUB_LEAF, \
            []() -> bool {\
                return CPUFeatureQuery::Global().CMD(LEAF,\
                                              CPUFeatureQuery::Global().EAX,\
                                              CPUFeatureQuery::Global().EBX,\
                                              CPUFeatureQuery::Global().ECX,\
                                              CPUFeatureQuery::Global().EDX,\
                                              SUB_LEAF);\
            }\
        )

// Features for leaf 0
DECLARE_FEATURE_LEAF(0)
    .REG_FEATURE_VAL(EAX, highest_valid_function_id, 0, 31)
    .REG_FEATURE_STR(EBX, vendor_b, 0, 31)
    .REG_FEATURE_STR(ECX, vendor_c, 0, 31)
    .REG_FEATURE_STR(EDX, vendor_d, 0, 31);

// Features for leaf 1
DECLARE_FEATURE_LEAF(1)
    .REG_FEATURE_VAL(EAX, ExtendedFamilyID, 20, 27)
    .REG_FEATURE_VAL(EAX, ExtendedModelID, 16, 19)
    .REG_FEATURE_VAL(EAX, ProcessorType, 12, 13)
    .REG_FEATURE_VAL(EAX, FamilyID, 8, 11)
    .REG_FEATURE_VAL(EAX, Model, 4, 7)
    .REG_FEATURE_VAL(EAX, SteppingID, 0, 3)
    .REG_FEATURE_VAL(EBX, BrandIndex, 0, 7)
    // not valid unless CPUID.01.EDX.CLFSH [bit 19]= 1 (if CLFLUSH feature flag is set.)
    .REG_FEATURE_VAL(EBX, CLFLUSH_LineSize, 8, 15) 
    // not valid unless CPUID.01.EDX.HTT [bit 28]=1 (if Hyper-threading feature flag is set.)
    .REG_FEATURE_VAL(EBX, MaximumNumberOfAddressableIDsForLogicalProcessors, 16, 23) 
    .REG_FEATURE_VAL(EBX, LocalAPIC_ID, 24, 31) // valid only if Pentium 4 and subsequent processors.
    .REG_FEATURE(ECX, sse3, 0)
    .REG_FEATURE(ECX, pclmulqdq, 1) // https://en.wikipedia.org/wiki/CLMUL_instruction_set
    .REG_FEATURE(ECX, dtes64, 2)
    .REG_FEATURE(ECX, monitor, 3)
    .REG_FEATURE(ECX, ds-cpl, 4)
    .REG_FEATURE(ECX, vmx, 5)
    .REG_FEATURE(ECX, smx, 6)
    .REG_FEATURE(ECX, est, 7)
    .REG_FEATURE(ECX, tm2, 8)
    .REG_FEATURE(ECX, ssse3, 9)
    .REG_FEATURE(ECX, cnxt-id, 10)
    .REG_FEATURE(ECX, sdbg, 11)
    .REG_FEATURE(ECX, fma, 12)
    .REG_FEATURE(ECX, cx16, 13)
    .REG_FEATURE(ECX, xtpr, 14)
    .REG_FEATURE(ECX, pdcm, 15)
    //.REG_FEATURE(ECX, , 16) // (reserved)
    .REG_FEATURE(ECX, pcid, 17)
    .REG_FEATURE(ECX, dca, 18)
    .REG_FEATURE(ECX, sse4.1, 19)
    .REG_FEATURE(ECX, sse4.2, 20)
    .REG_FEATURE(ECX, x2apic, 21)
    .REG_FEATURE(ECX, movbe, 22)
    .REG_FEATURE(ECX, popcnt, 23)
    .REG_FEATURE(ECX, tsc-deadline, 24)
    .REG_FEATURE(ECX, aes, 25)
    .REG_FEATURE(ECX, xsave, 26)
    .REG_FEATURE(ECX, osxsave, 27)
    .REG_FEATURE(ECX, avx, 28)
    .REG_FEATURE(ECX, f16c, 29)
    .REG_FEATURE(ECX, rdrnd, 30)
    .REG_FEATURE(ECX, hypervisor, 31)
    .REG_FEATURE(EDX, fpu, 0)
    .REG_FEATURE(EDX, vme, 1)
    .REG_FEATURE(EDX, de, 2)
    .REG_FEATURE(EDX, pse, 3)
    .REG_FEATURE(EDX, tsc, 4)
    .REG_FEATURE(EDX, msr, 5)
    .REG_FEATURE(EDX, pae, 6)
    .REG_FEATURE(EDX, mce, 7)
    .REG_FEATURE(EDX, cx8, 8)
    .REG_FEATURE(EDX, apic, 9)
    //.REG_FEATURE(EDX, , 10)
    .REG_FEATURE(EDX, sep, 11)
    .REG_FEATURE(EDX, mtrr, 12)
    .REG_FEATURE(EDX, pge, 13)
    .REG_FEATURE(EDX, mca, 14)
    .REG_FEATURE(EDX, cmov, 15)
    .REG_FEATURE(EDX, pat, 16)
    .REG_FEATURE(EDX, pse-36, 17)
    .REG_FEATURE(EDX, psn, 18)
    .REG_FEATURE(EDX, clfsh, 19)
    //.REG_FEATURE(EDX, , 20)
    .REG_FEATURE(EDX, ds, 21)
    .REG_FEATURE(EDX, acpi, 22)
    .REG_FEATURE(EDX, mmx, 23)
    .REG_FEATURE(EDX, fxsr, 24)
    .REG_FEATURE(EDX, sse, 25)
    .REG_FEATURE(EDX, sse2, 26)
    .REG_FEATURE(EDX, ss, 27)
    .REG_FEATURE(EDX, htt, 28)
    .REG_FEATURE(EDX, tm, 29)
    .REG_FEATURE(EDX, ia64, 30)
    .REG_FEATURE(EDX, pbe, 31);

// Features for leaf 7 sub-leaf 0
DECLARE_FEATURE_LEAF_WITH_SUB(7, 0)
    .REG_FEATURE(EBX, fsgsbase, 0)
    //.REG_FEATURE(EBX, , 1)
    .REG_FEATURE(EBX, sgx, 2)
    .REG_FEATURE(EBX, bmi1, 3)
    .REG_FEATURE(EBX, hle, 4)
    .REG_FEATURE(EBX, avx2, 5)
    //.REG_FEATURE(EBX, , 6)
    .REG_FEATURE(EBX, smep, 7)
    .REG_FEATURE(EBX, bmi2, 8)
    .REG_FEATURE(EBX, erms, 9)
    .REG_FEATURE(EBX, invpcid, 10)
    .REG_FEATURE(EBX, rtm, 11)
    .REG_FEATURE(EBX, pqm, 12)
    //.REG_FEATURE(EBX, , 13)
    .REG_FEATURE(EBX, mpx, 14)
    .REG_FEATURE(EBX, pqe, 15)
    .REG_FEATURE(EBX, avx512_f, 16)
    .REG_FEATURE(EBX, avx512_dq, 17)
    .REG_FEATURE(EBX, rdseed, 18)
    .REG_FEATURE(EBX, adx, 19)
    .REG_FEATURE(EBX, smap, 20)
    .REG_FEATURE(EBX, avx512_ifma, 21)
    .REG_FEATURE(EBX, pcommit, 22)
    .REG_FEATURE(EBX, clflushopt, 23)
    .REG_FEATURE(EBX, clwb, 24)
    .REG_FEATURE(EBX, intel_pt, 25)
    .REG_FEATURE(EBX, avx512_pf, 26)
    .REG_FEATURE(EBX, avx512_er, 27)
    .REG_FEATURE(EBX, avx512_cd, 28)
    .REG_FEATURE(EBX, sha, 29)
    .REG_FEATURE(EBX, avx512_bw, 30)
    .REG_FEATURE(EBX, avx512_vl, 31)
    .REG_FEATURE(ECX, prefetchwt1, 0)
    .REG_FEATURE(ECX, avx512_vbmi, 1)
    .REG_FEATURE(ECX, umip, 2)
    .REG_FEATURE(ECX, pku, 3)
    .REG_FEATURE(ECX, ospke, 4)
    .REG_FEATURE(ECX, waitpkg, 5)
    .REG_FEATURE(ECX, avx512_vbmi2, 6)
    .REG_FEATURE(ECX, shstk, 7)
    .REG_FEATURE(ECX, gfni, 8)
    .REG_FEATURE(ECX, vaes, 9)
    .REG_FEATURE(ECX, vpclmulqdq, 10)
    .REG_FEATURE(ECX, avx512_vnni, 11)
    .REG_FEATURE(ECX, avx512_bitalg, 12)
    //.REG_FEATURE(ECX, , 13)
    .REG_FEATURE(ECX, avx512_vpopcntdq, 14)
    //.REG_FEATURE(ECX, , 15)
    .REG_FEATURE(ECX, 5-level-paging, 16)
    .REG_FEATURE_VAL(ECX, mawau, 17, 21) // from 17 ~ 21 is mawau
    .REG_FEATURE(ECX, rdpid , 22)
    //.REG_FEATURE(ECX, , 23)
    //.REG_FEATURE(ECX, , 24)
    .REG_FEATURE(ECX, cldemote, 25)
    //.REG_FEATURE(ECX, , 26)
    .REG_FEATURE(ECX, MOVDIRI, 27)
    .REG_FEATURE(ECX, MOVDIR64B, 28)
    .REG_FEATURE(ECX, ENQCMD, 29)
    .REG_FEATURE(ECX, sgx_lc, 30)
    //.REG_FEATURE(ECX, , 31)
    //.REG_FEATURE(EDX, , 0)
    //.REG_FEATURE(EDX, , 1)
    .REG_FEATURE(EDX, avx512_4vnniw, 2)
    .REG_FEATURE(EDX, avx512_4fmaps, 3)
    .REG_FEATURE(EDX, fsrm, 4)
    //.REG_FEATURE(EDX, , 5)
    //.REG_FEATURE(EDX, , 6)
    //.REG_FEATURE(EDX, , 7)
    .REG_FEATURE(EDX, avx512_vp2intersect, 8)
    //.REG_FEATURE(EDX, , 9)
    .REG_FEATURE(EDX, md_clear, 10)
    //.REG_FEATURE(EDX, , 11)
    //.REG_FEATURE(EDX, , 12)
    .REG_FEATURE(EDX, tsx_force_abort, 13)
    .REG_FEATURE(EDX, SERIALIZE, 14)
    .REG_FEATURE(EDX, Hybrid, 15)
    .REG_FEATURE(EDX, TSXLDTRK, 16)
    //.REG_FEATURE(EDX, , 17)
    .REG_FEATURE(EDX, pconfig, 18)
    //.REG_FEATURE(EDX, , 19)
    .REG_FEATURE(EDX, ibt, 20)
    .REG_FEATURE(EDX, , 21)
    .REG_FEATURE(EDX, , 22)
    .REG_FEATURE(EDX, , 23)
    .REG_FEATURE(EDX, , 24)
    .REG_FEATURE(EDX, , 25)
    .REG_FEATURE(EDX, IBRS_IBPB/spec_ctrl, 26)
    .REG_FEATURE(EDX, stibp, 27)
    //.REG_FEATURE(EDX, , 28)
    .REG_FEATURE(EDX, IA32_ARCH_CAPABILITIES, 29)
    .REG_FEATURE(EDX, IA32_CORE_CAPABILITIES, 30)
    .REG_FEATURE(EDX, ssbd, 31);

// Features in %eax for leaf 7 sub-leaf 1
DECLARE_FEATURE_LEAF_WITH_SUB(7, 0)
    .REG_FEATURE(EAX, avx512_bf16, 5);

// Features in %eax for leaf 13 sub-leaf 1
DECLARE_FEATURE_LEAF_WITH_SUB(13, 1)
    .REG_FEATURE(EAX, XSAVEOPT, 0)
    .REG_FEATURE(EAX, XSAVEC, 1)
    .REG_FEATURE(EAX, XSAVES, 3);

// Features in %eax for leaf 14 sub-leaf 0
DECLARE_FEATURE_LEAF_WITH_SUB(14, 0)
    .REG_FEATURE(EAX, PTWRITE, 4);

// Features for leaf 0x80000000
DECLARE_FEATURE_LEAF(0x80000000)
    .REG_FEATURE_VAL(EAX, highest_extended_function_id, 0, 31);

// Features for leaf 0x80000001
DECLARE_FEATURE_LEAF(0x80000001)
    .REG_FEATURE(ECX, lahf_lm, 0)
    .REG_FEATURE(ECX, cmp_legacy, 1)
    .REG_FEATURE(ECX, svm, 2)
    .REG_FEATURE(ECX, extapic, 3)
    .REG_FEATURE(ECX, cr8_legacy, 4)
    .REG_FEATURE(ECX, abm, 5)
    .REG_FEATURE(ECX, sse4a, 6)
    .REG_FEATURE(ECX, misalignsse, 7)
    .REG_FEATURE(ECX, 3dnowprefetch, 8)
    .REG_FEATURE(ECX, osvw, 9)
    .REG_FEATURE(ECX, ibs, 10)
    .REG_FEATURE(ECX, xop, 11)
    .REG_FEATURE(ECX, skinit, 12)
    .REG_FEATURE(ECX, wdt, 13)
    //.REG_FEATURE(ECX, , 14)
    .REG_FEATURE(ECX, lwp, 15)
    .REG_FEATURE(ECX, fma4, 16)
    .REG_FEATURE(ECX, tce, 17)
    //.REG_FEATURE(ECX, , 18)
    .REG_FEATURE(ECX, nodeid_msr, 19)
    //.REG_FEATURE(ECX, , 20)
    .REG_FEATURE(ECX, tbm, 21)
    .REG_FEATURE(ECX, topoext, 22)
    .REG_FEATURE(ECX, perfctr_core, 23)
    .REG_FEATURE(ECX, perfctr_nb, 24)
    //.REG_FEATURE(ECX, , 25)
    .REG_FEATURE(ECX, dbx, 26)
    .REG_FEATURE(ECX, perftsc, 27)
    .REG_FEATURE(ECX, pcx_l2i, 28)
    //.REG_FEATURE(ECX, , 29)
    //.REG_FEATURE(ECX, , 30)
    //.REG_FEATURE(ECX, , 31)
    //.REG_FEATURE(EDX, mmxext, 0)
    //.REG_FEATURE(EDX, , 1)
    //.REG_FEATURE(EDX, , 2)
    //.REG_FEATURE(EDX, , 3)
    //.REG_FEATURE(EDX, , 4)
    //.REG_FEATURE(EDX, , 5)
    //.REG_FEATURE(EDX, , 6)
    //.REG_FEATURE(EDX, , 7)
    //.REG_FEATURE(EDX, , 8)
    //.REG_FEATURE(EDX, , 9)
    //.REG_FEATURE(EDX, , 10)
    //.REG_FEATURE(EDX, , 11)
    //.REG_FEATURE(EDX, , 12)
    //.REG_FEATURE(EDX, , 13)
    //.REG_FEATURE(EDX, , 14)
    //.REG_FEATURE(EDX, , 15)
    //.REG_FEATURE(EDX, , 16)
    //.REG_FEATURE(EDX, , 17)
    //.REG_FEATURE(EDX, , 18)
    //.REG_FEATURE(EDX, , 19)
    //.REG_FEATURE(EDX, , 20)
    //.REG_FEATURE(EDX, , 21)
    .REG_FEATURE(EDX, mmxext, 22)
    .REG_FEATURE(EDX, mmx, 23)
    //.REG_FEATURE(EDX, , 24)
    //.REG_FEATURE(EDX, , 25)
    //.REG_FEATURE(EDX, , 26)
    //.REG_FEATURE(EDX, , 27)
    //.REG_FEATURE(EDX, , 28)
    .REG_FEATURE(EDX, lm, 29)
    .REG_FEATURE(EDX, 3dnowext, 30)
    .REG_FEATURE(EDX, 3dnow, 31);

// Features for leaf 0x80000002
DECLARE_FEATURE_LEAF(0x80000002)
    .REG_FEATURE_STR(EBX, brand_b, 0, 31)
    .REG_FEATURE_STR(ECX, brand_c, 0, 31)
    .REG_FEATURE_STR(EDX, brand_d, 0, 31);

// Features for leaf 0x80000003
DECLARE_FEATURE_LEAF(0x80000003)
    .REG_FEATURE_STR(EBX, brand_b, 0, 31)
    .REG_FEATURE_STR(ECX, brand_c, 0, 31)
    .REG_FEATURE_STR(EDX, brand_d, 0, 31);

// Features for leaf 0x80000004
DECLARE_FEATURE_LEAF(0x80000004)
    .REG_FEATURE_STR(EBX, brand_b, 0, 31)
    .REG_FEATURE_STR(ECX, brand_c, 0, 31)
    .REG_FEATURE_STR(EDX, brand_d, 0, 31);

// Features for leaf 0x80000006
DECLARE_FEATURE_LEAF(0x80000006)
    .REG_FEATURE_VAL(ECX, l2_Line_size, 0, 7)
    .REG_FEATURE_VAL(ECX, associativity, 12, 15)
    .REG_FEATURE_VAL(ECX, l2_cache_size, 16, 31);

// Features for leaf 0x80000008
DECLARE_FEATURE_LEAF(0x80000008)
    .REG_FEATURE_VAL(EAX, PhysicalAddressBits, 0, 7)
    .REG_FEATURE_VAL(EAX, VirtualLinearAddressBits, 8, 15);


bool CPUID::operator() (unsigned int leaf, 
                        Bits<32>& eax, 
                        Bits<32>& ebx, 
                        Bits<32>& ecx, 
                        Bits<32>& edx,
                        unsigned int sub_leaf) {
#if __i386__
    int cpuid_supported;
    __asm("  pushfl\n"
          "  popl   %%eax\n"
          "  movl   %%eax,%%ecx\n"
          "  xorl   $0x00200000,%%eax\n"
          "  pushl  %%eax\n"
          "  popfl\n"
          "  pushfl\n"
          "  popl   %%eax\n"
          "  movl   $0,%0\n"
          "  cmpl   %%eax,%%ecx\n"
          "  je     1f\n"
          "  movl   $1,%0\n"
          "1:"
          : "=r" (cpuid_supported) : \
          : "eax", "ecx");
    if (!cpuid_supported) {
        return false; 
    }
#endif
    if(sub_leaf < 0) {
        check(leaf, eax, ebx, ecx, edx);
    } else {
        check(leaf, sub_leaf, eax, ebx, ecx, edx);
    }
    return true;
}

void CPUID::check(unsigned int leaf, 
                  Bits<32>& eax, 
                  Bits<32>& ebx, 
                  Bits<32>& ecx, 
                  Bits<32>& edx) {
#if __i386__
    __asm("cpuid" 
          : "=a"(eax.to_data()), "=b" (ebx.to_data()), "=c"(ecx.to_data()), "=d"(edx.to_data()) \
          : "0"(leaf));
#else
    /* x86-64 uses %rbx as the base register, so preserve it. */
    __asm("  xchgq  %%rbx,%q1\n"
          "  cpuid\n"
          "  xchgq  %%rbx,%q1"
          : "=a"(eax.to_data()), "=r" (ebx.to_data()), "=c"(ecx.to_data()), "=d"(edx.to_data()) \
          : "0"(leaf));
#endif
    return true; 
}

void CPUID::check(unsigned int leaf, 
               unsigned int sub_leaf,
               Bits<32>& eax, 
               Bits<32>& ebx, 
               Bits<32>& ecx, 
               Bits<32>& edx) {
#if __i386__
    __asm("cpuid" 
          : "=a"(eax.to_data()), "=b" (ebx.to_data()), "=c"(ecx.to_data()), "=d"(edx.to_data()) \
          : "0"(leaf), "2"(sub_leaf));
#else
    /* x86-64 uses %rbx as the base register, so preserve it. */
    __asm("  xchgq  %%rbx,%q1\n"
          "  cpuid\n"
          "  xchgq  %%rbx,%q1"
          : "=a"(eax.to_data()), "=r" (ebx.to_data()), "=c"(ecx.to_data()), "=d"(edx.to_data()) \
          : "0"(leaf), "2"(sub_leaf));
#endif
    return true; 
}

X86Info::X86Info() noexcept {
    auto& leaf_0_pack = CPUFeatureQuery::Global().cpu_features["0@NULL"];

    auto query = [&](CPUFeatureQuery::Pack& pack) -> bool {
        bool ret = pack.query_leaf_func();
        if(pack.query_leaf_func()){
            fprintf(stderr, "X86 CPUID cmd error!\n");
            exit(1);
        }
        for(auto& it : pack.query_bits_func) { 
            auto& key = it.first;
            bool support = pack.query_bits_func[key]();
            insert(key, support);
        }
        for(auto& it : pack.query_val_func) { 
            auto& key = it.first;
            int val = pack.query_val_func[key]();
            insert(key, val);
        }
        for(auto& it : pack.query_str_func) { 
            auto& key = it.first;
            std::string str = pack.query_str_func[key]();
            insert(key, str);
        }
        return ret;
    };
    query(leaf_0_pack);
    // get highest_valid_function_id
    int highest_valid_function_id = get_val("highest_valid_function_id");
    // set vandor of cpu
    _vendor = get_str("vendor_b") + get_str("vendor_c") + get_str("vendor_d");

    // get highest_extended_function_id
    auto& leaf_0x80000000_pack = CPUFeatureQuery::Global().cpu_features["0x80000000NULL"];
    query(leaf_0x80000000_pack);
    int highest_extended_function_id = get_val("highest_extended_function_id");

    for(auto& it : CPUFeatureQuery::Global().cpu_features) { 
        int leaf = extract_leaf(it.first);
        if((leaf <= highest_valid_function_id) && (leaf > 0)) {
            auto& pack_tmp = CPUFeatureQuery::Global().cpu_features[it.first];
            query(pack_tmp);
        }
        if((leaf <= highest_extended_function_id) && (leaf > 0x80000000)) {
            auto& pack_tmp = CPUFeatureQuery::Global().cpu_features[it.first];
            query(pack_tmp);
        }
    }
    if(highest_extended_function_id >= 0x80000004) {
        // get brand info
        _brand = get_str("brand_b") + get_str("brand_c") + get_str("brand_d");
    }
    // sort into ascending order
    std::sort(_cpu_features.begin(), _cpu_features.end());
}

}
}
} 


