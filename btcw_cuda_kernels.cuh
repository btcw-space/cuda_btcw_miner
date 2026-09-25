
// CUDA compatibility layer for BTCW v40 OpenCL crypto source.
#include <cuda_runtime.h>
#include <stdint.h>

// OpenCL scalar aliases used by the original v40 crypto source.
typedef unsigned char uchar;
#ifdef _WIN32
typedef unsigned int uint;
typedef unsigned long long ulong;
// OpenCL/Linux long is 64-bit. MSVC long is 32-bit and truncates safegcd.
#define long long long
#define BTCW_MSVC_LONG_REMAP 1
#endif

static __device__ __forceinline__ ulong mul_hi(ulong a, ulong b) { return __umul64hi(a,b); }
static __device__ __forceinline__ uint rotate(uint x, uint n) { n &= 31u; return (x << n) | (x >> ((32u-n)&31u)); }
#define __NV_CL_C_VERSION 120
#define get_global_size(dim) ((uint)(gridDim.x * blockDim.x))
// =============================================================================
// UltrafastSecp256k1 OpenCL Kernels - Field Arithmetic
// =============================================================================
// secp256k1 field: F_p where p = 2^256 - 2^32 - 977
// Little-endian 256-bit integers using 4x64-bit limbs
// =============================================================================

// Field prime p = 2^256 - 0x1000003D1
// In 64-bit limbs (little-endian):
// p = {0xFFFFFFFEFFFFFC2F, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF}

// Constants
#define SECP256K1_P0 0xFFFFFFFEFFFFFC2FUL
#define SECP256K1_P1 0xFFFFFFFFFFFFFFFFUL
#define SECP256K1_P2 0xFFFFFFFFFFFFFFFFUL
#define SECP256K1_P3 0xFFFFFFFFFFFFFFFFUL

// K = 2^32 + 977 = 0x1000003D1 (for fast reduction)
#define SECP256K1_K 0x1000003D1UL

// =============================================================================
// Forced Inlining
// =============================================================================
// NVIDIA's OpenCL compiler (nvoc) treats 'inline' as advisory.
// __attribute__((always_inline)) forces inlining of the entire field arithmetic
// call chain (field_mul → comba → reduce), matching CUDA's __forceinline__.
#ifdef __NV_CL_C_VERSION
  #define FORCE_INLINE static __device__ __forceinline__
  #define FORCE_INLINE_STATIC static __device__ __forceinline__
#else
  #define FORCE_INLINE static __device__ __forceinline__
  #define FORCE_INLINE_STATIC static __device__ __forceinline__
#endif

// =============================================================================
// 64-bit Multiplication Helpers
// =============================================================================

// Multiply two 64-bit numbers, get 128-bit result as (x=lo, y=hi).
// Do not use CUDA ulong2 here: on Windows that type is 2x32-bit.
typedef struct { ulong x; ulong y; } u64x2;
FORCE_INLINE u64x2 mul64_full(ulong a, ulong b) {
    u64x2 r;
    r.x = a * b;
    r.y = mul_hi(a, b);
    return r;
}

// Add with carry: result = a + b + carry_in, returns new carry
FORCE_INLINE ulong add_with_carry(ulong a, ulong b, ulong carry_in, ulong* carry_out) {
    ulong sum = a + b;
    ulong c1 = (sum < a) ? 1UL : 0UL;
    sum += carry_in;
    ulong c2 = (sum < carry_in) ? 1UL : 0UL;
    *carry_out = c1 + c2;
    return sum;
}

// Subtract with borrow: result = a - b - borrow_in, returns new borrow
FORCE_INLINE ulong sub_with_borrow(ulong a, ulong b, ulong borrow_in, ulong* borrow_out) {
    ulong diff = a - b;
    ulong b1 = (a < b) ? 1UL : 0UL;
    ulong temp = diff;
    diff -= borrow_in;
    ulong b2 = (temp < borrow_in) ? 1UL : 0UL;
    *borrow_out = b1 + b2;
    return diff;
}

// =============================================================================
// Field Element Type (256-bit)
// =============================================================================

typedef struct {
    ulong limbs[4];  // Little-endian: limbs[0] is LSB
} FieldElement;

// =============================================================================
// NVIDIA OpenCL PTX Acceleration (Level 1+2+3)
// =============================================================================
// On consumer NVIDIA GPUs (Turing/Ampere/Ada/Blackwell), INT32 multiply
// throughput is 32x higher than INT64. Inline PTX enables:
//   Level 1+2: mad.lo.cc.u64/madc.hi.cc.u64 carry chains (no comparison-carry)
//   Level 3:   mad.lo.cc.u32/madc.hi.cc.u32 32-bit Comba (INT32 throughput)
// Fallback (AMD, Intel, portable): mul_hi + comparison-based carry unchanged.
// Guard: __NV_CL_C_VERSION is defined only by NVIDIA's OpenCL compiler.
// =============================================================================

#ifdef __NV_CL_C_VERSION

// 32-bit MAD accumulate: (r0:r1:r2) += a * b  [3-register 96-bit accumulator]
#define OCL_MAD32(r0, r1, r2, a, b) \
    asm volatile( \
        "mad.lo.cc.u32 %0, %3, %4, %0; \n\t" \
        "madc.hi.cc.u32 %1, %3, %4, %1; \n\t" \
        "addc.u32 %2, %2, 0; \n\t" \
        : "+r"(r0), "+r"(r1), "+r"(r2) \
        : "r"(a), "r"(b) \
    )

// 32-bit squaring diagonal: (r0:r1:r2) += a*a
#define OCL_SQR32_D(r0, r1, r2, a) \
    asm volatile( \
        "mad.lo.cc.u32 %0, %3, %3, %0; \n\t" \
        "madc.hi.cc.u32 %1, %3, %3, %1; \n\t" \
        "addc.u32 %2, %2, 0; \n\t" \
        : "+r"(r0), "+r"(r1), "+r"(r2) \
        : "r"(a) \
    )

// 32-bit squaring off-diagonal: (r0:r1:r2) += 2 * a*b
#define OCL_SQR32_M2(r0, r1, r2, a, b) \
    do { \
        uint _lo, _hi; \
        asm volatile( \
            "mul.lo.u32 %0, %2, %3; \n\t" \
            "mul.hi.u32 %1, %2, %3; \n\t" \
            : "=r"(_lo), "=r"(_hi) : "r"(a), "r"(b) \
        ); \
        asm volatile( \
            "add.cc.u32 %0, %0, %3; \n\t" \
            "addc.cc.u32 %1, %1, %4; \n\t" \
            "addc.u32 %2, %2, 0; \n\t" \
            "add.cc.u32 %0, %0, %3; \n\t" \
            "addc.cc.u32 %1, %1, %4; \n\t" \
            "addc.u32 %2, %2, 0; \n\t" \
            : "+r"(r0), "+r"(r1), "+r"(r2) : "r"(_lo), "r"(_hi) \
        ); \
    } while(0)

// ----------------------------------------------------------------------------
// 32-bit Comba multiplication: 4x64 FieldElement reinterpreted as 8x32 limbs.
// Produces uint[16] raw output (little-endian 32-bit limbs of 512-bit product).
// Mirrors CUDA's mul_256_comba32 from secp256k1_32_hybrid_final.cuh.
// ----------------------------------------------------------------------------
FORCE_INLINE_STATIC void mul_256_comba32_ocl(
    const FieldElement* a, const FieldElement* b, uint t32[16]
) {
    uint a32[8], b32[8];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        a32[2*i]   = (uint)(a->limbs[i]);
        a32[2*i+1] = (uint)(a->limbs[i] >> 32);
        b32[2*i]   = (uint)(b->limbs[i]);
        b32[2*i+1] = (uint)(b->limbs[i] >> 32);
    }
    uint r0 = 0, r1 = 0, r2 = 0;

    OCL_MAD32(r0,r1,r2, a32[0],b32[0]);
    t32[0]=r0; r0=r1; r1=r2; r2=0;

    OCL_MAD32(r0,r1,r2, a32[0],b32[1]); OCL_MAD32(r0,r1,r2, a32[1],b32[0]);
    t32[1]=r0; r0=r1; r1=r2; r2=0;

    OCL_MAD32(r0,r1,r2, a32[0],b32[2]); OCL_MAD32(r0,r1,r2, a32[1],b32[1]); OCL_MAD32(r0,r1,r2, a32[2],b32[0]);
    t32[2]=r0; r0=r1; r1=r2; r2=0;

    OCL_MAD32(r0,r1,r2, a32[0],b32[3]); OCL_MAD32(r0,r1,r2, a32[1],b32[2]); OCL_MAD32(r0,r1,r2, a32[2],b32[1]); OCL_MAD32(r0,r1,r2, a32[3],b32[0]);
    t32[3]=r0; r0=r1; r1=r2; r2=0;

    OCL_MAD32(r0,r1,r2, a32[0],b32[4]); OCL_MAD32(r0,r1,r2, a32[1],b32[3]); OCL_MAD32(r0,r1,r2, a32[2],b32[2]); OCL_MAD32(r0,r1,r2, a32[3],b32[1]); OCL_MAD32(r0,r1,r2, a32[4],b32[0]);
    t32[4]=r0; r0=r1; r1=r2; r2=0;

    OCL_MAD32(r0,r1,r2, a32[0],b32[5]); OCL_MAD32(r0,r1,r2, a32[1],b32[4]); OCL_MAD32(r0,r1,r2, a32[2],b32[3]); OCL_MAD32(r0,r1,r2, a32[3],b32[2]); OCL_MAD32(r0,r1,r2, a32[4],b32[1]); OCL_MAD32(r0,r1,r2, a32[5],b32[0]);
    t32[5]=r0; r0=r1; r1=r2; r2=0;

    OCL_MAD32(r0,r1,r2, a32[0],b32[6]); OCL_MAD32(r0,r1,r2, a32[1],b32[5]); OCL_MAD32(r0,r1,r2, a32[2],b32[4]); OCL_MAD32(r0,r1,r2, a32[3],b32[3]); OCL_MAD32(r0,r1,r2, a32[4],b32[2]); OCL_MAD32(r0,r1,r2, a32[5],b32[1]); OCL_MAD32(r0,r1,r2, a32[6],b32[0]);
    t32[6]=r0; r0=r1; r1=r2; r2=0;

    OCL_MAD32(r0,r1,r2, a32[0],b32[7]); OCL_MAD32(r0,r1,r2, a32[1],b32[6]); OCL_MAD32(r0,r1,r2, a32[2],b32[5]); OCL_MAD32(r0,r1,r2, a32[3],b32[4]); OCL_MAD32(r0,r1,r2, a32[4],b32[3]); OCL_MAD32(r0,r1,r2, a32[5],b32[2]); OCL_MAD32(r0,r1,r2, a32[6],b32[1]); OCL_MAD32(r0,r1,r2, a32[7],b32[0]);
    t32[7]=r0; r0=r1; r1=r2; r2=0;

    OCL_MAD32(r0,r1,r2, a32[1],b32[7]); OCL_MAD32(r0,r1,r2, a32[2],b32[6]); OCL_MAD32(r0,r1,r2, a32[3],b32[5]); OCL_MAD32(r0,r1,r2, a32[4],b32[4]); OCL_MAD32(r0,r1,r2, a32[5],b32[3]); OCL_MAD32(r0,r1,r2, a32[6],b32[2]); OCL_MAD32(r0,r1,r2, a32[7],b32[1]);
    t32[8]=r0; r0=r1; r1=r2; r2=0;

    OCL_MAD32(r0,r1,r2, a32[2],b32[7]); OCL_MAD32(r0,r1,r2, a32[3],b32[6]); OCL_MAD32(r0,r1,r2, a32[4],b32[5]); OCL_MAD32(r0,r1,r2, a32[5],b32[4]); OCL_MAD32(r0,r1,r2, a32[6],b32[3]); OCL_MAD32(r0,r1,r2, a32[7],b32[2]);
    t32[9]=r0; r0=r1; r1=r2; r2=0;

    OCL_MAD32(r0,r1,r2, a32[3],b32[7]); OCL_MAD32(r0,r1,r2, a32[4],b32[6]); OCL_MAD32(r0,r1,r2, a32[5],b32[5]); OCL_MAD32(r0,r1,r2, a32[6],b32[4]); OCL_MAD32(r0,r1,r2, a32[7],b32[3]);
    t32[10]=r0; r0=r1; r1=r2; r2=0;

    OCL_MAD32(r0,r1,r2, a32[4],b32[7]); OCL_MAD32(r0,r1,r2, a32[5],b32[6]); OCL_MAD32(r0,r1,r2, a32[6],b32[5]); OCL_MAD32(r0,r1,r2, a32[7],b32[4]);
    t32[11]=r0; r0=r1; r1=r2; r2=0;

    OCL_MAD32(r0,r1,r2, a32[5],b32[7]); OCL_MAD32(r0,r1,r2, a32[6],b32[6]); OCL_MAD32(r0,r1,r2, a32[7],b32[5]);
    t32[12]=r0; r0=r1; r1=r2; r2=0;

    OCL_MAD32(r0,r1,r2, a32[6],b32[7]); OCL_MAD32(r0,r1,r2, a32[7],b32[6]);
    t32[13]=r0; r0=r1; r1=r2; r2=0;

    OCL_MAD32(r0,r1,r2, a32[7],b32[7]);
    t32[14]=r0; t32[15]=r1;
}

// 32-bit Comba squaring: ~40% fewer multiplications (symmetry exploitation).
// Mirrors CUDA's sqr_256_comba32 from secp256k1_32_hybrid_final.cuh.
FORCE_INLINE_STATIC void sqr_256_comba32_ocl(const FieldElement* a, uint t32[16]) {
    uint a32[8];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        a32[2*i]   = (uint)(a->limbs[i]);
        a32[2*i+1] = (uint)(a->limbs[i] >> 32);
    }
    uint r0 = 0, r1 = 0, r2 = 0;

    OCL_SQR32_D(r0,r1,r2, a32[0]);
    t32[0]=r0; r0=r1; r1=r2; r2=0;

    OCL_SQR32_M2(r0,r1,r2, a32[0],a32[1]);
    t32[1]=r0; r0=r1; r1=r2; r2=0;

    OCL_SQR32_M2(r0,r1,r2, a32[0],a32[2]); OCL_SQR32_D(r0,r1,r2, a32[1]);
    t32[2]=r0; r0=r1; r1=r2; r2=0;

    OCL_SQR32_M2(r0,r1,r2, a32[0],a32[3]); OCL_SQR32_M2(r0,r1,r2, a32[1],a32[2]);
    t32[3]=r0; r0=r1; r1=r2; r2=0;

    OCL_SQR32_M2(r0,r1,r2, a32[0],a32[4]); OCL_SQR32_M2(r0,r1,r2, a32[1],a32[3]); OCL_SQR32_D(r0,r1,r2, a32[2]);
    t32[4]=r0; r0=r1; r1=r2; r2=0;

    OCL_SQR32_M2(r0,r1,r2, a32[0],a32[5]); OCL_SQR32_M2(r0,r1,r2, a32[1],a32[4]); OCL_SQR32_M2(r0,r1,r2, a32[2],a32[3]);
    t32[5]=r0; r0=r1; r1=r2; r2=0;

    OCL_SQR32_M2(r0,r1,r2, a32[0],a32[6]); OCL_SQR32_M2(r0,r1,r2, a32[1],a32[5]); OCL_SQR32_M2(r0,r1,r2, a32[2],a32[4]); OCL_SQR32_D(r0,r1,r2, a32[3]);
    t32[6]=r0; r0=r1; r1=r2; r2=0;

    OCL_SQR32_M2(r0,r1,r2, a32[0],a32[7]); OCL_SQR32_M2(r0,r1,r2, a32[1],a32[6]); OCL_SQR32_M2(r0,r1,r2, a32[2],a32[5]); OCL_SQR32_M2(r0,r1,r2, a32[3],a32[4]);
    t32[7]=r0; r0=r1; r1=r2; r2=0;

    OCL_SQR32_M2(r0,r1,r2, a32[1],a32[7]); OCL_SQR32_M2(r0,r1,r2, a32[2],a32[6]); OCL_SQR32_M2(r0,r1,r2, a32[3],a32[5]); OCL_SQR32_D(r0,r1,r2, a32[4]);
    t32[8]=r0; r0=r1; r1=r2; r2=0;

    OCL_SQR32_M2(r0,r1,r2, a32[2],a32[7]); OCL_SQR32_M2(r0,r1,r2, a32[3],a32[6]); OCL_SQR32_M2(r0,r1,r2, a32[4],a32[5]);
    t32[9]=r0; r0=r1; r1=r2; r2=0;

    OCL_SQR32_M2(r0,r1,r2, a32[3],a32[7]); OCL_SQR32_M2(r0,r1,r2, a32[4],a32[6]); OCL_SQR32_D(r0,r1,r2, a32[5]);
    t32[10]=r0; r0=r1; r1=r2; r2=0;

    OCL_SQR32_M2(r0,r1,r2, a32[4],a32[7]); OCL_SQR32_M2(r0,r1,r2, a32[5],a32[6]);
    t32[11]=r0; r0=r1; r1=r2; r2=0;

    OCL_SQR32_M2(r0,r1,r2, a32[5],a32[7]); OCL_SQR32_D(r0,r1,r2, a32[6]);
    t32[12]=r0; r0=r1; r1=r2; r2=0;

    OCL_SQR32_M2(r0,r1,r2, a32[6],a32[7]);
    t32[13]=r0; r0=r1; r1=r2; r2=0;

    OCL_SQR32_D(r0,r1,r2, a32[7]);
    t32[14]=r0; t32[15]=r1;
}

// 32-bit reduction: T_hi x K_MOD (32-bit MAD chain) + conditional P-subtract.
// Phase 1: T_hi[8..15] x 977 (scalar, 32-bit MAD chain)
// Phase 1b: add T_hi << 32  (K_MOD = 2^32 + 977)
// Phase 2: T_lo[0..7] += result (32-bit carry chain)
// Phase 3+4: pack to 64-bit, fold overflow, conditional P-subtract (64-bit PTX)
// Mirrors CUDA's reduce_512_to_256_32 from secp256k1_32_hybrid_final.cuh.
FORCE_INLINE_STATIC void reduce_512_to_256_32_ocl(uint t32[16], FieldElement* r) {
    uint t0=t32[0], t1=t32[1], t2=t32[2], t3=t32[3];
    uint t4=t32[4], t5=t32[5], t6=t32[6], t7=t32[7];
    const uint t8 =t32[8],  t9 =t32[9],  t10=t32[10], t11=t32[11];
    const uint t12=t32[12], t13=t32[13], t14=t32[14], t15=t32[15];

    // Phase 1: A = T_hi[8..15] x 977 (32-bit scalar MAD chain -> 9 limbs)
    uint a0, a1, a2, a3, a4, a5, a6, a7, a8;
    asm volatile(
        "mul.lo.u32 %0, %9,  977;\n\t"
        "mul.hi.u32 %1, %9,  977;\n\t"
        "mad.lo.cc.u32 %1, %10, 977, %1;\n\t"
        "madc.hi.u32 %2, %10, 977, 0;\n\t"
        "mad.lo.cc.u32 %2, %11, 977, %2;\n\t"
        "madc.hi.u32 %3, %11, 977, 0;\n\t"
        "mad.lo.cc.u32 %3, %12, 977, %3;\n\t"
        "madc.hi.u32 %4, %12, 977, 0;\n\t"
        "mad.lo.cc.u32 %4, %13, 977, %4;\n\t"
        "madc.hi.u32 %5, %13, 977, 0;\n\t"
        "mad.lo.cc.u32 %5, %14, 977, %5;\n\t"
        "madc.hi.u32 %6, %14, 977, 0;\n\t"
        "mad.lo.cc.u32 %6, %15, 977, %6;\n\t"
        "madc.hi.u32 %7, %15, 977, 0;\n\t"
        "mad.lo.cc.u32 %7, %16, 977, %7;\n\t"
        "madc.hi.u32 %8, %16, 977, 0;\n\t"
        : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3),"=r"(a4),
          "=r"(a5),"=r"(a6),"=r"(a7),"=r"(a8)
        : "r"(t8),"r"(t9),"r"(t10),"r"(t11),
          "r"(t12),"r"(t13),"r"(t14),"r"(t15)
    );

    // Phase 1b: add T_hi << 32 (a[1..8] += T_hi[8..15], yielding a9 overflow)
    uint a9;
    asm volatile(
        "add.cc.u32  %0, %0, %9;\n\t"
        "addc.cc.u32 %1, %1, %10;\n\t"
        "addc.cc.u32 %2, %2, %11;\n\t"
        "addc.cc.u32 %3, %3, %12;\n\t"
        "addc.cc.u32 %4, %4, %13;\n\t"
        "addc.cc.u32 %5, %5, %14;\n\t"
        "addc.cc.u32 %6, %6, %15;\n\t"
        "addc.cc.u32 %7, %7, %16;\n\t"
        "addc.u32    %8, 0, 0;\n\t"
        : "+r"(a1),"+r"(a2),"+r"(a3),"+r"(a4),
          "+r"(a5),"+r"(a6),"+r"(a7),"+r"(a8),"=r"(a9)
        : "r"(t8),"r"(t9),"r"(t10),"r"(t11),
          "r"(t12),"r"(t13),"r"(t14),"r"(t15)
    );

    // Phase 2: T_lo[0..7] += A[0..7] (32-bit carry chain)
    uint carry;
    asm volatile(
        "add.cc.u32  %0, %0, %9;\n\t"
        "addc.cc.u32 %1, %1, %10;\n\t"
        "addc.cc.u32 %2, %2, %11;\n\t"
        "addc.cc.u32 %3, %3, %12;\n\t"
        "addc.cc.u32 %4, %4, %13;\n\t"
        "addc.cc.u32 %5, %5, %14;\n\t"
        "addc.cc.u32 %6, %6, %15;\n\t"
        "addc.cc.u32 %7, %7, %16;\n\t"
        "addc.u32    %8, 0, 0;\n\t"
        : "+r"(t0),"+r"(t1),"+r"(t2),"+r"(t3),
          "+r"(t4),"+r"(t5),"+r"(t6),"+r"(t7),"=r"(carry)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),
          "r"(a4),"r"(a5),"r"(a6),"r"(a7)
    );

    // Phase 3: pack to 64-bit and fold overflow (extra * K)
    // Phase 3: overflow fold (fully 32-bit — no INT64 multiply)
    // extra = a8 + carry + a9*2^32, extra * K_MOD = extra*977 + extra<<32
    uint e_lo, e_carry;
    asm volatile(
        "add.cc.u32 %0, %2, %3;\n\t"
        "addc.u32 %1, 0, 0;\n\t"
        : "=r"(e_lo), "=r"(e_carry)
        : "r"(a8), "r"(carry)
    );
    uint e_hi = a9 + e_carry;
    uint p_lo, p_hi;
    asm volatile(
        "mul.lo.u32 %0, %2, 977;\n\t"
        "mul.hi.u32 %1, %2, 977;\n\t"
        : "=r"(p_lo), "=r"(p_hi)
        : "r"(e_lo)
    );
    uint q = e_hi * 977u;
    uint m1 = p_hi + q;
    uint ek0 = p_lo;
    uint ek1, ek1_carry;
    asm volatile(
        "add.cc.u32 %0, %2, %3;\n\t"
        "addc.u32 %1, 0, 0;\n\t"
        : "=r"(ek1), "=r"(ek1_carry)
        : "r"(m1), "r"(e_lo)
    );
    uint ek2 = e_hi + ek1_carry;
    ulong r0 = ((ulong)t1 << 32) | t0;
    ulong r1 = ((ulong)t3 << 32) | t2;
    ulong r2 = ((ulong)t5 << 32) | t4;
    ulong r3 = ((ulong)t7 << 32) | t6;
    ulong ek_lo = ((ulong)ek1 << 32) | ek0;
    ulong ek_hi = (ulong)ek2;
    ulong c;
    asm volatile(
        "add.cc.u64  %0, %0, %5;\n\t"
        "addc.cc.u64 %1, %1, %6;\n\t"
        "addc.cc.u64 %2, %2, 0;\n\t"
        "addc.cc.u64 %3, %3, 0;\n\t"
        "addc.u64    %4, 0, 0;\n\t"
        : "+l"(r0),"+l"(r1),"+l"(r2),"+l"(r3),"=l"(c)
        : "l"(ek_lo),"l"(ek_hi)
    );
    // CONSTANT-TIME rare-carry fold: was `if (c) { add SECP256K1_K }` — a
    // data-dependent branch on a secret-derived carry. Now always execute the add
    // with a masked addend (0 when c==0 -> no-op), so wavefront execution is
    // uniform. Mirrors the CUDA reduce_512_to_256_32 (proven CT via ncu).
    {
        ulong cmask = (ulong)0 - (ulong)(c != 0UL);
        asm volatile("" : "+l"(cmask));   // value barrier
        ulong cfold = (ulong)SECP256K1_K & cmask;
        asm volatile(
            "add.cc.u64  %0, %0, %4;\n\t"
            "addc.cc.u64 %1, %1, 0;\n\t"
            "addc.cc.u64 %2, %2, 0;\n\t"
            "addc.u64    %3, %3, 0;\n\t"
            : "+l"(r0),"+l"(r1),"+l"(r2),"+l"(r3)
            : "l"(cfold)
        );
    }

    // Phase 4: conditional subtraction of P (64-bit PTX sub.cc chain)
    ulong s0, s1, s2, s3, borrow;
    asm volatile(
        "sub.cc.u64  %0, %5, %9;\n\t"
        "subc.cc.u64 %1, %6, %10;\n\t"
        "subc.cc.u64 %2, %7, %11;\n\t"
        "subc.cc.u64 %3, %8, %12;\n\t"
        "subc.u64    %4, 0, 0;\n\t"
        : "=l"(s0),"=l"(s1),"=l"(s2),"=l"(s3),"=l"(borrow)
        : "l"(r0),"l"(r1),"l"(r2),"l"(r3),
          "l"(SECP256K1_P0),"l"(SECP256K1_P1),"l"(SECP256K1_P2),"l"(SECP256K1_P3)
    );
    // CONSTANT-TIME final reduction: was `if (borrow==0) r=s; else r=r;` — a
    // data-dependent branch (borrow==0 <=> r >= P). Now branchless cmov: select
    // the subtracted limbs s iff r >= P, else keep r, via a value-barriered mask.
    // Mirrors the CUDA reduce_512_to_256_32 (proven CT via ncu).
    {
        ulong mask = (ulong)0 - (ulong)(borrow == 0UL);
        asm volatile("" : "+l"(mask));   // value barrier
        r->limbs[0] = (s0 & mask) | (r0 & ~mask);
        r->limbs[1] = (s1 & mask) | (r1 & ~mask);
        r->limbs[2] = (s2 & mask) | (r2 & ~mask);
        r->limbs[3] = (s3 & mask) | (r3 & ~mask);
    }
}

#endif // __NV_CL_C_VERSION

// =============================================================================
// Field Reduction: r = a mod p
// Uses the fact that p = 2^256 - K where K = 0x1000003D1
// So 2^256 ≡ K (mod p), meaning we can reduce by replacing high bits with K*high
// =============================================================================

FORCE_INLINE void field_reduce(FieldElement* r, const ulong* a8) {
    // a8 is 512-bit number (8 limbs), reduce to 256-bit mod p
    // Since p = 2^256 - K, we have: a mod p = a_low + K * a_high (mod p)

    ulong carry = 0;
    ulong temp[5];

    // First reduction: fold a[4..7] into a[0..3] using K
    // temp = a[0..3] + K * a[4..7]

    // Process each high limb
    u64x2 prod;

    // limb 0: a[0] + K * a[4]
    prod = mul64_full(SECP256K1_K, a8[4]);
    temp[0] = a8[0] + prod.x;
    carry = (temp[0] < a8[0]) ? 1UL : 0UL;
    carry += prod.y;

    // limb 1: a[1] + K * a[5] + carry
    prod = mul64_full(SECP256K1_K, a8[5]);
    temp[1] = a8[1] + carry;
    ulong c1 = (temp[1] < carry) ? 1UL : 0UL;
    temp[1] += prod.x;
    c1 += (temp[1] < prod.x) ? 1UL : 0UL;
    carry = c1 + prod.y;

    // limb 2: a[2] + K * a[6] + carry
    prod = mul64_full(SECP256K1_K, a8[6]);
    temp[2] = a8[2] + carry;
    c1 = (temp[2] < carry) ? 1UL : 0UL;
    temp[2] += prod.x;
    c1 += (temp[2] < prod.x) ? 1UL : 0UL;
    carry = c1 + prod.y;

    // limb 3: a[3] + K * a[7] + carry
    prod = mul64_full(SECP256K1_K, a8[7]);
    temp[3] = a8[3] + carry;
    c1 = (temp[3] < carry) ? 1UL : 0UL;
    temp[3] += prod.x;
    c1 += (temp[3] < prod.x) ? 1UL : 0UL;
    temp[4] = c1 + prod.y;

    // Second reduction: fold temp[4]. CONSTANT-TIME: was `if (temp[4] != 0)` with a
    // nested `if (carry)` — data-dependent branches on a secret-derived overflow
    // during signing. Now ALWAYS folded (temp[4]==0 -> K*0=0 -> no-op) and the rare
    // carry fold is masked (mirrors the CUDA/Metal masked rare-carry fold; the
    // per-limb `? :` carries are branchless selects, not branches).
    {
        prod = mul64_full(SECP256K1_K, temp[4]);
        temp[0] += prod.x;
        carry = (temp[0] < prod.x) ? 1UL : 0UL;
        carry += prod.y;

        temp[1] += carry;
        carry = (temp[1] < carry) ? 1UL : 0UL;

        temp[2] += carry;
        carry = (temp[2] < carry) ? 1UL : 0UL;

        temp[3] += carry;
        carry = (temp[3] < carry) ? 1UL : 0UL;

        // Rare carry overflow (~2^-190): fold residual carry branchlessly (0 when carry==0).
        ulong cmask = (ulong)0 - (ulong)(carry != 0UL);
        ulong kfold = (ulong)SECP256K1_K & cmask;
        temp[0] += kfold;
        ulong c2 = (temp[0] < kfold) ? 1UL : 0UL;
        temp[1] += c2;
        c2 = (temp[1] < c2) ? 1UL : 0UL;
        temp[2] += c2;
        c2 = (temp[2] < c2) ? 1UL : 0UL;
        temp[3] += c2;
    }

    // Final reduction: if result >= p, subtract p
    // Check if result >= p by comparing limbs
    ulong borrow = 0;
    ulong diff[4];

    diff[0] = sub_with_borrow(temp[0], SECP256K1_P0, 0, &borrow);
    diff[1] = sub_with_borrow(temp[1], SECP256K1_P1, borrow, &borrow);
    diff[2] = sub_with_borrow(temp[2], SECP256K1_P2, borrow, &borrow);
    diff[3] = sub_with_borrow(temp[3], SECP256K1_P3, borrow, &borrow);

    // If no borrow, result >= p, use subtracted value
    // Otherwise, use original value
    // Branchless selection
    ulong mask = (borrow == 0) ? ~0UL : 0UL;

    r->limbs[0] = (diff[0] & mask) | (temp[0] & ~mask);
    r->limbs[1] = (diff[1] & mask) | (temp[1] & ~mask);
    r->limbs[2] = (diff[2] & mask) | (temp[2] & ~mask);
    r->limbs[3] = (diff[3] & mask) | (temp[3] & ~mask);
}

// =============================================================================
// Field Addition: r = (a + b) mod p
// =============================================================================

FORCE_INLINE void field_add_impl(FieldElement* r, const FieldElement* a, const FieldElement* b) {
#ifdef __NV_CL_C_VERSION
    // Level 2: native add.cc/addc carry chains (no comparison-based carry)
    ulong s0, s1, s2, s3, carry;
    asm volatile(
        "add.cc.u64  %0, %5, %9;\n\t"
        "addc.cc.u64 %1, %6, %10;\n\t"
        "addc.cc.u64 %2, %7, %11;\n\t"
        "addc.cc.u64 %3, %8, %12;\n\t"
        "addc.u64    %4, 0, 0;\n\t"
        : "=l"(s0),"=l"(s1),"=l"(s2),"=l"(s3),"=l"(carry)
        : "l"(a->limbs[0]),"l"(a->limbs[1]),"l"(a->limbs[2]),"l"(a->limbs[3]),
          "l"(b->limbs[0]),"l"(b->limbs[1]),"l"(b->limbs[2]),"l"(b->limbs[3])
    );
    ulong d0, d1, d2, d3, borrow;
    asm volatile(
        "sub.cc.u64  %0, %5, %9;\n\t"
        "subc.cc.u64 %1, %6, %10;\n\t"
        "subc.cc.u64 %2, %7, %11;\n\t"
        "subc.cc.u64 %3, %8, %12;\n\t"
        "subc.u64    %4, 0, 0;\n\t"
        : "=l"(d0),"=l"(d1),"=l"(d2),"=l"(d3),"=l"(borrow)
        : "l"(s0),"l"(s1),"l"(s2),"l"(s3),
          "l"(SECP256K1_P0),"l"(SECP256K1_P1),"l"(SECP256K1_P2),"l"(SECP256K1_P3)
    );
    // use diff if: no borrow (s >= P) OR carry from add (sum overflowed 2^256)
    ulong mask = ~borrow | (0UL - carry);
    r->limbs[0] = (d0 & mask) | (s0 & ~mask);
    r->limbs[1] = (d1 & mask) | (s1 & ~mask);
    r->limbs[2] = (d2 & mask) | (s2 & ~mask);
    r->limbs[3] = (d3 & mask) | (s3 & ~mask);
#else
    ulong carry = 0;
    ulong sum[4];
    sum[0] = add_with_carry(a->limbs[0], b->limbs[0], 0, &carry);
    sum[1] = add_with_carry(a->limbs[1], b->limbs[1], carry, &carry);
    sum[2] = add_with_carry(a->limbs[2], b->limbs[2], carry, &carry);
    sum[3] = add_with_carry(a->limbs[3], b->limbs[3], carry, &carry);
    ulong borrow = 0;
    ulong diff[4];
    diff[0] = sub_with_borrow(sum[0], SECP256K1_P0, 0, &borrow);
    diff[1] = sub_with_borrow(sum[1], SECP256K1_P1, borrow, &borrow);
    diff[2] = sub_with_borrow(sum[2], SECP256K1_P2, borrow, &borrow);
    diff[3] = sub_with_borrow(sum[3], SECP256K1_P3, borrow, &borrow);
    ulong use_diff = (carry != 0) | (borrow == 0);
    ulong mask = use_diff ? ~0UL : 0UL;
    r->limbs[0] = (diff[0] & mask) | (sum[0] & ~mask);
    r->limbs[1] = (diff[1] & mask) | (sum[1] & ~mask);
    r->limbs[2] = (diff[2] & mask) | (sum[2] & ~mask);
    r->limbs[3] = (diff[3] & mask) | (sum[3] & ~mask);
#endif
}

// =============================================================================
// Field Subtraction: r = (a - b) mod p
// =============================================================================

FORCE_INLINE void field_sub_impl(FieldElement* r, const FieldElement* a, const FieldElement* b) {
#ifdef __NV_CL_C_VERSION
    // Level 2: native sub.cc/subc + add.cc/addc carry chains
    ulong d0, d1, d2, d3, borrow;
    asm volatile(
        "sub.cc.u64  %0, %5, %9;\n\t"
        "subc.cc.u64 %1, %6, %10;\n\t"
        "subc.cc.u64 %2, %7, %11;\n\t"
        "subc.cc.u64 %3, %8, %12;\n\t"
        "subc.u64    %4, 0, 0;\n\t"
        : "=l"(d0),"=l"(d1),"=l"(d2),"=l"(d3),"=l"(borrow)
        : "l"(a->limbs[0]),"l"(a->limbs[1]),"l"(a->limbs[2]),"l"(a->limbs[3]),
          "l"(b->limbs[0]),"l"(b->limbs[1]),"l"(b->limbs[2]),"l"(b->limbs[3])
    );
    // borrow = 0xFFFF...FFFF if a < b (underflow), 0 otherwise
    ulong p0 = SECP256K1_P0 & borrow;
    ulong p1 = SECP256K1_P1 & borrow;
    ulong p2 = SECP256K1_P2 & borrow;
    ulong p3 = SECP256K1_P3 & borrow;
    asm volatile(
        "add.cc.u64  %0, %4, %8;\n\t"
        "addc.cc.u64 %1, %5, %9;\n\t"
        "addc.cc.u64 %2, %6, %10;\n\t"
        "addc.u64    %3, %7, %11;\n\t"
        : "=l"(r->limbs[0]),"=l"(r->limbs[1]),"=l"(r->limbs[2]),"=l"(r->limbs[3])
        : "l"(d0),"l"(d1),"l"(d2),"l"(d3), "l"(p0),"l"(p1),"l"(p2),"l"(p3)
    );
#else
    ulong borrow = 0;
    ulong diff[4];
    diff[0] = sub_with_borrow(a->limbs[0], b->limbs[0], 0, &borrow);
    diff[1] = sub_with_borrow(a->limbs[1], b->limbs[1], borrow, &borrow);
    diff[2] = sub_with_borrow(a->limbs[2], b->limbs[2], borrow, &borrow);
    diff[3] = sub_with_borrow(a->limbs[3], b->limbs[3], borrow, &borrow);
    ulong mask = borrow ? ~0UL : 0UL;
    ulong carry = 0;
    ulong adj[4];
    adj[0] = add_with_carry(diff[0], SECP256K1_P0 & mask, 0, &carry);
    adj[1] = add_with_carry(diff[1], SECP256K1_P1 & mask, carry, &carry);
    adj[2] = add_with_carry(diff[2], SECP256K1_P2 & mask, carry, &carry);
    adj[3] = add_with_carry(diff[3], SECP256K1_P3 & mask, carry, &carry);
    r->limbs[0] = adj[0];
    r->limbs[1] = adj[1];
    r->limbs[2] = adj[2];
    r->limbs[3] = adj[3];
#endif
}

// =============================================================================
// Field Multiplication: r = (a * b) mod p
// =============================================================================

// Helper: add 128-bit product (hi:lo) into 3-register accumulator (c2:c1:c0)
FORCE_INLINE void muladd(ulong lo, ulong hi, ulong* c0, ulong* c1, ulong* c2) {
    ulong carry;
    *c0 = add_with_carry(*c0, lo, 0, &carry);
    *c1 = add_with_carry(*c1, hi, carry, &carry);
    *c2 += carry;
}

// Helper: add 128-bit product (hi:lo) doubled into accumulator
FORCE_INLINE void muladd2(ulong lo, ulong hi, ulong* c0, ulong* c1, ulong* c2) {
    muladd(lo, hi, c0, c1, c2);
    muladd(lo, hi, c0, c1, c2);
}

FORCE_INLINE void field_mul_impl(FieldElement* r, const FieldElement* a, const FieldElement* b) {
#ifdef __NV_CL_C_VERSION
    // Level 3: 32-bit hybrid Comba + 32-bit reduction (INT32 throughput 32x > INT64)
    uint t32[16];
    mul_256_comba32_ocl(a, b, t32);
    reduce_512_to_256_32_ocl(t32, r);
#else
    ulong a0 = a->limbs[0], a1 = a->limbs[1], a2 = a->limbs[2], a3 = a->limbs[3];
    ulong b0 = b->limbs[0], b1 = b->limbs[1], b2 = b->limbs[2], b3 = b->limbs[3];
    ulong product[8];
    ulong c0, c1, c2;
    u64x2 m;

    // Column 0: a0*b0
    c0 = 0; c1 = 0; c2 = 0;
    m = mul64_full(a0, b0); muladd(m.x, m.y, &c0, &c1, &c2);
    product[0] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 1: a0*b1 + a1*b0
    m = mul64_full(a0, b1); muladd(m.x, m.y, &c0, &c1, &c2);
    m = mul64_full(a1, b0); muladd(m.x, m.y, &c0, &c1, &c2);
    product[1] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 2: a0*b2 + a1*b1 + a2*b0
    m = mul64_full(a0, b2); muladd(m.x, m.y, &c0, &c1, &c2);
    m = mul64_full(a1, b1); muladd(m.x, m.y, &c0, &c1, &c2);
    m = mul64_full(a2, b0); muladd(m.x, m.y, &c0, &c1, &c2);
    product[2] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 3: a0*b3 + a1*b2 + a2*b1 + a3*b0
    m = mul64_full(a0, b3); muladd(m.x, m.y, &c0, &c1, &c2);
    m = mul64_full(a1, b2); muladd(m.x, m.y, &c0, &c1, &c2);
    m = mul64_full(a2, b1); muladd(m.x, m.y, &c0, &c1, &c2);
    m = mul64_full(a3, b0); muladd(m.x, m.y, &c0, &c1, &c2);
    product[3] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 4: a1*b3 + a2*b2 + a3*b1
    m = mul64_full(a1, b3); muladd(m.x, m.y, &c0, &c1, &c2);
    m = mul64_full(a2, b2); muladd(m.x, m.y, &c0, &c1, &c2);
    m = mul64_full(a3, b1); muladd(m.x, m.y, &c0, &c1, &c2);
    product[4] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 5: a2*b3 + a3*b2
    m = mul64_full(a2, b3); muladd(m.x, m.y, &c0, &c1, &c2);
    m = mul64_full(a3, b2); muladd(m.x, m.y, &c0, &c1, &c2);
    product[5] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 6: a3*b3
    m = mul64_full(a3, b3); muladd(m.x, m.y, &c0, &c1, &c2);
    product[6] = c0;
    product[7] = c1;

    field_reduce(r, product);
#endif
}

// =============================================================================
// Field Squaring: r = a² mod p
// =============================================================================

// Forward declaration for field_sqr_n_impl
FORCE_INLINE void field_sqr_impl(FieldElement* r, const FieldElement* a);

// Repeated squaring helper: r = r^(2^n) — in-place
FORCE_INLINE void field_sqr_n_impl(FieldElement* r, int n) {
    for (int i = 0; i < n; i++) field_sqr_impl(r, r);
}

FORCE_INLINE void field_sqr_impl(FieldElement* r, const FieldElement* a) {
#ifdef __NV_CL_C_VERSION
    // Level 3: 32-bit hybrid squaring (40% fewer multiplications + INT32 throughput)
    uint t32[16];
    sqr_256_comba32_ocl(a, t32);
    reduce_512_to_256_32_ocl(t32, r);
#else
    ulong a0 = a->limbs[0], a1 = a->limbs[1], a2 = a->limbs[2], a3 = a->limbs[3];
    ulong product[8];
    ulong c0, c1, c2;
    u64x2 m;

    // Column 0: a0*a0
    c0 = 0; c1 = 0; c2 = 0;
    m = mul64_full(a0, a0); muladd(m.x, m.y, &c0, &c1, &c2);
    product[0] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 1: 2*a0*a1
    m = mul64_full(a0, a1); muladd2(m.x, m.y, &c0, &c1, &c2);
    product[1] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 2: 2*a0*a2 + a1*a1
    m = mul64_full(a0, a2); muladd2(m.x, m.y, &c0, &c1, &c2);
    m = mul64_full(a1, a1); muladd(m.x, m.y, &c0, &c1, &c2);
    product[2] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 3: 2*a0*a3 + 2*a1*a2
    m = mul64_full(a0, a3); muladd2(m.x, m.y, &c0, &c1, &c2);
    m = mul64_full(a1, a2); muladd2(m.x, m.y, &c0, &c1, &c2);
    product[3] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 4: 2*a1*a3 + a2*a2
    m = mul64_full(a1, a3); muladd2(m.x, m.y, &c0, &c1, &c2);
    m = mul64_full(a2, a2); muladd(m.x, m.y, &c0, &c1, &c2);
    product[4] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 5: 2*a2*a3
    m = mul64_full(a2, a3); muladd2(m.x, m.y, &c0, &c1, &c2);
    product[5] = c0; c0 = c1; c1 = c2; c2 = 0;

    // Column 6: a3*a3
    m = mul64_full(a3, a3); muladd(m.x, m.y, &c0, &c1, &c2);
    product[6] = c0;
    product[7] = c1;

    field_reduce(r, product);
#endif
}

// =============================================================================
// Field Negation: r = -a mod p = p - a
// =============================================================================

FORCE_INLINE void field_neg_impl(FieldElement* r, const FieldElement* a) {
    // Check if a is zero
    ulong is_zero = ((a->limbs[0] | a->limbs[1] | a->limbs[2] | a->limbs[3]) == 0) ? 1UL : 0UL;

    ulong borrow = 0;
    r->limbs[0] = sub_with_borrow(SECP256K1_P0, a->limbs[0], 0, &borrow);
    r->limbs[1] = sub_with_borrow(SECP256K1_P1, a->limbs[1], borrow, &borrow);
    r->limbs[2] = sub_with_borrow(SECP256K1_P2, a->limbs[2], borrow, &borrow);
    r->limbs[3] = sub_with_borrow(SECP256K1_P3, a->limbs[3], borrow, &borrow);

    // If a was zero, result should be zero
    ulong mask = is_zero ? 0UL : ~0UL;
    r->limbs[0] &= mask;
    r->limbs[1] &= mask;
    r->limbs[2] &= mask;
    r->limbs[3] &= mask;
}

// =============================================================================
// Field Inversion: r = a^(-1) mod p
// Using Fermat's little theorem with optimized addition chain
// Matches CUDA's field_inv_fermat_chain for minimal mul+sqr count
// p-2 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
// =============================================================================

FORCE_INLINE void field_inv_impl(FieldElement* r, const FieldElement* a) {
    FieldElement x2, x3, x6, x12, x24, x48, x96, x192, x7, x31, x223;
    FieldElement x5, x11, x22;
    FieldElement t;

    // 1. x2 = a^2 * a  (2 consecutive ones)
    field_sqr_impl(&x2, a);
    field_mul_impl(&x2, &x2, a);

    // 2. x3 = x2^2 * a  (3 consecutive ones)
    field_sqr_impl(&x3, &x2);
    field_mul_impl(&x3, &x3, a);

    // 3. x6 = x3^(2^3) * x3  (6 consecutive ones)
    field_sqr_impl(&x6, &x3);
    field_sqr_n_impl(&x6, 2);
    field_mul_impl(&x6, &x6, &x3);

    // 4. x12 = x6^(2^6) * x6  (12 consecutive ones)
    t = x6;
    field_sqr_n_impl(&t, 6);
    field_mul_impl(&x12, &t, &x6);

    // 5. x24 = x12^(2^12) * x12  (24 consecutive ones)
    t = x12;
    field_sqr_n_impl(&t, 12);
    field_mul_impl(&x24, &t, &x12);

    // 6. x48 = x24^(2^24) * x24  (48 consecutive ones)
    t = x24;
    field_sqr_n_impl(&t, 24);
    field_mul_impl(&x48, &t, &x24);

    // 7. x96 = x48^(2^48) * x48  (96 consecutive ones)
    t = x48;
    field_sqr_n_impl(&t, 48);
    field_mul_impl(&x96, &t, &x48);

    // 8. x192 = x96^(2^96) * x96  (192 consecutive ones)
    t = x96;
    field_sqr_n_impl(&t, 96);
    field_mul_impl(&x192, &t, &x96);

    // 9. x7 = x6^2 * a  (7 consecutive ones)
    field_sqr_impl(&x7, &x6);
    field_mul_impl(&x7, &x7, a);

    // 10. x31 = x24^(2^7) * x7  (31 consecutive ones)
    t = x24;
    field_sqr_n_impl(&t, 7);
    field_mul_impl(&x31, &t, &x7);

    // 11. x223 = x192^(2^31) * x31  (223 consecutive ones)
    t = x192;
    field_sqr_n_impl(&t, 31);
    field_mul_impl(&x223, &t, &x31);

    // 12. x5 = x3^(2^2) * x2  (5 consecutive ones)
    t = x3;
    field_sqr_n_impl(&t, 2);
    field_mul_impl(&x5, &t, &x2);

    // 13. x11 = x6^(2^5) * x5  (11 consecutive ones)
    t = x6;
    field_sqr_n_impl(&t, 5);
    field_mul_impl(&x11, &t, &x5);

    // 14. x22 = x11^(2^11) * x11  (22 consecutive ones)
    t = x11;
    field_sqr_n_impl(&t, 11);
    field_mul_impl(&x22, &t, &x11);

    // 15. t = x223^2  (bit 32 is 0)
    field_sqr_impl(&t, &x223);

    // 16. t = t^(2^22) * x22  (append 22 ones)
    field_sqr_n_impl(&t, 22);
    field_mul_impl(&t, &t, &x22);

    // 17. t = t^(2^4)  (bits 9,8,7,6 are 0)
    field_sqr_n_impl(&t, 4);

    // 18. Process remaining 6 bits: 101101
    // bit 5: 1
    field_sqr_impl(&t, &t);
    field_mul_impl(&t, &t, a);
    // bit 4: 0
    field_sqr_impl(&t, &t);
    // bit 3: 1
    field_sqr_impl(&t, &t);
    field_mul_impl(&t, &t, a);
    // bit 2: 1
    field_sqr_impl(&t, &t);
    field_mul_impl(&t, &t, a);
    // bit 1: 0
    field_sqr_impl(&t, &t);
    // bit 0: 1
    field_sqr_impl(&t, &t);
    field_mul_impl(r, &t, a);
}

// =============================================================================
// OpenCL Kernels
// =============================================================================

extern "C" __global__ void field_add(
    const FieldElement* a,
    const FieldElement* b,
    FieldElement* result,
    const uint count
) {
    uint gid = ((uint)(blockIdx.x * blockDim.x + threadIdx.x));
    if (gid >= count) return;

    // Copy from global to private memory
    FieldElement a_local = a[gid];
    FieldElement b_local = b[gid];
    FieldElement r;
    field_add_impl(&r, &a_local, &b_local);
    result[gid] = r;
}

extern "C" __global__ void field_sub(
    const FieldElement* a,
    const FieldElement* b,
    FieldElement* result,
    const uint count
) {
    uint gid = ((uint)(blockIdx.x * blockDim.x + threadIdx.x));
    if (gid >= count) return;

    // Copy from global to private memory
    FieldElement a_local = a[gid];
    FieldElement b_local = b[gid];
    FieldElement r;
    field_sub_impl(&r, &a_local, &b_local);
    result[gid] = r;
}

extern "C" __global__ void field_mul(
    const FieldElement* a,
    const FieldElement* b,
    FieldElement* result,
    const uint count
) {
    uint gid = ((uint)(blockIdx.x * blockDim.x + threadIdx.x));
    if (gid >= count) return;

    // Copy from global to private memory
    FieldElement a_local = a[gid];
    FieldElement b_local = b[gid];
    FieldElement r;
    field_mul_impl(&r, &a_local, &b_local);
    result[gid] = r;
}

extern "C" __global__ void field_sqr(
    const FieldElement* a,
    FieldElement* result,
    const uint count
) {
    uint gid = ((uint)(blockIdx.x * blockDim.x + threadIdx.x));
    if (gid >= count) return;

    // Copy from global to private memory
    FieldElement a_local = a[gid];
    FieldElement r;
    field_sqr_impl(&r, &a_local);
    result[gid] = r;
}

extern "C" __global__ void field_inv(
    const FieldElement* a,
    FieldElement* result,
    const uint count
) {
    uint gid = ((uint)(blockIdx.x * blockDim.x + threadIdx.x));
    if (gid >= count) return;

    // Copy from global to private memory
    FieldElement a_local = a[gid];
    FieldElement r;
    field_inv_impl(&r, &a_local);
    result[gid] = r;
}


// =============================================================================
// UltrafastSecp256k1 OpenCL Kernels - Point Operations
// =============================================================================
// Elliptic curve point operations on secp256k1: y² = x³ + 7
// Jacobian coordinates for efficient operations
// =============================================================================

// Include field arithmetic
// field arithmetic included via host concatenation

// =============================================================================
// Curve Constants
// =============================================================================

// Generator point G (affine coordinates)
// Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
// Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

#define SECP256K1_GX0 0x59F2815B16F81798UL
#define SECP256K1_GX1 0x029BFCDB2DCE28D9UL
#define SECP256K1_GX2 0x55A06295CE870B07UL
#define SECP256K1_GX3 0x79BE667EF9DCBBACUL

#define SECP256K1_GY0 0x9C47D08FFB10D4B8UL
#define SECP256K1_GY1 0xFD17B448A6855419UL
#define SECP256K1_GY2 0x5DA4FBFC0E1108A8UL
#define SECP256K1_GY3 0x483ADA7726A3C465UL

// Curve order n
#define SECP256K1_N0 0xBFD25E8CD0364141UL
#define SECP256K1_N1 0xBAAEDCE6AF48A03BUL
#define SECP256K1_N2 0xFFFFFFFFFFFFFFFEUL
#define SECP256K1_N3 0xFFFFFFFFFFFFFFFFUL

// =============================================================================
// Point Types
// =============================================================================

typedef struct {
    FieldElement x;
    FieldElement y;
} AffinePoint;

typedef struct {
    FieldElement x;
    FieldElement y;
    FieldElement z;
    uint infinity;  // 1 if point at infinity
    uint pad[7];    // Match host alignas(128) layout — sizeof = 128 bytes
} JacobianPoint;

typedef struct {
    ulong limbs[4];
} Scalar;

// =============================================================================
// Point Utilities
// =============================================================================

FORCE_INLINE void point_set_infinity(JacobianPoint* p) {
    p->x.limbs[0] = 0; p->x.limbs[1] = 0; p->x.limbs[2] = 0; p->x.limbs[3] = 0;
    p->y.limbs[0] = 1; p->y.limbs[1] = 0; p->y.limbs[2] = 0; p->y.limbs[3] = 0;
    p->z.limbs[0] = 0; p->z.limbs[1] = 0; p->z.limbs[2] = 0; p->z.limbs[3] = 0;
    p->infinity = 1;
}

FORCE_INLINE int point_is_infinity(const JacobianPoint* p) {
    return p->infinity ||
           ((p->z.limbs[0] | p->z.limbs[1] | p->z.limbs[2] | p->z.limbs[3]) == 0);
}

FORCE_INLINE void point_from_affine(JacobianPoint* j, const AffinePoint* a) {
    j->x = a->x;
    j->y = a->y;
    j->z.limbs[0] = 1; j->z.limbs[1] = 0; j->z.limbs[2] = 0; j->z.limbs[3] = 0;
    j->infinity = 0;
}

FORCE_INLINE void get_generator(AffinePoint* g) {
    g->x.limbs[0] = SECP256K1_GX0;
    g->x.limbs[1] = SECP256K1_GX1;
    g->x.limbs[2] = SECP256K1_GX2;
    g->x.limbs[3] = SECP256K1_GX3;

    g->y.limbs[0] = SECP256K1_GY0;
    g->y.limbs[1] = SECP256K1_GY1;
    g->y.limbs[2] = SECP256K1_GY2;
    g->y.limbs[3] = SECP256K1_GY3;
}

// =============================================================================
// Point Doubling: R = 2*P (Jacobian coordinates)
// Using standard doubling formula for a = 0 curves (secp256k1)
// =============================================================================

FORCE_INLINE void point_double_impl(JacobianPoint* r, const JacobianPoint* p) {
    if (point_is_infinity(p)) {
        point_set_infinity(r);
        return;
    }

    // Check if Y = 0 (point of order 2, but secp256k1 doesn't have one)
    if ((p->y.limbs[0] | p->y.limbs[1] | p->y.limbs[2] | p->y.limbs[3]) == 0) {
        point_set_infinity(r);
        return;
    }

    FieldElement S, M, X3, Y3, Z3, YY, YYYY, ZZ, t1, t2;

    // S = 4*X*Y^2
    field_sqr_impl(&YY, &p->y);           // YY = Y^2
    field_mul_impl(&S, &p->x, &YY);       // S = X * Y^2
    field_add_impl(&S, &S, &S);           // S = 2*X*Y^2
    field_add_impl(&S, &S, &S);           // S = 4*X*Y^2

    // M = 3*X^2 (since a=0 for secp256k1)
    field_sqr_impl(&M, &p->x);            // M = X^2
    field_add_impl(&t1, &M, &M);          // t1 = 2*X^2
    field_add_impl(&M, &M, &t1);          // M = 3*X^2

    // X3 = M^2 - 2*S
    field_sqr_impl(&X3, &M);              // X3 = M^2
    field_add_impl(&t1, &S, &S);          // t1 = 2*S
    field_sub_impl(&X3, &X3, &t1);        // X3 = M^2 - 2*S

    // Y3 = M*(S - X3) - 8*Y^4
    field_sqr_impl(&YYYY, &YY);           // YYYY = Y^4
    field_add_impl(&t1, &YYYY, &YYYY);    // t1 = 2*Y^4
    field_add_impl(&t1, &t1, &t1);        // t1 = 4*Y^4
    field_add_impl(&t1, &t1, &t1);        // t1 = 8*Y^4
    field_sub_impl(&t2, &S, &X3);         // t2 = S - X3
    field_mul_impl(&Y3, &M, &t2);         // Y3 = M*(S - X3)
    field_sub_impl(&Y3, &Y3, &t1);        // Y3 = M*(S - X3) - 8*Y^4

    // Z3 = 2*Y*Z
    field_mul_impl(&Z3, &p->y, &p->z);    // Z3 = Y*Z
    field_add_impl(&Z3, &Z3, &Z3);        // Z3 = 2*Y*Z

    r->x = X3;
    r->y = Y3;
    r->z = Z3;
    r->infinity = 0;
}

// =============================================================================
// Point Addition: R = P + Q (Jacobian + Jacobian)
// Complete addition formula
// =============================================================================

// Unchecked doubling: skips infinity and Y==0 checks.
// Precondition: p is a valid, non-infinity point with Y != 0.
FORCE_INLINE void point_double_unchecked(JacobianPoint* r, const JacobianPoint* p) {
    FieldElement S, M, X3, Y3, Z3, YY, YYYY, t1, t2;

    field_sqr_impl(&YY, &p->y);
    field_mul_impl(&S, &p->x, &YY);
    field_add_impl(&S, &S, &S);
    field_add_impl(&S, &S, &S);

    field_sqr_impl(&M, &p->x);
    field_add_impl(&t1, &M, &M);
    field_add_impl(&M, &M, &t1);

    field_sqr_impl(&X3, &M);
    field_add_impl(&t1, &S, &S);
    field_sub_impl(&X3, &X3, &t1);

    field_sqr_impl(&YYYY, &YY);
    field_add_impl(&t1, &YYYY, &YYYY);
    field_add_impl(&t1, &t1, &t1);
    field_add_impl(&t1, &t1, &t1);
    field_sub_impl(&t2, &S, &X3);
    field_mul_impl(&Y3, &M, &t2);
    field_sub_impl(&Y3, &Y3, &t1);

    field_mul_impl(&Z3, &p->y, &p->z);
    field_add_impl(&Z3, &Z3, &Z3);

    r->x = X3;
    r->y = Y3;
    r->z = Z3;
    r->infinity = 0;
}

// Unchecked mixed addition: skips p->infinity check.
// Precondition: p is a valid, non-infinity Jacobian point.
// Keeps the H==0 check for algebraic completeness.
FORCE_INLINE void point_add_mixed_unchecked(JacobianPoint* r, const JacobianPoint* p, const AffinePoint* q) {
    FieldElement Z1Z1, U2, S2, H, HH, I, J, rr, V, X3, Y3, Z3, t1, t2;

    field_sqr_impl(&Z1Z1, &p->z);
    field_mul_impl(&U2, &q->x, &Z1Z1);
    field_mul_impl(&t1, &q->y, &p->z);
    field_mul_impl(&S2, &t1, &Z1Z1);
    field_sub_impl(&H, &U2, &p->x);

    if ((H.limbs[0] | H.limbs[1] | H.limbs[2] | H.limbs[3]) == 0) {
        field_sub_impl(&t1, &S2, &p->y);
        if ((t1.limbs[0] | t1.limbs[1] | t1.limbs[2] | t1.limbs[3]) == 0) {
            point_double_unchecked(r, p);
            return;
        }
        point_set_infinity(r);
        return;
    }

    field_sqr_impl(&HH, &H);
    field_add_impl(&I, &HH, &HH);
    field_add_impl(&I, &I, &I);
    field_mul_impl(&J, &H, &I);
    field_sub_impl(&rr, &S2, &p->y);
    field_add_impl(&rr, &rr, &rr);
    field_mul_impl(&V, &p->x, &I);

    field_sqr_impl(&X3, &rr);
    field_sub_impl(&X3, &X3, &J);
    field_add_impl(&t1, &V, &V);
    field_sub_impl(&X3, &X3, &t1);

    field_sub_impl(&t1, &V, &X3);
    field_mul_impl(&Y3, &rr, &t1);
    field_mul_impl(&t2, &p->y, &J);
    field_add_impl(&t2, &t2, &t2);
    field_sub_impl(&Y3, &Y3, &t2);

    field_add_impl(&t1, &p->z, &H);
    field_sqr_impl(&Z3, &t1);
    field_sub_impl(&Z3, &Z3, &Z1Z1);
    field_sub_impl(&Z3, &Z3, &HH);

    r->x = X3;
    r->y = Y3;
    r->z = Z3;
}

// =============================================================================
// Point Addition: R = P + Q (Jacobian + Jacobian)
// Complete addition formula
// =============================================================================

FORCE_INLINE void point_add_impl(JacobianPoint* r, const JacobianPoint* p, const JacobianPoint* q) {
    // Handle infinity cases
    if (point_is_infinity(p)) {
        *r = *q;
        return;
    }
    if (point_is_infinity(q)) {
        *r = *p;
        return;
    }

    FieldElement U1, U2, S1, S2, H, I, J, rr, V, X3, Y3, Z3;
    FieldElement Z1Z1, Z2Z2, t1, t2;

    // Z1Z1 = Z1^2
    field_sqr_impl(&Z1Z1, &p->z);

    // Z2Z2 = Z2^2
    field_sqr_impl(&Z2Z2, &q->z);

    // U1 = X1*Z2Z2
    field_mul_impl(&U1, &p->x, &Z2Z2);

    // U2 = X2*Z1Z1
    field_mul_impl(&U2, &q->x, &Z1Z1);

    // S1 = Y1*Z2*Z2Z2
    field_mul_impl(&t1, &p->y, &q->z);
    field_mul_impl(&S1, &t1, &Z2Z2);

    // S2 = Y2*Z1*Z1Z1
    field_mul_impl(&t1, &q->y, &p->z);
    field_mul_impl(&S2, &t1, &Z1Z1);

    // H = U2 - U1
    field_sub_impl(&H, &U2, &U1);

    // Check if H = 0 (points have same X coordinate)
    if ((H.limbs[0] | H.limbs[1] | H.limbs[2] | H.limbs[3]) == 0) {
        // Check if S1 == S2 (same point, do doubling)
        field_sub_impl(&t1, &S2, &S1);
        if ((t1.limbs[0] | t1.limbs[1] | t1.limbs[2] | t1.limbs[3]) == 0) {
            point_double_impl(r, p);
            return;
        }
        // Points are negatives, result is infinity
        point_set_infinity(r);
        return;
    }

    // I = (2*H)^2
    field_add_impl(&I, &H, &H);           // I = 2*H
    field_sqr_impl(&I, &I);               // I = (2*H)^2

    // J = H*I
    field_mul_impl(&J, &H, &I);

    // r = 2*(S2 - S1)
    field_sub_impl(&rr, &S2, &S1);
    field_add_impl(&rr, &rr, &rr);

    // V = U1*I
    field_mul_impl(&V, &U1, &I);

    // X3 = r^2 - J - 2*V
    field_sqr_impl(&X3, &rr);
    field_sub_impl(&X3, &X3, &J);
    field_add_impl(&t1, &V, &V);
    field_sub_impl(&X3, &X3, &t1);

    // Y3 = r*(V - X3) - 2*S1*J
    field_sub_impl(&t1, &V, &X3);
    field_mul_impl(&Y3, &rr, &t1);
    field_mul_impl(&t2, &S1, &J);
    field_add_impl(&t2, &t2, &t2);
    field_sub_impl(&Y3, &Y3, &t2);

    // Z3 = ((Z1 + Z2)^2 - Z1Z1 - Z2Z2) * H
    field_add_impl(&t1, &p->z, &q->z);
    field_sqr_impl(&t1, &t1);
    field_sub_impl(&t1, &t1, &Z1Z1);
    field_sub_impl(&t1, &t1, &Z2Z2);
    field_mul_impl(&Z3, &t1, &H);

    r->x = X3;
    r->y = Y3;
    r->z = Z3;
    r->infinity = 0;
}

// =============================================================================
// Mixed Addition: R = P + Q (Jacobian + Affine)
// More efficient when one point is affine (Z = 1)
// =============================================================================

FORCE_INLINE void point_add_mixed_impl(JacobianPoint* r, const JacobianPoint* p, const AffinePoint* q) {
    if (point_is_infinity(p)) {
        point_from_affine(r, q);
        return;
    }

    FieldElement Z1Z1, U2, S2, H, HH, I, J, rr, V, X3, Y3, Z3, t1, t2;

    // Z1Z1 = Z1^2
    field_sqr_impl(&Z1Z1, &p->z);

    // U2 = X2*Z1Z1 (U1 = X1 since Z2 = 1)
    field_mul_impl(&U2, &q->x, &Z1Z1);

    // S2 = Y2*Z1*Z1Z1 (S1 = Y1 since Z2 = 1)
    field_mul_impl(&t1, &q->y, &p->z);
    field_mul_impl(&S2, &t1, &Z1Z1);

    // H = U2 - X1
    field_sub_impl(&H, &U2, &p->x);

    // Check if points are same or negatives
    if ((H.limbs[0] | H.limbs[1] | H.limbs[2] | H.limbs[3]) == 0) {
        field_sub_impl(&t1, &S2, &p->y);
        if ((t1.limbs[0] | t1.limbs[1] | t1.limbs[2] | t1.limbs[3]) == 0) {
            point_double_impl(r, p);
            return;
        }
        point_set_infinity(r);
        return;
    }

    // HH = H^2
    field_sqr_impl(&HH, &H);

    // I = 4*HH
    field_add_impl(&I, &HH, &HH);
    field_add_impl(&I, &I, &I);

    // J = H*I
    field_mul_impl(&J, &H, &I);

    // r = 2*(S2 - Y1)
    field_sub_impl(&rr, &S2, &p->y);
    field_add_impl(&rr, &rr, &rr);

    // V = X1*I
    field_mul_impl(&V, &p->x, &I);

    // X3 = r^2 - J - 2*V
    field_sqr_impl(&X3, &rr);
    field_sub_impl(&X3, &X3, &J);
    field_add_impl(&t1, &V, &V);
    field_sub_impl(&X3, &X3, &t1);

    // Y3 = r*(V - X3) - 2*Y1*J
    field_sub_impl(&t1, &V, &X3);
    field_mul_impl(&Y3, &rr, &t1);
    field_mul_impl(&t2, &p->y, &J);
    field_add_impl(&t2, &t2, &t2);
    field_sub_impl(&Y3, &Y3, &t2);

    // Z3 = (Z1 + H)^2 - Z1Z1 - HH
    field_add_impl(&t1, &p->z, &H);
    field_sqr_impl(&Z3, &t1);
    field_sub_impl(&Z3, &Z3, &Z1Z1);
    field_sub_impl(&Z3, &Z3, &HH);

    r->x = X3;
    r->y = Y3;
    r->z = Z3;
    r->infinity = 0;
}

// Mixed Jacobian+affine addition with H output for batch inversion.
// h_out receives H = U2 - X1 (the Z-coordinate ratio).
// For degenerate cases (infinity, doubling, negation), h_out = ONE.
FORCE_INLINE void point_add_mixed_h_impl(JacobianPoint* r, const JacobianPoint* p,
                                   const AffinePoint* q, FieldElement* h_out) {
    h_out->limbs[0] = 1UL; h_out->limbs[1] = 0; h_out->limbs[2] = 0; h_out->limbs[3] = 0;

    if (point_is_infinity(p)) {
        point_from_affine(r, q);
        return;
    }

    FieldElement Z1Z1, U2, S2, H, HH, I, J, rr, V, X3, Y3, Z3, t1, t2;

    field_sqr_impl(&Z1Z1, &p->z);
    field_mul_impl(&U2, &q->x, &Z1Z1);
    field_mul_impl(&t1, &q->y, &p->z);
    field_mul_impl(&S2, &t1, &Z1Z1);

    field_sub_impl(&H, &U2, &p->x);

    if ((H.limbs[0] | H.limbs[1] | H.limbs[2] | H.limbs[3]) == 0) {
        field_sub_impl(&t1, &S2, &p->y);
        if ((t1.limbs[0] | t1.limbs[1] | t1.limbs[2] | t1.limbs[3]) == 0) {
            point_double_impl(r, p);
            return;
        }
        point_set_infinity(r);
        return;
    }

    // Z3 = (Z1+H)^2 - Z1Z1 - HH = 2*Z1*H, so Z-ratio is 2*H
    field_add_impl(h_out, &H, &H);

    field_sqr_impl(&HH, &H);
    field_add_impl(&I, &HH, &HH);
    field_add_impl(&I, &I, &I);
    field_mul_impl(&J, &H, &I);
    field_sub_impl(&rr, &S2, &p->y);
    field_add_impl(&rr, &rr, &rr);
    field_mul_impl(&V, &p->x, &I);

    field_sqr_impl(&X3, &rr);
    field_sub_impl(&X3, &X3, &J);
    field_add_impl(&t1, &V, &V);
    field_sub_impl(&X3, &X3, &t1);

    field_sub_impl(&t1, &V, &X3);
    field_mul_impl(&Y3, &rr, &t1);
    field_mul_impl(&t2, &p->y, &J);
    field_add_impl(&t2, &t2, &t2);
    field_sub_impl(&Y3, &Y3, &t2);

    field_add_impl(&t1, &p->z, &H);
    field_sqr_impl(&Z3, &t1);
    field_sub_impl(&Z3, &Z3, &Z1Z1);
    field_sub_impl(&Z3, &Z3, &HH);

    r->x = X3; r->y = Y3; r->z = Z3; r->infinity = 0;
}

// =============================================================================
// Scalar Utilities for wNAF
// =============================================================================

FORCE_INLINE int scalar_is_zero(const Scalar* k) {
    return (k->limbs[0] | k->limbs[1] | k->limbs[2] | k->limbs[3]) == 0;
}

FORCE_INLINE int scalar_bit(const Scalar* k, int pos) {
    int limb = pos / 64;
    int bit = pos % 64;
    return (int)((k->limbs[limb] >> bit) & 1UL);
}

FORCE_INLINE void scalar_sub_u64(Scalar* a, ulong val, Scalar* r) {
    *r = *a;
    ulong old = r->limbs[0];
    r->limbs[0] -= val;
    if (r->limbs[0] > old) { // borrow
        for (int i = 1; i < 4; i++) {
            r->limbs[i] -= 1;
            if (r->limbs[i] != ~0UL) break; // no further borrow
        }
    }
}

FORCE_INLINE void scalar_add_u64(Scalar* a, ulong val, Scalar* r) {
    *r = *a;
    ulong old = r->limbs[0];
    r->limbs[0] += val;
    if (r->limbs[0] < old) { // carry
        for (int i = 1; i < 4; i++) {
            r->limbs[i] += 1;
            if (r->limbs[i] != 0) break; // no further carry
        }
    }
}

// Convert scalar to wNAF representation (window width 5)
// Returns length of wNAF representation
FORCE_INLINE int scalar_to_wnaf(const Scalar* k, int wnaf[260]) {
    Scalar temp = *k;
    int len = 0;
    const int window_size = 32;   // 2^5
    const int window_mask = 31;   // 2^5 - 1
    const int window_half = 16;   // 2^(5-1)
    
    int digit;
    ulong limb;

    while (!scalar_is_zero(&temp) && len < 260) {
        if (scalar_bit(&temp, 0) == 1) { // temp is odd
            digit = (int)(temp.limbs[0] & window_mask);
            
            if (digit >= window_half) {
                digit -= window_size;
                scalar_add_u64(&temp, (ulong)(-digit), &temp);
            } else {
                scalar_sub_u64(&temp, (ulong)digit, &temp);
            }
            
            wnaf[len] = digit;
        } else {
            wnaf[len] = 0;
        }
        
        // Right shift by 1
        limb = temp.limbs[3];
        temp.limbs[3] = (limb >> 1);
        ulong carry = limb & 1;
        
        limb = temp.limbs[2];
        temp.limbs[2] = (limb >> 1) | (carry << 63);
        carry = limb & 1;
        
        limb = temp.limbs[1];
        temp.limbs[1] = (limb >> 1) | (carry << 63);
        carry = limb & 1;
        
        limb = temp.limbs[0];
        temp.limbs[0] = (limb >> 1) | (carry << 63);
        
        len++;
    }
    
    return len;
}

// Negate Y coordinate of Jacobian point
FORCE_INLINE void point_negate_y(JacobianPoint* p) {
    FieldElement zero;
    zero.limbs[0] = 0; zero.limbs[1] = 0;
    zero.limbs[2] = 0; zero.limbs[3] = 0;
    field_neg_impl(&p->y, &p->y);
}

// =============================================================================
// Scalar Multiplication: R = k * P
// wNAF (window width 5) — matches CUDA's scalar_mul
// =============================================================================

FORCE_INLINE void scalar_mul_impl(JacobianPoint* r, const Scalar* k, const AffinePoint* p) {
    // Check for zero scalar
    if (scalar_is_zero(k)) {
        point_set_infinity(r);
        return;
    }

    // Convert scalar to wNAF representation
    int wnaf[260];
    int wnaf_len = scalar_to_wnaf(k, wnaf);

    // Precompute table: [P, 3P, 5P, ..., 15P] (8 entries)
    JacobianPoint table[8];
    JacobianPoint double_p;
    
    point_from_affine(&table[0], p);
    point_double_impl(&double_p, &table[0]);
    
    for (int i = 1; i < 8; i++) {
        point_add_impl(&table[i], &table[i-1], &double_p);
    }

    // Initialize result as infinity
    point_set_infinity(r);

    int digit;
    int idx;

    // Process wNAF from MSB to LSB
    for (int i = wnaf_len - 1; i >= 0; --i) {
        point_double_impl(r, r);

        digit = wnaf[i];
        if (digit > 0) {
            idx = (digit - 1) / 2;
            point_add_impl(r, r, &table[idx]);
        } else if (digit < 0) {
            idx = (-digit - 1) / 2;
            JacobianPoint neg_point = table[idx];
            point_negate_y(&neg_point);
            point_add_impl(r, r, &neg_point);
        }
    }
}

// =============================================================================
// Scalar Multiplication with Generator: R = k * G
// Fixed-window w=4 with precomputed affine table of {0G..15G}.
// Uses mixed J+A additions and unchecked variants for maximum throughput.
// =============================================================================

FORCE_INLINE void scalar_mul_generator_impl(JacobianPoint* r, const Scalar* k) {
    // Precomputed affine table: table[i] = i*G for i = 0..15.
    // table[0] is the point at infinity (unused except as sentinel).
    AffinePoint table[16];
    table[0].x.limbs[0] = 0; table[0].x.limbs[1] = 0; table[0].x.limbs[2] = 0; table[0].x.limbs[3] = 0;
    table[0].y.limbs[0] = 0; table[0].y.limbs[1] = 0; table[0].y.limbs[2] = 0; table[0].y.limbs[3] = 0;
    // 1*G
    table[1].x.limbs[0] = 0x59F2815B16F81798UL; table[1].x.limbs[1] = 0x029BFCDB2DCE28D9UL;
    table[1].x.limbs[2] = 0x55A06295CE870B07UL; table[1].x.limbs[3] = 0x79BE667EF9DCBBACUL;
    table[1].y.limbs[0] = 0x9C47D08FFB10D4B8UL; table[1].y.limbs[1] = 0xFD17B448A6855419UL;
    table[1].y.limbs[2] = 0x5DA4FBFC0E1108A8UL; table[1].y.limbs[3] = 0x483ADA7726A3C465UL;
    // 2*G
    table[2].x.limbs[0] = 0xABAC09B95C709EE5UL; table[2].x.limbs[1] = 0x5C778E4B8CEF3CA7UL;
    table[2].x.limbs[2] = 0x3045406E95C07CD8UL; table[2].x.limbs[3] = 0xC6047F9441ED7D6DUL;
    table[2].y.limbs[0] = 0x236431A950CFE52AUL; table[2].y.limbs[1] = 0xF7F632653266D0E1UL;
    table[2].y.limbs[2] = 0xA3C58419466CEAEEUL; table[2].y.limbs[3] = 0x1AE168FEA63DC339UL;
    // 3*G
    table[3].x.limbs[0] = 0x8601F113BCE036F9UL; table[3].x.limbs[1] = 0xB531C845836F99B0UL;
    table[3].x.limbs[2] = 0x49344F85F89D5229UL; table[3].x.limbs[3] = 0xF9308A019258C310UL;
    table[3].y.limbs[0] = 0x6CB9FD7584B8E672UL; table[3].y.limbs[1] = 0x6500A99934C2231BUL;
    table[3].y.limbs[2] = 0x0FE337E62A37F356UL; table[3].y.limbs[3] = 0x388F7B0F632DE814UL;
    // 4*G
    table[4].x.limbs[0] = 0x74FA94ABE8C4CD13UL; table[4].x.limbs[1] = 0xCC6C13900EE07584UL;
    table[4].x.limbs[2] = 0x581E4904930B1404UL; table[4].x.limbs[3] = 0xE493DBF1C10D80F3UL;
    table[4].y.limbs[0] = 0xCFE97BDC47739922UL; table[4].y.limbs[1] = 0xD967AE33BFBDFE40UL;
    table[4].y.limbs[2] = 0x5642E2098EA51448UL; table[4].y.limbs[3] = 0x51ED993EA0D455B7UL;
    // 5*G
    table[5].x.limbs[0] = 0xCBA8D569B240EFE4UL; table[5].x.limbs[1] = 0xE88B84BDDC619AB7UL;
    table[5].x.limbs[2] = 0x55B4A7250A5C5128UL; table[5].x.limbs[3] = 0x2F8BDE4D1A072093UL;
    table[5].y.limbs[0] = 0xDCA87D3AA6AC62D6UL; table[5].y.limbs[1] = 0xF788271BAB0D6840UL;
    table[5].y.limbs[2] = 0xD4DBA9DDA6C9C426UL; table[5].y.limbs[3] = 0xD8AC222636E5E3D6UL;
    // 6*G
    table[6].x.limbs[0] = 0x2F057A1460297556UL; table[6].x.limbs[1] = 0x82F6472F8568A18BUL;
    table[6].x.limbs[2] = 0x20453A14355235D3UL; table[6].x.limbs[3] = 0xFFF97BD5755EEEA4UL;
    table[6].y.limbs[0] = 0x3C870C36B075F297UL; table[6].y.limbs[1] = 0xDE80F0F6518FE4A0UL;
    table[6].y.limbs[2] = 0xF3BE96017F45C560UL; table[6].y.limbs[3] = 0xAE12777AACFBB620UL;
    // 7*G
    table[7].x.limbs[0] = 0xE92BDDEDCAC4F9BCUL; table[7].x.limbs[1] = 0x3D419B7E0330E39CUL;
    table[7].x.limbs[2] = 0xA398F365F2EA7A0EUL; table[7].x.limbs[3] = 0x5CBDF0646E5DB4EAUL;
    table[7].y.limbs[0] = 0xA5082628087264DAUL; table[7].y.limbs[1] = 0xA813D0B813FDE7B5UL;
    table[7].y.limbs[2] = 0xA3178D6D861A54DBUL; table[7].y.limbs[3] = 0x6AEBCA40BA255960UL;
    // 8*G
    table[8].x.limbs[0] = 0x67784EF3E10A2A01UL; table[8].x.limbs[1] = 0x0A1BDD05E5AF888AUL;
    table[8].x.limbs[2] = 0xAFF3843FB70F3C2FUL; table[8].x.limbs[3] = 0x2F01E5E15CCA351DUL;
    table[8].y.limbs[0] = 0xB5DA2CB76CBDE904UL; table[8].y.limbs[1] = 0xC2E213D6BA5B7617UL;
    table[8].y.limbs[2] = 0x293D082A132D13B4UL; table[8].y.limbs[3] = 0x5C4DA8A741539949UL;
    // 9*G
    table[9].x.limbs[0] = 0xC35F110DFC27CCBEUL; table[9].x.limbs[1] = 0xE09796974C57E714UL;
    table[9].x.limbs[2] = 0x09AD178A9F559ABDUL; table[9].x.limbs[3] = 0xACD484E2F0C7F653UL;
    table[9].y.limbs[0] = 0x05CC262AC64F9C37UL; table[9].y.limbs[1] = 0xADD888A4375F8E0FUL;
    table[9].y.limbs[2] = 0x64380971763B61E9UL; table[9].y.limbs[3] = 0xCC338921B0A7D9FDUL;
    // 10*G
    table[10].x.limbs[0] = 0x52A68E2A47E247C7UL; table[10].x.limbs[1] = 0x3442D49B1943C2B7UL;
    table[10].x.limbs[2] = 0x35477C7B1AE6AE5DUL; table[10].x.limbs[3] = 0xA0434D9E47F3C862UL;
    table[10].y.limbs[0] = 0x3CBEE53B037368D7UL; table[10].y.limbs[1] = 0x6F794C2ED877A159UL;
    table[10].y.limbs[2] = 0xA3B6C7E693A24C69UL; table[10].y.limbs[3] = 0x893ABA425419BC27UL;
    // 11*G
    table[11].x.limbs[0] = 0xBBEC17895DA008CBUL; table[11].x.limbs[1] = 0x5649980BE5C17891UL;
    table[11].x.limbs[2] = 0x5EF4246B70C65AACUL; table[11].x.limbs[3] = 0x774AE7F858A9411EUL;
    table[11].y.limbs[0] = 0x301D74C9C953C61BUL; table[11].y.limbs[1] = 0x372DB1E2DFF9D6A8UL;
    table[11].y.limbs[2] = 0x0243DD56D7B7B365UL; table[11].y.limbs[3] = 0xD984A032EB6B5E19UL;
    // 12*G
    table[12].x.limbs[0] = 0xC5B0F47070AFE85AUL; table[12].x.limbs[1] = 0x687CF4419620095BUL;
    table[12].x.limbs[2] = 0x15C38F004D734633UL; table[12].x.limbs[3] = 0xD01115D548E7561BUL;
    table[12].y.limbs[0] = 0x6B051B13F4062327UL; table[12].y.limbs[1] = 0x79238C5DD9A86D52UL;
    table[12].y.limbs[2] = 0xA8B64537E17BD815UL; table[12].y.limbs[3] = 0xA9F34FFDC815E0D7UL;
    // 13*G
    table[13].x.limbs[0] = 0xDEEDDF8F19405AA8UL; table[13].x.limbs[1] = 0xB075FBC6610E58CDUL;
    table[13].x.limbs[2] = 0xC7D1D205C3748651UL; table[13].x.limbs[3] = 0xF28773C2D975288BUL;
    table[13].y.limbs[0] = 0x29B5CB52DB03ED81UL; table[13].y.limbs[1] = 0x3A1A06DA521FA91FUL;
    table[13].y.limbs[2] = 0x758212EB65CDAF47UL; table[13].y.limbs[3] = 0x0AB0902E8D880A89UL;
    // 14*G
    table[14].x.limbs[0] = 0xE49B241A60E823E4UL; table[14].x.limbs[1] = 0x26AA7B63678949E6UL;
    table[14].x.limbs[2] = 0xFD64E67F07D38E32UL; table[14].x.limbs[3] = 0x499FDF9E895E719CUL;
    table[14].y.limbs[0] = 0xC65F40D403A13F5BUL; table[14].y.limbs[1] = 0x464279C27A3F95BCUL;
    table[14].y.limbs[2] = 0x90F044E4A7B3D464UL; table[14].y.limbs[3] = 0xCAC2F6C4B54E8551UL;
    // 15*G
    table[15].x.limbs[0] = 0x44ADBCF8E27E080EUL; table[15].x.limbs[1] = 0x31E5946F3C85F79EUL;
    table[15].x.limbs[2] = 0x5A465AE3095FF411UL; table[15].x.limbs[3] = 0xD7924D4F7D43EA96UL;
    table[15].y.limbs[0] = 0xC504DC9FF6A26B58UL; table[15].y.limbs[1] = 0xEA40AF2BD896D3A5UL;
    table[15].y.limbs[2] = 0x83842EC228CC6DEFUL; table[15].y.limbs[3] = 0x581E2872A86C72A6UL;

    // Process scalar 4 bits at a time (MSB first)
    point_set_infinity(r);
    int started = 0;

    for (int limb = 3; limb >= 0; limb--) {
        ulong w = k->limbs[limb];
        for (int nib = 15; nib >= 0; nib--) {
            uint idx = (uint)((w >> (nib * 4)) & 0xFUL);

            if (started) {
                point_double_unchecked(r, r);
                point_double_unchecked(r, r);
                point_double_unchecked(r, r);
                point_double_unchecked(r, r);
            }

            if (idx != 0) {
                if (!started) {
                    point_from_affine(r, &table[idx]);
                    started = 1;
                } else {
                    point_add_mixed_unchecked(r, r, &table[idx]);
                }
            }
        }
    }
}

// =============================================================================
// OpenCL Kernels - Point Operations
// =============================================================================

extern "C" __global__ void point_double(
    const JacobianPoint* points,
    JacobianPoint* results,
    const uint count
) {
    uint gid = ((uint)(blockIdx.x * blockDim.x + threadIdx.x));
    if (gid >= count) return;

    // Copy from global to private memory
    JacobianPoint p_local = points[gid];
    JacobianPoint r;
    point_double_impl(&r, &p_local);
    results[gid] = r;
}

extern "C" __global__ void point_add(
    const JacobianPoint* p,
    const JacobianPoint* q,
    JacobianPoint* results,
    const uint count
) {
    uint gid = ((uint)(blockIdx.x * blockDim.x + threadIdx.x));
    if (gid >= count) return;

    // Copy from global to private memory
    JacobianPoint p_local = p[gid];
    JacobianPoint q_local = q[gid];
    JacobianPoint r;
    point_add_impl(&r, &p_local, &q_local);
    results[gid] = r;
}

extern "C" __global__ void scalar_mul(
    const Scalar* scalars,
    const AffinePoint* points,
    JacobianPoint* results,
    const uint count
) {
    uint gid = ((uint)(blockIdx.x * blockDim.x + threadIdx.x));
    if (gid >= count) return;

    // Copy from global to private memory
    Scalar k_local = scalars[gid];
    AffinePoint p_local = points[gid];
    JacobianPoint r;
    scalar_mul_impl(&r, &k_local, &p_local);
    results[gid] = r;
}

extern "C" __global__ void scalar_mul_generator(
    const Scalar* scalars,
    JacobianPoint* results,
    const uint count
) {
    uint gid = ((uint)(blockIdx.x * blockDim.x + threadIdx.x));
    if (gid >= count) return;

    // Copy from global to private memory
    Scalar k_local = scalars[gid];
    JacobianPoint r;
    scalar_mul_generator_impl(&r, &k_local);
    results[gid] = r;
}


// =============================================================================
// BTCW OpenCL Mining Kernel
// =============================================================================
// Copyright (c) 2026 btcw.space <btcw.space@proton.me>
//
// Implements the Bitcoin-PoW Stage 2 mining algorithm:
//   1. Pick nonce from upper 64-bit space (GPU partitioned)
//   2. mud = hash_no_sig + nonce  (uint256 LE addition)
//   3. ECDSA sign(seckey, mud_LE_bytes): RFC 6979 nonce k, R = k*G, DER encode
//   4. preimage = nonce(8 LE) || CompactSize(sig_len)(1) || DER_sig(N)
//   5. hashPoW = double-SHA256(preimage)
//   6. Check trailing bytes [28-31] for difficulty (reversed byte order)
//
// Copyright (c) 2026 btcw.space. All rights reserved.
// =============================================================================

// =============================================================================
// secp256k1 OpenCL primitives (field + point operations)
// These files are concatenated during kernel build.
// =============================================================================

// --- secp256k1_field.cl inlined (field arithmetic) ---
// We rely on the build system to prepend secp256k1_field.cl and secp256k1_point.cl
// via clCreateProgramWithSource with multiple source strings.
// The field and point types/functions are thus available here.

// =============================================================================
// SHA-256 Implementation (for GPU)
// =============================================================================

typedef struct {
    uint state[8];
    uchar buf[64];
    uint bytes;
} SHA256_CTX;

__device__ __constant__ uint SHA256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

#define SHA_Ch(x,y,z)    ((z) ^ ((x) & ((y) ^ (z))))
#define SHA_Maj(x,y,z)   (((x) & (y)) | ((z) & ((x) | (y))))
#define SHA_Sigma0(x)    (rotate((x), 30U) ^ rotate((x), 19U) ^ rotate((x), 10U))
#define SHA_Sigma1(x)    (rotate((x), 26U) ^ rotate((x), 21U) ^ rotate((x), 7U))
#define SHA_sigma0(x)    (rotate((x), 25U) ^ rotate((x), 14U) ^ ((x) >> 3))
#define SHA_sigma1(x)    (rotate((x), 15U) ^ rotate((x), 13U) ^ ((x) >> 10))

FORCE_INLINE uint read_be32(const uchar* p) {
    return ((uint)p[0] << 24) | ((uint)p[1] << 16) | ((uint)p[2] << 8) | (uint)p[3];
}

FORCE_INLINE void write_be32(uchar* p, uint v) {
    p[0] = (uchar)(v >> 24);
    p[1] = (uchar)(v >> 16);
    p[2] = (uchar)(v >> 8);
    p[3] = (uchar)(v);
}

FORCE_INLINE void sha256_transform(uint* s, const uchar* buf) {
    uint a = s[0], b = s[1], c = s[2], d = s[3];
    uint e = s[4], f = s[5], g = s[6], h = s[7];
    uint w[64];

    #pragma unroll
    for (int i = 0; i < 16; i++)
        w[i] = read_be32(&buf[i * 4]);

    #pragma unroll
    for (int i = 16; i < 64; i++)
        w[i] = SHA_sigma1(w[i-2]) + w[i-7] + SHA_sigma0(w[i-15]) + w[i-16];

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint t1 = h + SHA_Sigma1(e) + SHA_Ch(e, f, g) + SHA256_K[i] + w[i];
        uint t2 = SHA_Sigma0(a) + SHA_Maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    s[0] += a; s[1] += b; s[2] += c; s[3] += d;
    s[4] += e; s[5] += f; s[6] += g; s[7] += h;
}

FORCE_INLINE void sha256_init(SHA256_CTX* ctx) {
    ctx->state[0] = 0x6a09e667;
    ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372;
    ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f;
    ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab;
    ctx->state[7] = 0x5be0cd19;
    ctx->bytes = 0;
}

FORCE_INLINE void sha256_update(SHA256_CTX* ctx, const uchar* data, uint len) {
    uint bufsize = ctx->bytes & 0x3F;
    ctx->bytes += len;
    while (len >= 64 - bufsize) {
        uint chunk = 64 - bufsize;
        for (uint i = 0; i < chunk; i++)
            ctx->buf[bufsize + i] = data[i];
        data += chunk;
        len -= chunk;
        sha256_transform(ctx->state, ctx->buf);
        bufsize = 0;
    }
    for (uint i = 0; i < len; i++)
        ctx->buf[bufsize + i] = data[i];
}

FORCE_INLINE void sha256_final(SHA256_CTX* ctx, uchar* out32) {
    uint bufsize = ctx->bytes & 0x3F;
    // Pad
    ctx->buf[bufsize++] = 0x80;
    if (bufsize > 56) {
        for (uint i = bufsize; i < 64; i++) ctx->buf[i] = 0;
        sha256_transform(ctx->state, ctx->buf);
        bufsize = 0;
    }
    for (uint i = bufsize; i < 56; i++) ctx->buf[i] = 0;
    // Length in bits (big-endian)
    ulong bitlen = (ulong)ctx->bytes * 8;
    write_be32(&ctx->buf[56], (uint)(bitlen >> 32));
    write_be32(&ctx->buf[60], (uint)(bitlen));
    sha256_transform(ctx->state, ctx->buf);
    for (int i = 0; i < 8; i++)
        write_be32(&out32[i * 4], ctx->state[i]);
}

// Double SHA-256
FORCE_INLINE void double_sha256(const uchar* data, uint len, uchar* out32) {
    SHA256_CTX ctx;
    uchar mid[32];
    sha256_init(&ctx);
    sha256_update(&ctx, data, len);
    sha256_final(&ctx, mid);
    sha256_init(&ctx);
    sha256_update(&ctx, mid, 32);
    sha256_final(&ctx, out32);
}

// =============================================================================
// HMAC-SHA256
// =============================================================================

typedef struct {
    SHA256_CTX inner;
    SHA256_CTX outer;
} HMAC_SHA256_CTX;

FORCE_INLINE void hmac_sha256_init(HMAC_SHA256_CTX* hmac, const uchar* key, uint keylen) {
    uchar rkey[64];
    for (int i = 0; i < 64; i++) rkey[i] = 0;

    if (keylen <= 64) {
        for (uint i = 0; i < keylen; i++) rkey[i] = key[i];
    } else {
        SHA256_CTX tmp;
        sha256_init(&tmp);
        sha256_update(&tmp, key, keylen);
        sha256_final(&tmp, rkey);
    }

    uchar opad[64], ipad[64];
    for (int i = 0; i < 64; i++) {
        opad[i] = rkey[i] ^ 0x5c;
        ipad[i] = rkey[i] ^ 0x36;
    }

    sha256_init(&hmac->outer);
    sha256_update(&hmac->outer, opad, 64);

    sha256_init(&hmac->inner);
    sha256_update(&hmac->inner, ipad, 64);
}

FORCE_INLINE void hmac_sha256_update(HMAC_SHA256_CTX* hmac, const uchar* data, uint len) {
    sha256_update(&hmac->inner, data, len);
}

FORCE_INLINE void hmac_sha256_final(HMAC_SHA256_CTX* hmac, uchar* out32) {
    uchar tmp[32];
    sha256_final(&hmac->inner, tmp);
    sha256_update(&hmac->outer, tmp, 32);
    sha256_final(&hmac->outer, out32);
}

// =============================================================================
// RFC 6979 Deterministic Nonce Generation
// =============================================================================
// Input: secret_key (32 bytes), message (32 bytes)
// Output: nonce32 (32 bytes)
// Follows the procedure from Bitcoin's libsecp256k1

// Forward declarations (defined later, needed for message mod n reduction)
FORCE_INLINE void scalar_set_b32(Scalar* s, const uchar* b32);
FORCE_INLINE void scalar_get_b32(uchar* b32, const Scalar* s);
FORCE_INLINE void scalar_reduce(Scalar* s);

FORCE_INLINE void rfc6979_generate_k(const uchar* seckey32, const uchar* msg32, uchar* nonce32) {
    uchar v[32], k[32];
    uchar keydata[64]; // seckey || msg_mod_n
    HMAC_SHA256_CTX hmac;

    // Copy seckey into keydata
    for (int i = 0; i < 32; i++) keydata[i] = seckey32[i];

    // Reduce message mod n before using as HMAC-DRBG input
    // (matches CUDA's nonce_function_rfc6979: scalar_set_b32 + scalar_get_b32)
    Scalar msg_tmp;
    scalar_set_b32(&msg_tmp, msg32);
    scalar_reduce(&msg_tmp);
    uchar msgmod32[32];
    scalar_get_b32(msgmod32, &msg_tmp);
    for (int i = 0; i < 32; i++) keydata[32 + i] = msgmod32[i];

    // Initialize: V = 0x01...01, K = 0x00...00  (RFC 6979 3.2.b, 3.2.c)
    for (int i = 0; i < 32; i++) { v[i] = 0x01; k[i] = 0x00; }

    // Step d: K = HMAC_K(V || 0x00 || keydata)
    hmac_sha256_init(&hmac, k, 32);
    hmac_sha256_update(&hmac, v, 32);
    uchar zero = 0x00;
    hmac_sha256_update(&hmac, &zero, 1);
    hmac_sha256_update(&hmac, keydata, 64);
    hmac_sha256_final(&hmac, k);

    // V = HMAC_K(V)
    hmac_sha256_init(&hmac, k, 32);
    hmac_sha256_update(&hmac, v, 32);
    hmac_sha256_final(&hmac, v);

    // Step f: K = HMAC_K(V || 0x01 || keydata)
    hmac_sha256_init(&hmac, k, 32);
    hmac_sha256_update(&hmac, v, 32);
    uchar one = 0x01;
    hmac_sha256_update(&hmac, &one, 1);
    hmac_sha256_update(&hmac, keydata, 64);
    hmac_sha256_final(&hmac, k);

    // V = HMAC_K(V)
    hmac_sha256_init(&hmac, k, 32);
    hmac_sha256_update(&hmac, v, 32);
    hmac_sha256_final(&hmac, v);

    // Step h: generate candidate
    // V = HMAC_K(V)
    hmac_sha256_init(&hmac, k, 32);
    hmac_sha256_update(&hmac, v, 32);
    hmac_sha256_final(&hmac, v);

    // Output the first valid nonce candidate
    for (int i = 0; i < 32; i++) nonce32[i] = v[i];
}



// =============================================================================
// Fixed-layout SHA/HMAC helpers for RFC6979 mining path
// =============================================================================

FORCE_INLINE void sha256_state_init_words_rfc(uint s[8]) {
    s[0]=0x6a09e667U; s[1]=0xbb67ae85U; s[2]=0x3c6ef372U; s[3]=0xa54ff53aU;
    s[4]=0x510e527fU; s[5]=0x9b05688cU; s[6]=0x1f83d9abU; s[7]=0x5be0cd19U;
}

FORCE_INLINE void sha256_state_to_bytes_rfc(const uint s[8], uchar out32[32]) {
#pragma unroll
    for (int i=0;i<8;i++) write_be32(out32 + 4*i, s[i]);
}



/* Faster fixed HMAC(K, 32-byte message) using direct block construction.
   The inner/outer key states are already precomputed. */

FORCE_INLINE void sha256_transform_words(uint* s, const uint w0[16])
{
    uint a=s[0], b=s[1], c=s[2], d=s[3];
    uint e=s[4], f=s[5], g=s[6], h=s[7];
    /* v40: 16-word circular SHA-256 schedule.  This is mathematically
       identical to the 64-word expansion but keeps only the last 16 words
       live, reducing private/register pressure in RFC6979's fixed HMACs. */
    uint w[16];

#pragma unroll
    for (int i=0;i<16;i++) w[i]=w0[i];

#pragma unroll
    for (int i=0;i<64;i++) {
        uint wi;
        if (i < 16) {
            wi = w[i];
        } else {
            const int j = i & 15;
            wi = SHA_sigma1(w[(i-2) & 15]) + w[(i-7) & 15]
               + SHA_sigma0(w[(i-15) & 15]) + w[j];
            w[j] = wi;
        }
        uint t1 = h + SHA_Sigma1(e) + SHA_Ch(e,f,g) + SHA256_K[i] + wi;
        uint t2 = SHA_Sigma0(a) + SHA_Maj(a,b,c);
        h=g; g=f; f=e; e=d+t1;
        d=c; c=b; b=a; a=t1+t2;
    }

    s[0]+=a; s[1]+=b; s[2]+=c; s[3]+=d;
    s[4]+=e; s[5]+=f; s[6]+=g; s[7]+=h;
}

// v199: RFC-only Ada PTX primitives layered on the v196 unrolled core.
// IMPORTANT: these helpers are used only by RFC6979 compression; generic
// signer/DER/PoW SHA remains exactly on the v192/v197 path.
FORCE_INLINE uint rfc_rotr32_ptx(uint x, uint n) {
    uint r;
    asm("shf.r.wrap.b32 %0, %1, %1, %2;" : "=r"(r) : "r"(x), "r"(n));
    return r;
}
FORCE_INLINE uint rfc_ch_ptx(uint x, uint y, uint z) {
    uint r;
    asm("lop3.b32 %0, %1, %2, %3, 0xca;" : "=r"(r) : "r"(x), "r"(y), "r"(z));
    return r;
}
FORCE_INLINE uint rfc_maj_ptx(uint x, uint y, uint z) {
    uint r;
    asm("lop3.b32 %0, %1, %2, %3, 0xe8;" : "=r"(r) : "r"(x), "r"(y), "r"(z));
    return r;
}
FORCE_INLINE uint rfc_Sigma0_ptx(uint x) {
    return rfc_rotr32_ptx(x,2U) ^ rfc_rotr32_ptx(x,13U) ^ rfc_rotr32_ptx(x,22U);
}
FORCE_INLINE uint rfc_Sigma1_ptx(uint x) {
    return rfc_rotr32_ptx(x,6U) ^ rfc_rotr32_ptx(x,11U) ^ rfc_rotr32_ptx(x,25U);
}
FORCE_INLINE uint rfc_sigma0_ptx(uint x) {
    return rfc_rotr32_ptx(x,7U) ^ rfc_rotr32_ptx(x,18U) ^ (x >> 3);
}
FORCE_INLINE uint rfc_sigma1_ptx(uint x) {
    return rfc_rotr32_ptx(x,17U) ^ rfc_rotr32_ptx(x,19U) ^ (x >> 10);
}

#define RFC_SHA_RND(A,B,C,D,E,F,G,H,W,K) do { \
    const uint _t1 = (H) + rfc_Sigma1_ptx(E) + rfc_ch_ptx((E),(F),(G)) + (K) + (W); \
    const uint _t2 = rfc_Sigma0_ptx(A) + rfc_maj_ptx((A),(B),(C)); \
    (D) += _t1; \
    (H) = _t1 + _t2; \
} while (0)

FORCE_INLINE void sha256_transform_words_rfc_unrolled(uint* s, const uint inw[16])
{
    uint a=s[0], b=s[1], c=s[2], d=s[3];
    uint e=s[4], f=s[5], g=s[6], h=s[7];
    uint w0=inw[0];
    uint w1=inw[1];
    uint w2=inw[2];
    uint w3=inw[3];
    uint w4=inw[4];
    uint w5=inw[5];
    uint w6=inw[6];
    uint w7=inw[7];
    uint w8=inw[8];
    uint w9=inw[9];
    uint w10=inw[10];
    uint w11=inw[11];
    uint w12=inw[12];
    uint w13=inw[13];
    uint w14=inw[14];
    uint w15=inw[15];
    RFC_SHA_RND(a,b,c,d,e,f,g,h,w0,0x428a2f98);
    RFC_SHA_RND(h,a,b,c,d,e,f,g,w1,0x71374491);
    RFC_SHA_RND(g,h,a,b,c,d,e,f,w2,0xb5c0fbcf);
    RFC_SHA_RND(f,g,h,a,b,c,d,e,w3,0xe9b5dba5);
    RFC_SHA_RND(e,f,g,h,a,b,c,d,w4,0x3956c25b);
    RFC_SHA_RND(d,e,f,g,h,a,b,c,w5,0x59f111f1);
    RFC_SHA_RND(c,d,e,f,g,h,a,b,w6,0x923f82a4);
    RFC_SHA_RND(b,c,d,e,f,g,h,a,w7,0xab1c5ed5);
    RFC_SHA_RND(a,b,c,d,e,f,g,h,w8,0xd807aa98);
    RFC_SHA_RND(h,a,b,c,d,e,f,g,w9,0x12835b01);
    RFC_SHA_RND(g,h,a,b,c,d,e,f,w10,0x243185be);
    RFC_SHA_RND(f,g,h,a,b,c,d,e,w11,0x550c7dc3);
    RFC_SHA_RND(e,f,g,h,a,b,c,d,w12,0x72be5d74);
    RFC_SHA_RND(d,e,f,g,h,a,b,c,w13,0x80deb1fe);
    RFC_SHA_RND(c,d,e,f,g,h,a,b,w14,0x9bdc06a7);
    RFC_SHA_RND(b,c,d,e,f,g,h,a,w15,0xc19bf174);
    w0 = rfc_sigma1_ptx(w14) + w9 + rfc_sigma0_ptx(w1) + w0;
    RFC_SHA_RND(a,b,c,d,e,f,g,h,w0,0xe49b69c1);
    w1 = rfc_sigma1_ptx(w15) + w10 + rfc_sigma0_ptx(w2) + w1;
    RFC_SHA_RND(h,a,b,c,d,e,f,g,w1,0xefbe4786);
    w2 = rfc_sigma1_ptx(w0) + w11 + rfc_sigma0_ptx(w3) + w2;
    RFC_SHA_RND(g,h,a,b,c,d,e,f,w2,0x0fc19dc6);
    w3 = rfc_sigma1_ptx(w1) + w12 + rfc_sigma0_ptx(w4) + w3;
    RFC_SHA_RND(f,g,h,a,b,c,d,e,w3,0x240ca1cc);
    w4 = rfc_sigma1_ptx(w2) + w13 + rfc_sigma0_ptx(w5) + w4;
    RFC_SHA_RND(e,f,g,h,a,b,c,d,w4,0x2de92c6f);
    w5 = rfc_sigma1_ptx(w3) + w14 + rfc_sigma0_ptx(w6) + w5;
    RFC_SHA_RND(d,e,f,g,h,a,b,c,w5,0x4a7484aa);
    w6 = rfc_sigma1_ptx(w4) + w15 + rfc_sigma0_ptx(w7) + w6;
    RFC_SHA_RND(c,d,e,f,g,h,a,b,w6,0x5cb0a9dc);
    w7 = rfc_sigma1_ptx(w5) + w0 + rfc_sigma0_ptx(w8) + w7;
    RFC_SHA_RND(b,c,d,e,f,g,h,a,w7,0x76f988da);
    w8 = rfc_sigma1_ptx(w6) + w1 + rfc_sigma0_ptx(w9) + w8;
    RFC_SHA_RND(a,b,c,d,e,f,g,h,w8,0x983e5152);
    w9 = rfc_sigma1_ptx(w7) + w2 + rfc_sigma0_ptx(w10) + w9;
    RFC_SHA_RND(h,a,b,c,d,e,f,g,w9,0xa831c66d);
    w10 = rfc_sigma1_ptx(w8) + w3 + rfc_sigma0_ptx(w11) + w10;
    RFC_SHA_RND(g,h,a,b,c,d,e,f,w10,0xb00327c8);
    w11 = rfc_sigma1_ptx(w9) + w4 + rfc_sigma0_ptx(w12) + w11;
    RFC_SHA_RND(f,g,h,a,b,c,d,e,w11,0xbf597fc7);
    w12 = rfc_sigma1_ptx(w10) + w5 + rfc_sigma0_ptx(w13) + w12;
    RFC_SHA_RND(e,f,g,h,a,b,c,d,w12,0xc6e00bf3);
    w13 = rfc_sigma1_ptx(w11) + w6 + rfc_sigma0_ptx(w14) + w13;
    RFC_SHA_RND(d,e,f,g,h,a,b,c,w13,0xd5a79147);
    w14 = rfc_sigma1_ptx(w12) + w7 + rfc_sigma0_ptx(w15) + w14;
    RFC_SHA_RND(c,d,e,f,g,h,a,b,w14,0x06ca6351);
    w15 = rfc_sigma1_ptx(w13) + w8 + rfc_sigma0_ptx(w0) + w15;
    RFC_SHA_RND(b,c,d,e,f,g,h,a,w15,0x14292967);
    w0 = rfc_sigma1_ptx(w14) + w9 + rfc_sigma0_ptx(w1) + w0;
    RFC_SHA_RND(a,b,c,d,e,f,g,h,w0,0x27b70a85);
    w1 = rfc_sigma1_ptx(w15) + w10 + rfc_sigma0_ptx(w2) + w1;
    RFC_SHA_RND(h,a,b,c,d,e,f,g,w1,0x2e1b2138);
    w2 = rfc_sigma1_ptx(w0) + w11 + rfc_sigma0_ptx(w3) + w2;
    RFC_SHA_RND(g,h,a,b,c,d,e,f,w2,0x4d2c6dfc);
    w3 = rfc_sigma1_ptx(w1) + w12 + rfc_sigma0_ptx(w4) + w3;
    RFC_SHA_RND(f,g,h,a,b,c,d,e,w3,0x53380d13);
    w4 = rfc_sigma1_ptx(w2) + w13 + rfc_sigma0_ptx(w5) + w4;
    RFC_SHA_RND(e,f,g,h,a,b,c,d,w4,0x650a7354);
    w5 = rfc_sigma1_ptx(w3) + w14 + rfc_sigma0_ptx(w6) + w5;
    RFC_SHA_RND(d,e,f,g,h,a,b,c,w5,0x766a0abb);
    w6 = rfc_sigma1_ptx(w4) + w15 + rfc_sigma0_ptx(w7) + w6;
    RFC_SHA_RND(c,d,e,f,g,h,a,b,w6,0x81c2c92e);
    w7 = rfc_sigma1_ptx(w5) + w0 + rfc_sigma0_ptx(w8) + w7;
    RFC_SHA_RND(b,c,d,e,f,g,h,a,w7,0x92722c85);
    w8 = rfc_sigma1_ptx(w6) + w1 + rfc_sigma0_ptx(w9) + w8;
    RFC_SHA_RND(a,b,c,d,e,f,g,h,w8,0xa2bfe8a1);
    w9 = rfc_sigma1_ptx(w7) + w2 + rfc_sigma0_ptx(w10) + w9;
    RFC_SHA_RND(h,a,b,c,d,e,f,g,w9,0xa81a664b);
    w10 = rfc_sigma1_ptx(w8) + w3 + rfc_sigma0_ptx(w11) + w10;
    RFC_SHA_RND(g,h,a,b,c,d,e,f,w10,0xc24b8b70);
    w11 = rfc_sigma1_ptx(w9) + w4 + rfc_sigma0_ptx(w12) + w11;
    RFC_SHA_RND(f,g,h,a,b,c,d,e,w11,0xc76c51a3);
    w12 = rfc_sigma1_ptx(w10) + w5 + rfc_sigma0_ptx(w13) + w12;
    RFC_SHA_RND(e,f,g,h,a,b,c,d,w12,0xd192e819);
    w13 = rfc_sigma1_ptx(w11) + w6 + rfc_sigma0_ptx(w14) + w13;
    RFC_SHA_RND(d,e,f,g,h,a,b,c,w13,0xd6990624);
    w14 = rfc_sigma1_ptx(w12) + w7 + rfc_sigma0_ptx(w15) + w14;
    RFC_SHA_RND(c,d,e,f,g,h,a,b,w14,0xf40e3585);
    w15 = rfc_sigma1_ptx(w13) + w8 + rfc_sigma0_ptx(w0) + w15;
    RFC_SHA_RND(b,c,d,e,f,g,h,a,w15,0x106aa070);
    w0 = rfc_sigma1_ptx(w14) + w9 + rfc_sigma0_ptx(w1) + w0;
    RFC_SHA_RND(a,b,c,d,e,f,g,h,w0,0x19a4c116);
    w1 = rfc_sigma1_ptx(w15) + w10 + rfc_sigma0_ptx(w2) + w1;
    RFC_SHA_RND(h,a,b,c,d,e,f,g,w1,0x1e376c08);
    w2 = rfc_sigma1_ptx(w0) + w11 + rfc_sigma0_ptx(w3) + w2;
    RFC_SHA_RND(g,h,a,b,c,d,e,f,w2,0x2748774c);
    w3 = rfc_sigma1_ptx(w1) + w12 + rfc_sigma0_ptx(w4) + w3;
    RFC_SHA_RND(f,g,h,a,b,c,d,e,w3,0x34b0bcb5);
    w4 = rfc_sigma1_ptx(w2) + w13 + rfc_sigma0_ptx(w5) + w4;
    RFC_SHA_RND(e,f,g,h,a,b,c,d,w4,0x391c0cb3);
    w5 = rfc_sigma1_ptx(w3) + w14 + rfc_sigma0_ptx(w6) + w5;
    RFC_SHA_RND(d,e,f,g,h,a,b,c,w5,0x4ed8aa4a);
    w6 = rfc_sigma1_ptx(w4) + w15 + rfc_sigma0_ptx(w7) + w6;
    RFC_SHA_RND(c,d,e,f,g,h,a,b,w6,0x5b9cca4f);
    w7 = rfc_sigma1_ptx(w5) + w0 + rfc_sigma0_ptx(w8) + w7;
    RFC_SHA_RND(b,c,d,e,f,g,h,a,w7,0x682e6ff3);
    w8 = rfc_sigma1_ptx(w6) + w1 + rfc_sigma0_ptx(w9) + w8;
    RFC_SHA_RND(a,b,c,d,e,f,g,h,w8,0x748f82ee);
    w9 = rfc_sigma1_ptx(w7) + w2 + rfc_sigma0_ptx(w10) + w9;
    RFC_SHA_RND(h,a,b,c,d,e,f,g,w9,0x78a5636f);
    w10 = rfc_sigma1_ptx(w8) + w3 + rfc_sigma0_ptx(w11) + w10;
    RFC_SHA_RND(g,h,a,b,c,d,e,f,w10,0x84c87814);
    w11 = rfc_sigma1_ptx(w9) + w4 + rfc_sigma0_ptx(w12) + w11;
    RFC_SHA_RND(f,g,h,a,b,c,d,e,w11,0x8cc70208);
    w12 = rfc_sigma1_ptx(w10) + w5 + rfc_sigma0_ptx(w13) + w12;
    RFC_SHA_RND(e,f,g,h,a,b,c,d,w12,0x90befffa);
    w13 = rfc_sigma1_ptx(w11) + w6 + rfc_sigma0_ptx(w14) + w13;
    RFC_SHA_RND(d,e,f,g,h,a,b,c,w13,0xa4506ceb);
    w14 = rfc_sigma1_ptx(w12) + w7 + rfc_sigma0_ptx(w15) + w14;
    RFC_SHA_RND(c,d,e,f,g,h,a,b,w14,0xbef9a3f7);
    w15 = rfc_sigma1_ptx(w13) + w8 + rfc_sigma0_ptx(w0) + w15;
    RFC_SHA_RND(b,c,d,e,f,g,h,a,w15,0xc67178f2);

    s[0]+=a; s[1]+=b; s[2]+=c; s[3]+=d;
    s[4]+=e; s[5]+=f; s[6]+=g; s[7]+=h;
}
// v197: RFC 32-byte continuation specialization.
// Initial W8..W15 are compile-time padding/length constants; only W0..W7 are dynamic.
FORCE_INLINE void sha256_transform_words_rfc_pad32_unrolled(uint* s, const uint in8[8])
{
    uint a=s[0], b=s[1], c=s[2], d=s[3];
    uint e=s[4], f=s[5], g=s[6], h=s[7];
    uint w0=in8[0];
    uint w1=in8[1];
    uint w2=in8[2];
    uint w3=in8[3];
    uint w4=in8[4];
    uint w5=in8[5];
    uint w6=in8[6];
    uint w7=in8[7];
    uint w8=0x80000000U;
    uint w9=0U;
    uint w10=0U;
    uint w11=0U;
    uint w12=0U;
    uint w13=0U;
    uint w14=0U;
    uint w15=768U;
    RFC_SHA_RND(a,b,c,d,e,f,g,h,w0,0x428a2f98);
    RFC_SHA_RND(h,a,b,c,d,e,f,g,w1,0x71374491);
    RFC_SHA_RND(g,h,a,b,c,d,e,f,w2,0xb5c0fbcf);
    RFC_SHA_RND(f,g,h,a,b,c,d,e,w3,0xe9b5dba5);
    RFC_SHA_RND(e,f,g,h,a,b,c,d,w4,0x3956c25b);
    RFC_SHA_RND(d,e,f,g,h,a,b,c,w5,0x59f111f1);
    RFC_SHA_RND(c,d,e,f,g,h,a,b,w6,0x923f82a4);
    RFC_SHA_RND(b,c,d,e,f,g,h,a,w7,0xab1c5ed5);
    RFC_SHA_RND(a,b,c,d,e,f,g,h,w8,0xd807aa98);
    RFC_SHA_RND(h,a,b,c,d,e,f,g,w9,0x12835b01);
    RFC_SHA_RND(g,h,a,b,c,d,e,f,w10,0x243185be);
    RFC_SHA_RND(f,g,h,a,b,c,d,e,w11,0x550c7dc3);
    RFC_SHA_RND(e,f,g,h,a,b,c,d,w12,0x72be5d74);
    RFC_SHA_RND(d,e,f,g,h,a,b,c,w13,0x80deb1fe);
    RFC_SHA_RND(c,d,e,f,g,h,a,b,w14,0x9bdc06a7);
    RFC_SHA_RND(b,c,d,e,f,g,h,a,w15,0xc19bf174);
    w0 = rfc_sigma1_ptx(w14) + w9 + rfc_sigma0_ptx(w1) + w0;
    RFC_SHA_RND(a,b,c,d,e,f,g,h,w0,0xe49b69c1);
    w1 = rfc_sigma1_ptx(w15) + w10 + rfc_sigma0_ptx(w2) + w1;
    RFC_SHA_RND(h,a,b,c,d,e,f,g,w1,0xefbe4786);
    w2 = rfc_sigma1_ptx(w0) + w11 + rfc_sigma0_ptx(w3) + w2;
    RFC_SHA_RND(g,h,a,b,c,d,e,f,w2,0x0fc19dc6);
    w3 = rfc_sigma1_ptx(w1) + w12 + rfc_sigma0_ptx(w4) + w3;
    RFC_SHA_RND(f,g,h,a,b,c,d,e,w3,0x240ca1cc);
    w4 = rfc_sigma1_ptx(w2) + w13 + rfc_sigma0_ptx(w5) + w4;
    RFC_SHA_RND(e,f,g,h,a,b,c,d,w4,0x2de92c6f);
    w5 = rfc_sigma1_ptx(w3) + w14 + rfc_sigma0_ptx(w6) + w5;
    RFC_SHA_RND(d,e,f,g,h,a,b,c,w5,0x4a7484aa);
    w6 = rfc_sigma1_ptx(w4) + w15 + rfc_sigma0_ptx(w7) + w6;
    RFC_SHA_RND(c,d,e,f,g,h,a,b,w6,0x5cb0a9dc);
    w7 = rfc_sigma1_ptx(w5) + w0 + rfc_sigma0_ptx(w8) + w7;
    RFC_SHA_RND(b,c,d,e,f,g,h,a,w7,0x76f988da);
    w8 = rfc_sigma1_ptx(w6) + w1 + rfc_sigma0_ptx(w9) + w8;
    RFC_SHA_RND(a,b,c,d,e,f,g,h,w8,0x983e5152);
    w9 = rfc_sigma1_ptx(w7) + w2 + rfc_sigma0_ptx(w10) + w9;
    RFC_SHA_RND(h,a,b,c,d,e,f,g,w9,0xa831c66d);
    w10 = rfc_sigma1_ptx(w8) + w3 + rfc_sigma0_ptx(w11) + w10;
    RFC_SHA_RND(g,h,a,b,c,d,e,f,w10,0xb00327c8);
    w11 = rfc_sigma1_ptx(w9) + w4 + rfc_sigma0_ptx(w12) + w11;
    RFC_SHA_RND(f,g,h,a,b,c,d,e,w11,0xbf597fc7);
    w12 = rfc_sigma1_ptx(w10) + w5 + rfc_sigma0_ptx(w13) + w12;
    RFC_SHA_RND(e,f,g,h,a,b,c,d,w12,0xc6e00bf3);
    w13 = rfc_sigma1_ptx(w11) + w6 + rfc_sigma0_ptx(w14) + w13;
    RFC_SHA_RND(d,e,f,g,h,a,b,c,w13,0xd5a79147);
    w14 = rfc_sigma1_ptx(w12) + w7 + rfc_sigma0_ptx(w15) + w14;
    RFC_SHA_RND(c,d,e,f,g,h,a,b,w14,0x06ca6351);
    w15 = rfc_sigma1_ptx(w13) + w8 + rfc_sigma0_ptx(w0) + w15;
    RFC_SHA_RND(b,c,d,e,f,g,h,a,w15,0x14292967);
    w0 = rfc_sigma1_ptx(w14) + w9 + rfc_sigma0_ptx(w1) + w0;
    RFC_SHA_RND(a,b,c,d,e,f,g,h,w0,0x27b70a85);
    w1 = rfc_sigma1_ptx(w15) + w10 + rfc_sigma0_ptx(w2) + w1;
    RFC_SHA_RND(h,a,b,c,d,e,f,g,w1,0x2e1b2138);
    w2 = rfc_sigma1_ptx(w0) + w11 + rfc_sigma0_ptx(w3) + w2;
    RFC_SHA_RND(g,h,a,b,c,d,e,f,w2,0x4d2c6dfc);
    w3 = rfc_sigma1_ptx(w1) + w12 + rfc_sigma0_ptx(w4) + w3;
    RFC_SHA_RND(f,g,h,a,b,c,d,e,w3,0x53380d13);
    w4 = rfc_sigma1_ptx(w2) + w13 + rfc_sigma0_ptx(w5) + w4;
    RFC_SHA_RND(e,f,g,h,a,b,c,d,w4,0x650a7354);
    w5 = rfc_sigma1_ptx(w3) + w14 + rfc_sigma0_ptx(w6) + w5;
    RFC_SHA_RND(d,e,f,g,h,a,b,c,w5,0x766a0abb);
    w6 = rfc_sigma1_ptx(w4) + w15 + rfc_sigma0_ptx(w7) + w6;
    RFC_SHA_RND(c,d,e,f,g,h,a,b,w6,0x81c2c92e);
    w7 = rfc_sigma1_ptx(w5) + w0 + rfc_sigma0_ptx(w8) + w7;
    RFC_SHA_RND(b,c,d,e,f,g,h,a,w7,0x92722c85);
    w8 = rfc_sigma1_ptx(w6) + w1 + rfc_sigma0_ptx(w9) + w8;
    RFC_SHA_RND(a,b,c,d,e,f,g,h,w8,0xa2bfe8a1);
    w9 = rfc_sigma1_ptx(w7) + w2 + rfc_sigma0_ptx(w10) + w9;
    RFC_SHA_RND(h,a,b,c,d,e,f,g,w9,0xa81a664b);
    w10 = rfc_sigma1_ptx(w8) + w3 + rfc_sigma0_ptx(w11) + w10;
    RFC_SHA_RND(g,h,a,b,c,d,e,f,w10,0xc24b8b70);
    w11 = rfc_sigma1_ptx(w9) + w4 + rfc_sigma0_ptx(w12) + w11;
    RFC_SHA_RND(f,g,h,a,b,c,d,e,w11,0xc76c51a3);
    w12 = rfc_sigma1_ptx(w10) + w5 + rfc_sigma0_ptx(w13) + w12;
    RFC_SHA_RND(e,f,g,h,a,b,c,d,w12,0xd192e819);
    w13 = rfc_sigma1_ptx(w11) + w6 + rfc_sigma0_ptx(w14) + w13;
    RFC_SHA_RND(d,e,f,g,h,a,b,c,w13,0xd6990624);
    w14 = rfc_sigma1_ptx(w12) + w7 + rfc_sigma0_ptx(w15) + w14;
    RFC_SHA_RND(c,d,e,f,g,h,a,b,w14,0xf40e3585);
    w15 = rfc_sigma1_ptx(w13) + w8 + rfc_sigma0_ptx(w0) + w15;
    RFC_SHA_RND(b,c,d,e,f,g,h,a,w15,0x106aa070);
    w0 = rfc_sigma1_ptx(w14) + w9 + rfc_sigma0_ptx(w1) + w0;
    RFC_SHA_RND(a,b,c,d,e,f,g,h,w0,0x19a4c116);
    w1 = rfc_sigma1_ptx(w15) + w10 + rfc_sigma0_ptx(w2) + w1;
    RFC_SHA_RND(h,a,b,c,d,e,f,g,w1,0x1e376c08);
    w2 = rfc_sigma1_ptx(w0) + w11 + rfc_sigma0_ptx(w3) + w2;
    RFC_SHA_RND(g,h,a,b,c,d,e,f,w2,0x2748774c);
    w3 = rfc_sigma1_ptx(w1) + w12 + rfc_sigma0_ptx(w4) + w3;
    RFC_SHA_RND(f,g,h,a,b,c,d,e,w3,0x34b0bcb5);
    w4 = rfc_sigma1_ptx(w2) + w13 + rfc_sigma0_ptx(w5) + w4;
    RFC_SHA_RND(e,f,g,h,a,b,c,d,w4,0x391c0cb3);
    w5 = rfc_sigma1_ptx(w3) + w14 + rfc_sigma0_ptx(w6) + w5;
    RFC_SHA_RND(d,e,f,g,h,a,b,c,w5,0x4ed8aa4a);
    w6 = rfc_sigma1_ptx(w4) + w15 + rfc_sigma0_ptx(w7) + w6;
    RFC_SHA_RND(c,d,e,f,g,h,a,b,w6,0x5b9cca4f);
    w7 = rfc_sigma1_ptx(w5) + w0 + rfc_sigma0_ptx(w8) + w7;
    RFC_SHA_RND(b,c,d,e,f,g,h,a,w7,0x682e6ff3);
    w8 = rfc_sigma1_ptx(w6) + w1 + rfc_sigma0_ptx(w9) + w8;
    RFC_SHA_RND(a,b,c,d,e,f,g,h,w8,0x748f82ee);
    w9 = rfc_sigma1_ptx(w7) + w2 + rfc_sigma0_ptx(w10) + w9;
    RFC_SHA_RND(h,a,b,c,d,e,f,g,w9,0x78a5636f);
    w10 = rfc_sigma1_ptx(w8) + w3 + rfc_sigma0_ptx(w11) + w10;
    RFC_SHA_RND(g,h,a,b,c,d,e,f,w10,0x84c87814);
    w11 = rfc_sigma1_ptx(w9) + w4 + rfc_sigma0_ptx(w12) + w11;
    RFC_SHA_RND(f,g,h,a,b,c,d,e,w11,0x8cc70208);
    w12 = rfc_sigma1_ptx(w10) + w5 + rfc_sigma0_ptx(w13) + w12;
    RFC_SHA_RND(e,f,g,h,a,b,c,d,w12,0x90befffa);
    w13 = rfc_sigma1_ptx(w11) + w6 + rfc_sigma0_ptx(w14) + w13;
    RFC_SHA_RND(d,e,f,g,h,a,b,c,w13,0xa4506ceb);
    w14 = rfc_sigma1_ptx(w12) + w7 + rfc_sigma0_ptx(w15) + w14;
    RFC_SHA_RND(c,d,e,f,g,h,a,b,w14,0xbef9a3f7);
    w15 = rfc_sigma1_ptx(w13) + w8 + rfc_sigma0_ptx(w0) + w15;
    RFC_SHA_RND(b,c,d,e,f,g,h,a,w15,0xc67178f2);

    s[0]+=a; s[1]+=b; s[2]+=c; s[3]+=d;
    s[4]+=e; s[5]+=f; s[6]+=g; s[7]+=h;
}

#undef RFC_SHA_RND

FORCE_INLINE uint pack_be4(uchar a, uchar b, uchar c, uchar d)
{
    return ((uint)a<<24) | ((uint)b<<16) | ((uint)c<<8) | (uint)d;
}

FORCE_INLINE void digest_words_to_bytes(const uint s[8], uchar out32[32])
{
#pragma unroll
    for (int i=0;i<8;i++) {
        const uint v=s[i];
        out32[4*i+0]=(uchar)(v>>24);
        out32[4*i+1]=(uchar)(v>>16);
        out32[4*i+2]=(uchar)(v>>8);
        out32[4*i+3]=(uchar)v;
    }
}

FORCE_INLINE void hmac_key_states32_rfc(
    const uchar key32[32],
    uint inner_state[8],
    uint outer_state[8])
{
    uint w[16];

    /* RFC v37: build HMAC ipad/opad directly as SHA words.  The old
       path materialized a 64-byte private block and sha256_transform()
       immediately parsed those bytes back into words. */
#pragma unroll
    for (int i=0;i<8;i++)
        w[i] = pack_be4(key32[4*i], key32[4*i+1], key32[4*i+2], key32[4*i+3]) ^ 0x36363636U;
#pragma unroll
    for (int i=8;i<16;i++) w[i] = 0x36363636U;

    sha256_state_init_words_rfc(inner_state);
    sha256_transform_words_rfc_unrolled(inner_state, w);

#pragma unroll
    for (int i=0;i<8;i++)
        w[i] ^= 0x6a6a6a6aU; /* (K^0x36) ^ 0x6a == K^0x5c */
#pragma unroll
    for (int i=8;i<16;i++) w[i] = 0x5c5c5c5cU;

    sha256_state_init_words_rfc(outer_state);
    sha256_transform_words_rfc_unrolled(outer_state, w);
}

/* Fixed HMAC(K, 32-byte message), word-scheduled. */
FORCE_INLINE void hmac_states_msg32_words(
    const uint inner0[8],
    const uint outer0[8],
    const uchar msg32[32],
    uchar out32[32])
{
    uint si[8], so[8], w[16];

#pragma unroll
    for (int i=0;i<8;i++) { si[i]=inner0[i]; so[i]=outer0[i]; }

#pragma unroll
    for (int i=0;i<8;i++)
        w[i]=pack_be4(msg32[4*i],msg32[4*i+1],msg32[4*i+2],msg32[4*i+3]);

    w[8]=0x80000000U;
#pragma unroll
    for (int i=9;i<15;i++) w[i]=0U;
    w[15]=768U;
    sha256_transform_words_rfc_pad32_unrolled(si,w);

#pragma unroll
    for (int i=0;i<8;i++) w[i]=si[i];
    w[8]=0x80000000U;
#pragma unroll
    for (int i=9;i<15;i++) w[i]=0U;
    w[15]=768U;
    sha256_transform_words_rfc_unrolled(so,w);

    digest_words_to_bytes(so,out32);
}

/* Fixed 97-byte RFC6979 HMAC payload:
   V[32] || tag || seckey[32] || msg[32].
   Builds SHA words directly and avoids byte-buffer reconstruction. */
FORCE_INLINE void hmac_states_msg97_words(
    const uint inner0[8],
    const uint outer0[8],
    const uchar v32[32],
    uchar tag,
    const uchar seckey32[32],
    const uchar msg32[32],
    uchar out32[32])
{
    uint si[8], so[8], w[16];

#pragma unroll
    for (int i=0;i<8;i++) { si[i]=inner0[i]; so[i]=outer0[i]; }

    /* First post-key SHA block: V[32] || tag || first 31 seckey bytes. */
#pragma unroll
    for (int i=0;i<8;i++)
        w[i]=pack_be4(v32[4*i],v32[4*i+1],v32[4*i+2],v32[4*i+3]);

    w[8]  = pack_be4(tag,seckey32[0],seckey32[1],seckey32[2]);
    w[9]  = pack_be4(seckey32[3],seckey32[4],seckey32[5],seckey32[6]);
    w[10] = pack_be4(seckey32[7],seckey32[8],seckey32[9],seckey32[10]);
    w[11] = pack_be4(seckey32[11],seckey32[12],seckey32[13],seckey32[14]);
    w[12] = pack_be4(seckey32[15],seckey32[16],seckey32[17],seckey32[18]);
    w[13] = pack_be4(seckey32[19],seckey32[20],seckey32[21],seckey32[22]);
    w[14] = pack_be4(seckey32[23],seckey32[24],seckey32[25],seckey32[26]);
    w[15] = pack_be4(seckey32[27],seckey32[28],seckey32[29],seckey32[30]);
    sha256_transform_words_rfc_unrolled(si,w);

    /* Second block: last seckey byte || msg32 || padding || bit length 1288. */
    w[0] = pack_be4(seckey32[31],msg32[0],msg32[1],msg32[2]);
    w[1] = pack_be4(msg32[3],msg32[4],msg32[5],msg32[6]);
    w[2] = pack_be4(msg32[7],msg32[8],msg32[9],msg32[10]);
    w[3] = pack_be4(msg32[11],msg32[12],msg32[13],msg32[14]);
    w[4] = pack_be4(msg32[15],msg32[16],msg32[17],msg32[18]);
    w[5] = pack_be4(msg32[19],msg32[20],msg32[21],msg32[22]);
    w[6] = pack_be4(msg32[23],msg32[24],msg32[25],msg32[26]);
    w[7] = pack_be4(msg32[27],msg32[28],msg32[29],msg32[30]);
    w[8] = ((uint)msg32[31] << 24) | 0x00800000U;
#pragma unroll
    for (int i=9;i<15;i++) w[i]=0U;
    w[15]=1288U;
    sha256_transform_words_rfc_unrolled(si,w);

    /* Outer HMAC block: inner digest || padding || 768-bit length. */
#pragma unroll
    for (int i=0;i<8;i++) w[i]=si[i];
    w[8]=0x80000000U;
#pragma unroll
    for (int i=9;i<15;i++) w[i]=0U;
    w[15]=768U;
    sha256_transform_words_rfc_unrolled(so,w);

    digest_words_to_bytes(so,out32);
}

FORCE_INLINE void hmac_states_msg32_rfc_fast(
    const uint inner0[8],
    const uint outer0[8],
    const uchar msg32[32],
    uchar out32[32])
{
    uint si[8], so[8];
    uchar block[64];

#pragma unroll
    for (int i=0;i<8;i++) {
        si[i]=inner0[i];
        so[i]=outer0[i];
    }

    /* Inner block: msg32 || 0x80 || zero... || bitlen(96 bytes = 768 bits) */
#pragma unroll
    for (int i=0;i<32;i++) block[i]=msg32[i];
    block[32]=0x80;
#pragma unroll
    for (int i=33;i<56;i++) block[i]=0;
    block[56]=0; block[57]=0; block[58]=0; block[59]=0;
    block[60]=0; block[61]=0; block[62]=3; block[63]=0; /* 768 */
    sha256_transform(si, block);

    /* Reuse block memory for outer hash: digest || pad || 768-bit length. */
#pragma unroll
    for (int i=0;i<8;i++) {
        uint v=si[i];
        block[4*i+0]=(uchar)(v>>24);
        block[4*i+1]=(uchar)(v>>16);
        block[4*i+2]=(uchar)(v>>8);
        block[4*i+3]=(uchar)v;
    }
    block[32]=0x80;
#pragma unroll
    for (int i=33;i<56;i++) block[i]=0;
    block[56]=0; block[57]=0; block[58]=0; block[59]=0;
    block[60]=0; block[61]=0; block[62]=3; block[63]=0;
    sha256_transform(so, block);

#pragma unroll
    for (int i=0;i<8;i++) {
        uint v=so[i];
        out32[4*i+0]=(uchar)(v>>24);
        out32[4*i+1]=(uchar)(v>>16);
        out32[4*i+2]=(uchar)(v>>8);
        out32[4*i+3]=(uchar)v;
    }
}

FORCE_INLINE void hmac_states_msg32_rfc(
    const uint inner0[8],
    const uint outer0[8],
    const uchar msg32[32],
    uchar out32[32])
{
    uint si[8], so[8];
    uchar block[64], inner_digest[32];

#pragma unroll
    for (int i=0;i<8;i++) { si[i]=inner0[i]; so[i]=outer0[i]; }

#pragma unroll
    for (int i=0;i<32;i++) block[i]=msg32[i];
    block[32]=0x80;
#pragma unroll
    for (int i=33;i<56;i++) block[i]=0;
    write_be32(block+56, 0U);
    write_be32(block+60, 768U);
    sha256_transform(si, block);
    sha256_state_to_bytes_rfc(si, inner_digest);

#pragma unroll
    for (int i=0;i<32;i++) block[i]=inner_digest[i];
    block[32]=0x80;
#pragma unroll
    for (int i=33;i<56;i++) block[i]=0;
    write_be32(block+56, 0U);
    write_be32(block+60, 768U);
    sha256_transform(so, block);
    sha256_state_to_bytes_rfc(so, out32);
}

FORCE_INLINE void hmac_states_msg97_rfc(
    const uint inner0[8],
    const uint outer0[8],
    const uchar v32[32],
    uchar tag,
    const uchar seckey32[32],
    const uchar msg32[32],
    uchar out32[32])
{
    uint si[8], so[8];
    uchar b0[64], b1[64], inner_digest[32];

#pragma unroll
    for (int i=0;i<8;i++) { si[i]=inner0[i]; so[i]=outer0[i]; }

#pragma unroll
    for (int i=0;i<32;i++) b0[i]=v32[i];
    b0[32]=tag;
#pragma unroll
    for (int i=0;i<31;i++) b0[33+i]=seckey32[i];
    sha256_transform(si, b0);

    b1[0]=seckey32[31];
#pragma unroll
    for (int i=0;i<32;i++) b1[1+i]=msg32[i];
    b1[33]=0x80;
#pragma unroll
    for (int i=34;i<56;i++) b1[i]=0;
    write_be32(b1+56, 0U);
    write_be32(b1+60, 1288U);
    sha256_transform(si, b1);
    sha256_state_to_bytes_rfc(si, inner_digest);

#pragma unroll
    for (int i=0;i<32;i++) b0[i]=inner_digest[i];
    b0[32]=0x80;
#pragma unroll
    for (int i=33;i<56;i++) b0[i]=0;
    write_be32(b0+56, 0U);
    write_be32(b0+60, 768U);
    sha256_transform(so, b0);
    sha256_state_to_bytes_rfc(so, out32);
}

// =============================================================================
// v56: word-native fixed-layout RFC6979 HMAC helpers
// Keep K/V as eight SHA-256 words between HMAC operations.  The previous
// path serialized each digest to 32 bytes and immediately reparsed those
// bytes into words for the next HMAC key/message block.
// =============================================================================
FORCE_INLINE void hmac_key_states32_words_rfc(
    const uint keyw[8], uint inner_state[8], uint outer_state[8])
{
    uint w[16];
#pragma unroll
    for (int i=0;i<8;i++) w[i] = keyw[i] ^ 0x36363636U;
#pragma unroll
    for (int i=8;i<16;i++) w[i] = 0x36363636U;
    sha256_state_init_words_rfc(inner_state);
    sha256_transform_words_rfc_unrolled(inner_state, w);

#pragma unroll
    for (int i=0;i<8;i++) w[i] ^= 0x6a6a6a6aU;
#pragma unroll
    for (int i=8;i<16;i++) w[i] = 0x5c5c5c5cU;
    sha256_state_init_words_rfc(outer_state);
    sha256_transform_words_rfc_unrolled(outer_state, w);
}

FORCE_INLINE void hmac_states_msg32_u32_rfc(
    const uint inner0[8], const uint outer0[8],
    const uint msgw[8], uint outw[8])
{
    uint si[8], so[8], w[16];
#pragma unroll
    for (int i=0;i<8;i++) { si[i]=inner0[i]; so[i]=outer0[i]; w[i]=msgw[i]; }
    w[8]=0x80000000U;
#pragma unroll
    for (int i=9;i<15;i++) w[i]=0U;
    w[15]=768U;
    sha256_transform_words_rfc_unrolled(si,w);

#pragma unroll
    for (int i=0;i<8;i++) w[i]=si[i];
    w[8]=0x80000000U;
#pragma unroll
    for (int i=9;i<15;i++) w[i]=0U;
    w[15]=768U;
    sha256_transform_words_rfc_pad32_unrolled(so,w);
#pragma unroll
    for (int i=0;i<8;i++) outw[i]=so[i];
}

FORCE_INLINE void hmac_states_msg97_vwords_rfc(
    const uint inner0[8], const uint outer0[8],
    const uint vw[8], uchar tag,
    const uchar seckey32[32], const uchar msg32[32], uint outw[8])
{
    uint si[8], so[8], w[16];
#pragma unroll
    for (int i=0;i<8;i++) { si[i]=inner0[i]; so[i]=outer0[i]; w[i]=vw[i]; }

    w[8]  = pack_be4(tag,seckey32[0],seckey32[1],seckey32[2]);
    w[9]  = pack_be4(seckey32[3],seckey32[4],seckey32[5],seckey32[6]);
    w[10] = pack_be4(seckey32[7],seckey32[8],seckey32[9],seckey32[10]);
    w[11] = pack_be4(seckey32[11],seckey32[12],seckey32[13],seckey32[14]);
    w[12] = pack_be4(seckey32[15],seckey32[16],seckey32[17],seckey32[18]);
    w[13] = pack_be4(seckey32[19],seckey32[20],seckey32[21],seckey32[22]);
    w[14] = pack_be4(seckey32[23],seckey32[24],seckey32[25],seckey32[26]);
    w[15] = pack_be4(seckey32[27],seckey32[28],seckey32[29],seckey32[30]);
    sha256_transform_words_rfc_unrolled(si,w);

    w[0] = pack_be4(seckey32[31],msg32[0],msg32[1],msg32[2]);
    w[1] = pack_be4(msg32[3],msg32[4],msg32[5],msg32[6]);
    w[2] = pack_be4(msg32[7],msg32[8],msg32[9],msg32[10]);
    w[3] = pack_be4(msg32[11],msg32[12],msg32[13],msg32[14]);
    w[4] = pack_be4(msg32[15],msg32[16],msg32[17],msg32[18]);
    w[5] = pack_be4(msg32[19],msg32[20],msg32[21],msg32[22]);
    w[6] = pack_be4(msg32[23],msg32[24],msg32[25],msg32[26]);
    w[7] = pack_be4(msg32[27],msg32[28],msg32[29],msg32[30]);
    w[8] = ((uint)msg32[31] << 24) | 0x00800000U;
#pragma unroll
    for (int i=9;i<15;i++) w[i]=0U;
    w[15]=1288U;
    sha256_transform_words_rfc_unrolled(si,w);

#pragma unroll
    for (int i=0;i<8;i++) w[i]=si[i];
    w[8]=0x80000000U;
#pragma unroll
    for (int i=9;i<15;i++) w[i]=0U;
    w[15]=768U;
    sha256_transform_words_rfc_pad32_unrolled(so,w);
#pragma unroll
    for (int i=0;i<8;i++) outw[i]=so[i];
}

// =============================================================================
// RFC6979 seckey precompute for mining
// =============================================================================
// The first RFC6979 HMAC uses the fixed all-zero K and begins with:
//   V(32 x 0x01) || 0x00 || seckey32
// Only the final reduced-message 32 bytes vary per nonce.  Precompute the
// HMAC state after the fixed/seckey prefix once per work-item.
typedef struct {
    /* RFC6979 step-d fixed prefix state.  The first post-key SHA block is
       V(32 x 0x01) || 0x00 || seckey[0..30], so it can be compressed once
       per work-item.  Only seckey[31] || msg32 remains per nonce. */
    uint step_d_inner[8];
    uint step_d_outer[8];
    uchar seckey31;
} RFC6979_SECKEY_PRECOMP;

FORCE_INLINE void rfc6979_precompute_seckey(const uchar* seckey32,
                                             RFC6979_SECKEY_PRECOMP* pc) {
    uchar zero_key[32];
    uint w[16];

#pragma unroll
    for (int i=0;i<32;i++) zero_key[i]=0;

    /* HMAC K=0: precompute SHA states after ipad/opad key blocks. */
    hmac_key_states32_rfc(zero_key, pc->step_d_inner, pc->step_d_outer);

    /* First 64-byte post-key block: V || 0x00 || seckey[0..30]. */
#pragma unroll
    for (int i=0;i<8;i++) w[i]=0x01010101U;
    w[8]  = pack_be4((uchar)0x00,seckey32[0],seckey32[1],seckey32[2]);
    w[9]  = pack_be4(seckey32[3],seckey32[4],seckey32[5],seckey32[6]);
    w[10] = pack_be4(seckey32[7],seckey32[8],seckey32[9],seckey32[10]);
    w[11] = pack_be4(seckey32[11],seckey32[12],seckey32[13],seckey32[14]);
    w[12] = pack_be4(seckey32[15],seckey32[16],seckey32[17],seckey32[18]);
    w[13] = pack_be4(seckey32[19],seckey32[20],seckey32[21],seckey32[22]);
    w[14] = pack_be4(seckey32[23],seckey32[24],seckey32[25],seckey32[26]);
    w[15] = pack_be4(seckey32[27],seckey32[28],seckey32[29],seckey32[30]);
    sha256_transform_words_rfc_unrolled(pc->step_d_inner,w);
    pc->seckey31=seckey32[31];
}

FORCE_INLINE void rfc6979_step_d_fast(const RFC6979_SECKEY_PRECOMP* pc,
                                       const uchar msg32[32],
                                       uchar out32[32]) {
    uint si[8], so[8], w[16];
#pragma unroll
    for (int i=0;i<8;i++) { si[i]=pc->step_d_inner[i]; so[i]=pc->step_d_outer[i]; }

    /* Remaining 33 bytes: seckey[31] || msg32, then SHA padding.
       Total HMAC inner length = 64-byte ipad + 97-byte message = 161 bytes. */
    w[0] = pack_be4(pc->seckey31,msg32[0],msg32[1],msg32[2]);
    w[1] = pack_be4(msg32[3],msg32[4],msg32[5],msg32[6]);
    w[2] = pack_be4(msg32[7],msg32[8],msg32[9],msg32[10]);
    w[3] = pack_be4(msg32[11],msg32[12],msg32[13],msg32[14]);
    w[4] = pack_be4(msg32[15],msg32[16],msg32[17],msg32[18]);
    w[5] = pack_be4(msg32[19],msg32[20],msg32[21],msg32[22]);
    w[6] = pack_be4(msg32[23],msg32[24],msg32[25],msg32[26]);
    w[7] = pack_be4(msg32[27],msg32[28],msg32[29],msg32[30]);
    w[8] = ((uint)msg32[31] << 24) | 0x00800000U;
#pragma unroll
    for (int i=9;i<15;i++) w[i]=0U;
    w[15]=1288U;
    sha256_transform_words_rfc_unrolled(si,w);

    /* HMAC outer: inner digest || pad; total length 64+32 = 96 bytes. */
#pragma unroll
    for (int i=0;i<8;i++) w[i]=si[i];
    w[8]=0x80000000U;
#pragma unroll
    for (int i=9;i<15;i++) w[i]=0U;
    w[15]=768U;
    sha256_transform_words_rfc_unrolled(so,w);
    digest_words_to_bytes(so,out32);
}

FORCE_INLINE void rfc6979_step_d_words(const RFC6979_SECKEY_PRECOMP* pc,
                                        const uchar msg32[32], uint outw[8]) {
    uint si[8], so[8], w[16];
#pragma unroll
    for (int i=0;i<8;i++) { si[i]=pc->step_d_inner[i]; so[i]=pc->step_d_outer[i]; }
    w[0] = pack_be4(pc->seckey31,msg32[0],msg32[1],msg32[2]);
    w[1] = pack_be4(msg32[3],msg32[4],msg32[5],msg32[6]);
    w[2] = pack_be4(msg32[7],msg32[8],msg32[9],msg32[10]);
    w[3] = pack_be4(msg32[11],msg32[12],msg32[13],msg32[14]);
    w[4] = pack_be4(msg32[15],msg32[16],msg32[17],msg32[18]);
    w[5] = pack_be4(msg32[19],msg32[20],msg32[21],msg32[22]);
    w[6] = pack_be4(msg32[23],msg32[24],msg32[25],msg32[26]);
    w[7] = pack_be4(msg32[27],msg32[28],msg32[29],msg32[30]);
    w[8] = ((uint)msg32[31] << 24) | 0x00800000U;
#pragma unroll
    for (int i=9;i<15;i++) w[i]=0U;
    w[15]=1288U;
    sha256_transform_words_rfc_unrolled(si,w);
#pragma unroll
    for (int i=0;i<8;i++) w[i]=si[i];
    w[8]=0x80000000U;
#pragma unroll
    for (int i=9;i<15;i++) w[i]=0U;
    w[15]=768U;
    sha256_transform_words_rfc_pad32_unrolled(so,w);
#pragma unroll
    for (int i=0;i<8;i++) outw[i]=so[i];
}

// NEWFORK step d with libsecp256k1's 32-byte extra-entropy argument.  The
// precomputed state already includes V=0x01..01, tag=0 and seckey[0..30].
// Only the reduced message and LE32(test_case) vary between candidates.
FORCE_INLINE void rfc6979_step_d_extra_words(
    const RFC6979_SECKEY_PRECOMP* pc, const uchar msg32[32],
    ulong extra64, uint outw[8]) {
    uint si[8], so[8], w[16];
#pragma unroll
    for (int i=0;i<8;i++) { si[i]=pc->step_d_inner[i]; so[i]=pc->step_d_outer[i]; }

    // seckey[31] || msg32 || extra[0..30], extra = LE64(nonce) || zeros
    w[0] = pack_be4(pc->seckey31,msg32[0],msg32[1],msg32[2]);
    w[1] = pack_be4(msg32[3],msg32[4],msg32[5],msg32[6]);
    w[2] = pack_be4(msg32[7],msg32[8],msg32[9],msg32[10]);
    w[3] = pack_be4(msg32[11],msg32[12],msg32[13],msg32[14]);
    w[4] = pack_be4(msg32[15],msg32[16],msg32[17],msg32[18]);
    w[5] = pack_be4(msg32[19],msg32[20],msg32[21],msg32[22]);
    w[6] = pack_be4(msg32[23],msg32[24],msg32[25],msg32[26]);
    w[7] = pack_be4(msg32[27],msg32[28],msg32[29],msg32[30]);
    w[8] = ((uint)msg32[31] << 24) |
           ((uint)(extra64 & 0xffUL) << 16) |
           ((uint)((extra64 >> 8) & 0xffUL) << 8) |
           ((uint)((extra64 >> 16) & 0xffUL));
    w[9] = ((uint)((extra64 >> 24) & 0xffUL) << 24) |
           ((uint)((extra64 >> 32) & 0xffUL) << 16) |
           ((uint)((extra64 >> 40) & 0xffUL) << 8) |
           ((uint)((extra64 >> 48) & 0xffUL));
    w[10] = ((uint)((extra64 >> 56) & 0xffUL) << 24);
#pragma unroll
    for (int i=11;i<16;i++) w[i]=0U;
    sha256_transform_words_rfc_unrolled(si,w);

    // extra[31] is zero, followed by padding.  Inner length is
    // 64-byte ipad + 129-byte RFC payload = 193 bytes = 1544 bits.
    w[0]=0x00800000U;
#pragma unroll
    for (int i=1;i<15;i++) w[i]=0U;
    w[15]=1544U;
    sha256_transform_words_rfc_unrolled(si,w);

#pragma unroll
    for (int i=0;i<8;i++) w[i]=si[i];
    w[8]=0x80000000U;
#pragma unroll
    for (int i=9;i<15;i++) w[i]=0U;
    w[15]=768U;
    sha256_transform_words_rfc_pad32_unrolled(so,w);
#pragma unroll
    for (int i=0;i<8;i++) outw[i]=so[i];
}

// HMAC(K, V || tag || seckey || msg || extra), where extra is
// LE32(test_case) followed by 28 zero bytes.  The 129-byte payload has a
// fixed three-block shape after the HMAC key block.
FORCE_INLINE void hmac_states_msg129_extra_vwords_rfc(
    const uint inner0[8], const uint outer0[8], const uint vw[8], uchar tag,
    const uchar seckey32[32], const uchar msg32[32], ulong extra64,
    uint outw[8]) {
    uint si[8], so[8], w[16];
#pragma unroll
    for (int i=0;i<8;i++) { si[i]=inner0[i]; so[i]=outer0[i]; w[i]=vw[i]; }

    w[8]  = pack_be4(tag,seckey32[0],seckey32[1],seckey32[2]);
    w[9]  = pack_be4(seckey32[3],seckey32[4],seckey32[5],seckey32[6]);
    w[10] = pack_be4(seckey32[7],seckey32[8],seckey32[9],seckey32[10]);
    w[11] = pack_be4(seckey32[11],seckey32[12],seckey32[13],seckey32[14]);
    w[12] = pack_be4(seckey32[15],seckey32[16],seckey32[17],seckey32[18]);
    w[13] = pack_be4(seckey32[19],seckey32[20],seckey32[21],seckey32[22]);
    w[14] = pack_be4(seckey32[23],seckey32[24],seckey32[25],seckey32[26]);
    w[15] = pack_be4(seckey32[27],seckey32[28],seckey32[29],seckey32[30]);
    sha256_transform_words_rfc_unrolled(si,w);

    w[0] = pack_be4(seckey32[31],msg32[0],msg32[1],msg32[2]);
    w[1] = pack_be4(msg32[3],msg32[4],msg32[5],msg32[6]);
    w[2] = pack_be4(msg32[7],msg32[8],msg32[9],msg32[10]);
    w[3] = pack_be4(msg32[11],msg32[12],msg32[13],msg32[14]);
    w[4] = pack_be4(msg32[15],msg32[16],msg32[17],msg32[18]);
    w[5] = pack_be4(msg32[19],msg32[20],msg32[21],msg32[22]);
    w[6] = pack_be4(msg32[23],msg32[24],msg32[25],msg32[26]);
    w[7] = pack_be4(msg32[27],msg32[28],msg32[29],msg32[30]);
    w[8] = ((uint)msg32[31] << 24) |
           ((uint)(extra64 & 0xffUL) << 16) |
           ((uint)((extra64 >> 8) & 0xffUL) << 8) |
           ((uint)((extra64 >> 16) & 0xffUL));
    w[9] = ((uint)((extra64 >> 24) & 0xffUL) << 24) |
           ((uint)((extra64 >> 32) & 0xffUL) << 16) |
           ((uint)((extra64 >> 40) & 0xffUL) << 8) |
           ((uint)((extra64 >> 48) & 0xffUL));
    w[10] = ((uint)((extra64 >> 56) & 0xffUL) << 24);
#pragma unroll
    for (int i=11;i<16;i++) w[i]=0U;
    sha256_transform_words_rfc_unrolled(si,w);

    w[0]=0x00800000U;
#pragma unroll
    for (int i=1;i<15;i++) w[i]=0U;
    w[15]=1544U;
    sha256_transform_words_rfc_unrolled(si,w);

#pragma unroll
    for (int i=0;i<8;i++) w[i]=si[i];
    w[8]=0x80000000U;
#pragma unroll
    for (int i=9;i<15;i++) w[i]=0U;
    w[15]=768U;
    sha256_transform_words_rfc_pad32_unrolled(so,w);
#pragma unroll
    for (int i=0;i<8;i++) outw[i]=so[i];
}

FORCE_INLINE void rfc6979_generate_k_testcase_words(
    const RFC6979_SECKEY_PRECOMP* pc, const uchar seckey32[32],
    const uchar msgmod32[32], ulong extra64, uint nonce_words[8]) {
    uint k[8], v[8], is[8], os[8];
    rfc6979_step_d_extra_words(pc,msgmod32,extra64,k);
#pragma unroll
    for (int i=0;i<8;i++) v[i]=0x01010101U;
    hmac_key_states32_words_rfc(k,is,os);
    hmac_states_msg32_u32_rfc(is,os,v,v);
    hmac_states_msg129_extra_vwords_rfc(
        is,os,v,(uchar)0x01,seckey32,msgmod32,extra64,k);
    hmac_key_states32_words_rfc(k,is,os);
    hmac_states_msg32_u32_rfc(is,os,v,v);
    hmac_states_msg32_u32_rfc(is,os,v,v);
#pragma unroll
    for (int i=0;i<8;i++) nonce_words[i]=v[i];
}

// msg_scalar MUST already be reduced mod n.  This avoids the duplicate
// scalar parse/reduction previously performed by RFC6979 and signing.
FORCE_INLINE void rfc6979_generate_k_precomp(const RFC6979_SECKEY_PRECOMP* pc,
                                              const uchar* seckey32,
                                              const Scalar* msg_scalar,
                                              uchar* nonce32) {
    uchar msgmod32[32], v[32], k[32];
    uint is[8], os[8];

    scalar_get_b32(msgmod32, msg_scalar);

    /* Step d: fixed-layout HMAC from precompressed seckey prefix. */
    rfc6979_step_d_fast(pc, msgmod32, k);

    /* From here on, all HMAC payloads have fixed sizes. */
#pragma unroll
    for (int i=0;i<32;i++) v[i]=1;

    /* V = HMAC_K(V) */
    hmac_key_states32_rfc(k, is, os);
    hmac_states_msg32_words(is, os, v, v);

    /* K = HMAC_K(V || 0x01 || seckey || msg) */
    hmac_states_msg97_words(is, os, v, (uchar)0x01, seckey32, msgmod32, k);

    /* New K for both following V updates. */
    hmac_key_states32_rfc(k, is, os);

    /* V = HMAC_K(V) */
    hmac_states_msg32_words(is, os, v, v);

    /* Candidate V = HMAC_K(V) */
    hmac_states_msg32_words(is, os, v, v);

#pragma unroll
    for (int i=0;i<32;i++) nonce32[i]=v[i];

}

// v56 word-native RFC6979 mining generator.  K and V never round-trip
// through byte arrays; only msgmod32/seckey remain byte-oriented because they
// enter the fixed 97-byte RFC payload.
FORCE_INLINE void rfc6979_generate_k_precomp_words(
    const RFC6979_SECKEY_PRECOMP* pc, const uchar* seckey32,
    const Scalar* msg_scalar, uint nonce_words[8])
{
    uchar msgmod32[32];
    uint k[8], v[8], is[8], os[8];
    scalar_get_b32(msgmod32, msg_scalar);

    rfc6979_step_d_words(pc, msgmod32, k);
#pragma unroll
    for (int i=0;i<8;i++) v[i]=0x01010101U;

    hmac_key_states32_words_rfc(k,is,os);
    hmac_states_msg32_u32_rfc(is,os,v,v);
    hmac_states_msg97_vwords_rfc(is,os,v,(uchar)0x01,seckey32,msgmod32,k);
    hmac_key_states32_words_rfc(k,is,os);
    hmac_states_msg32_u32_rfc(is,os,v,v);
    hmac_states_msg32_u32_rfc(is,os,v,v);
#pragma unroll
    for (int i=0;i<8;i++) nonce_words[i]=v[i];
}


// New-fork RFC6979: fixed message, test_case is libsecp256k1 extra_entropy.
// test_case==0 matches a nullptr data argument; nonzero cases pass a 32-byte
// buffer containing LE32(test_case) followed by 28 zero bytes.
FORCE_INLINE void rfc6979_generate_k_testcase(const uchar* seckey32,
                                               const uchar* msg32,
                                               ulong extra64,
                                               uint nonce_words[8]) {
    uchar v[32], k[32], keydata[96], extra[32];
    HMAC_SHA256_CTX hmac;
    Scalar msg_tmp;
    scalar_set_b32(&msg_tmp, msg32);
    scalar_reduce(&msg_tmp);
    uchar msgmod32[32];
    scalar_get_b32(msgmod32, &msg_tmp);
    for (int i=0;i<32;i++) { keydata[i]=seckey32[i]; keydata[32+i]=msgmod32[i]; extra[i]=0; }
    extra[0]=(uchar)extra64; extra[1]=(uchar)(extra64>>8); extra[2]=(uchar)(extra64>>16); extra[3]=(uchar)(extra64>>24);
    extra[4]=(uchar)(extra64>>32); extra[5]=(uchar)(extra64>>40); extra[6]=(uchar)(extra64>>48); extra[7]=(uchar)(extra64>>56);
    for (int i=0;i<32;i++) keydata[64+i]=extra[i];
    for (int i=0;i<32;i++) { v[i]=1; k[i]=0; }
    uchar zero=0, one=1;
    const uint kdlen = extra64 ? 96u : 64u;
    hmac_sha256_init(&hmac,k,32); hmac_sha256_update(&hmac,v,32); hmac_sha256_update(&hmac,&zero,1); hmac_sha256_update(&hmac,keydata,kdlen); hmac_sha256_final(&hmac,k);
    hmac_sha256_init(&hmac,k,32); hmac_sha256_update(&hmac,v,32); hmac_sha256_final(&hmac,v);
    hmac_sha256_init(&hmac,k,32); hmac_sha256_update(&hmac,v,32); hmac_sha256_update(&hmac,&one,1); hmac_sha256_update(&hmac,keydata,kdlen); hmac_sha256_final(&hmac,k);
    hmac_sha256_init(&hmac,k,32); hmac_sha256_update(&hmac,v,32); hmac_sha256_final(&hmac,v);
    hmac_sha256_init(&hmac,k,32); hmac_sha256_update(&hmac,v,32); hmac_sha256_final(&hmac,v);
    #pragma unroll
    for (int i=0;i<8;i++) nonce_words[i]=((uint)v[4*i]<<24)|((uint)v[4*i+1]<<16)|((uint)v[4*i+2]<<8)|(uint)v[4*i+3];
}

FORCE_INLINE int hash_meets_target_le(const uchar hash32[32], const uchar target32[32]) {
    #pragma unroll
    for (int i=31;i>=0;--i) {
        if (hash32[i] < target32[i]) return 1;
        if (hash32[i] > target32[i]) return 0;
    }
    return 1;
}

// =============================================================================
// Scalar Arithmetic (mod curve order n)
// =============================================================================
// n = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

// Scalar type is already defined in secp256k1_point.cl

// Load big-endian 32 bytes into Scalar (little-endian limbs)
FORCE_INLINE void scalar_set_b32(Scalar* s, const uchar* b32) {
    s->limbs[3] = ((ulong)b32[0] << 56) | ((ulong)b32[1] << 48) | ((ulong)b32[2] << 40) | ((ulong)b32[3] << 32) |
                   ((ulong)b32[4] << 24) | ((ulong)b32[5] << 16) | ((ulong)b32[6] << 8)  | (ulong)b32[7];
    s->limbs[2] = ((ulong)b32[8] << 56) | ((ulong)b32[9] << 48) | ((ulong)b32[10] << 40) | ((ulong)b32[11] << 32) |
                   ((ulong)b32[12] << 24) | ((ulong)b32[13] << 16) | ((ulong)b32[14] << 8)  | (ulong)b32[15];
    s->limbs[1] = ((ulong)b32[16] << 56) | ((ulong)b32[17] << 48) | ((ulong)b32[18] << 40) | ((ulong)b32[19] << 32) |
                   ((ulong)b32[20] << 24) | ((ulong)b32[21] << 16) | ((ulong)b32[22] << 8)  | (ulong)b32[23];
    s->limbs[0] = ((ulong)b32[24] << 56) | ((ulong)b32[25] << 48) | ((ulong)b32[26] << 40) | ((ulong)b32[27] << 32) |
                   ((ulong)b32[28] << 24) | ((ulong)b32[29] << 16) | ((ulong)b32[30] << 8)  | (ulong)b32[31];
}

// v56: direct SHA-digest-word -> scalar conversion. SHA state words are
// exactly the big-endian 32-bit words that scalar_set_b32 would parse.
FORCE_INLINE void scalar_set_sha256_words(Scalar* s, const uint w[8]) {
    s->limbs[3] = ((ulong)w[0] << 32) | (ulong)w[1];
    s->limbs[2] = ((ulong)w[2] << 32) | (ulong)w[3];
    s->limbs[1] = ((ulong)w[4] << 32) | (ulong)w[5];
    s->limbs[0] = ((ulong)w[6] << 32) | (ulong)w[7];
}

// Store Scalar to big-endian 32 bytes
FORCE_INLINE void scalar_get_b32(uchar* b32, const Scalar* s) {
    b32[0]  = (uchar)(s->limbs[3] >> 56); b32[1]  = (uchar)(s->limbs[3] >> 48);
    b32[2]  = (uchar)(s->limbs[3] >> 40); b32[3]  = (uchar)(s->limbs[3] >> 32);
    b32[4]  = (uchar)(s->limbs[3] >> 24); b32[5]  = (uchar)(s->limbs[3] >> 16);
    b32[6]  = (uchar)(s->limbs[3] >> 8);  b32[7]  = (uchar)(s->limbs[3]);
    b32[8]  = (uchar)(s->limbs[2] >> 56); b32[9]  = (uchar)(s->limbs[2] >> 48);
    b32[10] = (uchar)(s->limbs[2] >> 40); b32[11] = (uchar)(s->limbs[2] >> 32);
    b32[12] = (uchar)(s->limbs[2] >> 24); b32[13] = (uchar)(s->limbs[2] >> 16);
    b32[14] = (uchar)(s->limbs[2] >> 8);  b32[15] = (uchar)(s->limbs[2]);
    b32[16] = (uchar)(s->limbs[1] >> 56); b32[17] = (uchar)(s->limbs[1] >> 48);
    b32[18] = (uchar)(s->limbs[1] >> 40); b32[19] = (uchar)(s->limbs[1] >> 32);
    b32[20] = (uchar)(s->limbs[1] >> 24); b32[21] = (uchar)(s->limbs[1] >> 16);
    b32[22] = (uchar)(s->limbs[1] >> 8);  b32[23] = (uchar)(s->limbs[1]);
    b32[24] = (uchar)(s->limbs[0] >> 56); b32[25] = (uchar)(s->limbs[0] >> 48);
    b32[26] = (uchar)(s->limbs[0] >> 40); b32[27] = (uchar)(s->limbs[0] >> 32);
    b32[28] = (uchar)(s->limbs[0] >> 24); b32[29] = (uchar)(s->limbs[0] >> 16);
    b32[30] = (uchar)(s->limbs[0] >> 8);  b32[31] = (uchar)(s->limbs[0]);
}

// n in little-endian limbs
#define N_LIMB0 0xBFD25E8CD0364141UL
#define N_LIMB1 0xBAAEDCE6AF48A03BUL
#define N_LIMB2 0xFFFFFFFFFFFFFFFEUL
#define N_LIMB3 0xFFFFFFFFFFFFFFFFUL

// Check if scalar >= n
FORCE_INLINE int scalar_check_overflow(const Scalar* s) {
    if (s->limbs[3] > N_LIMB3) return 1;
    if (s->limbs[3] < N_LIMB3) return 0;
    if (s->limbs[2] > N_LIMB2) return 1;
    if (s->limbs[2] < N_LIMB2) return 0;
    if (s->limbs[1] > N_LIMB1) return 1;
    if (s->limbs[1] < N_LIMB1) return 0;
    if (s->limbs[0] >= N_LIMB0) return 1;
    return 0;
}

// Reduce scalar mod n (simple subtraction if >= n)
FORCE_INLINE void scalar_reduce(Scalar* s) {
    if (!scalar_check_overflow(s)) return;
    ulong borrow = 0;
    ulong d0 = sub_with_borrow(s->limbs[0], N_LIMB0, 0, &borrow);
    ulong d1 = sub_with_borrow(s->limbs[1], N_LIMB1, borrow, &borrow);
    ulong d2 = sub_with_borrow(s->limbs[2], N_LIMB2, borrow, &borrow);
    ulong d3 = sub_with_borrow(s->limbs[3], N_LIMB3, borrow, &borrow);
    s->limbs[0] = d0; s->limbs[1] = d1;
    s->limbs[2] = d2; s->limbs[3] = d3;
}

// Scalar addition: r = (a + b) mod n
FORCE_INLINE void scalar_add_mod_n(Scalar* r, const Scalar* a, const Scalar* b) {
    ulong carry = 0;
    r->limbs[0] = add_with_carry(a->limbs[0], b->limbs[0], 0, &carry);
    r->limbs[1] = add_with_carry(a->limbs[1], b->limbs[1], carry, &carry);
    r->limbs[2] = add_with_carry(a->limbs[2], b->limbs[2], carry, &carry);
    r->limbs[3] = add_with_carry(a->limbs[3], b->limbs[3], carry, &carry);
    // If carry or >= n, reduce
    if (carry || scalar_check_overflow(r)) {
        ulong borrow = 0;
        r->limbs[0] = sub_with_borrow(r->limbs[0], N_LIMB0, 0, &borrow);
        r->limbs[1] = sub_with_borrow(r->limbs[1], N_LIMB1, borrow, &borrow);
        r->limbs[2] = sub_with_borrow(r->limbs[2], N_LIMB2, borrow, &borrow);
        r->limbs[3] = sub_with_borrow(r->limbs[3], N_LIMB3, borrow, &borrow);
    }
}

// =============================================================================
// Scalar Reduction: reduce 512-bit product mod n
// =============================================================================
// Direct port of libsecp256k1's secp256k1_scalar_reduce_512.
// Uses a 192-bit accumulator (c0, c1, c2) to avoid any possibility of
// carry overflow — the same approach used by the proven CUDA miner.
//
// Reduces in three passes: 512→385 bits, 385→258 bits, 258→256 bits.

// --- 192-bit accumulator macros (ported from CUDA secp256k1) ---
// These use the same carry-propagation logic as libsecp256k1.
// c2:c1:c0 forms the 192-bit running sum.

#define ACC_MULADD_FAST(a, b) { \
    u64x2 _m = mul64_full((a), (b)); \
    c0 += _m.x; \
    ulong _th = _m.y + ((c0 < _m.x) ? 1UL : 0UL); \
    c1 += _th; \
}

#define ACC_MULADD(a, b) { \
    u64x2 _m = mul64_full((a), (b)); \
    c0 += _m.x; \
    ulong _th = _m.y + ((c0 < _m.x) ? 1UL : 0UL); \
    c1 += _th; \
    c2 += (c1 < _th) ? 1UL : 0UL; \
}

#define ACC_SUMADD_FAST(a) { \
    c0 += (a); \
    c1 += (c0 < (a)) ? 1UL : 0UL; \
}

#define ACC_SUMADD(a) { \
    c0 += (a); \
    ulong _over = (c0 < (a)) ? 1UL : 0UL; \
    c1 += _over; \
    c2 += (c1 < _over) ? 1UL : 0UL; \
}

#define ACC_EXTRACT(n) { \
    (n) = c0; \
    c0 = c1; \
    c1 = c2; \
    c2 = 0; \
}

#define ACC_EXTRACT_FAST(n) { \
    (n) = c0; \
    c0 = c1; \
    c1 = 0; \
}

FORCE_INLINE void scalar_reduce_512(Scalar* r, const ulong* l) {
    // nc = 2^256 - n (the complement)
    const ulong NC0 = 0x402DA1732FC9BEBFUL;  // ~N_LIMB0 + 1
    const ulong NC1 = 0x4551231950B75FC4UL;  // ~N_LIMB1

    ulong n0 = l[4], n1 = l[5], n2 = l[6], n3 = l[7];
    ulong m0, m1, m2, m3, m4, m5;
    ulong m6;
    ulong p0, p1, p2, p3;
    ulong p4;

    // --- Pass 1: Reduce 512 bits into 385 ---
    // m[0..6] = l[0..3] + n[0..3] * NC
    ulong c0, c1, c2;

    c0 = l[0]; c1 = 0; c2 = 0;
    ACC_MULADD_FAST(n0, NC0);
    ACC_EXTRACT_FAST(m0);
    ACC_SUMADD_FAST(l[1]);
    ACC_MULADD(n1, NC0);
    ACC_MULADD(n0, NC1);
    ACC_EXTRACT(m1);
    ACC_SUMADD(l[2]);
    ACC_MULADD(n2, NC0);
    ACC_MULADD(n1, NC1);
    ACC_SUMADD(n0);
    ACC_EXTRACT(m2);
    ACC_SUMADD(l[3]);
    ACC_MULADD(n3, NC0);
    ACC_MULADD(n2, NC1);
    ACC_SUMADD(n1);
    ACC_EXTRACT(m3);
    ACC_MULADD(n3, NC1);
    ACC_SUMADD(n2);
    ACC_EXTRACT(m4);
    ACC_SUMADD_FAST(n3);
    ACC_EXTRACT_FAST(m5);
    m6 = c0;

    // --- Pass 2: Reduce 385 bits into 258 ---
    // p[0..4] = m[0..3] + m[4..6] * NC
    c0 = m0; c1 = 0; c2 = 0;
    ACC_MULADD_FAST(m4, NC0);
    ACC_EXTRACT_FAST(p0);
    ACC_SUMADD_FAST(m1);
    ACC_MULADD(m5, NC0);
    ACC_MULADD(m4, NC1);
    ACC_EXTRACT(p1);
    ACC_SUMADD(m2);
    ACC_MULADD(m6, NC0);
    ACC_MULADD(m5, NC1);
    ACC_SUMADD(m4);
    ACC_EXTRACT(p2);
    ACC_SUMADD_FAST(m3);
    ACC_MULADD_FAST(m6, NC1);
    ACC_SUMADD_FAST(m5);
    ACC_EXTRACT_FAST(p3);
    p4 = c0 + m6;

    // --- Pass 3: Reduce 258 bits into 256 ---
    // r[0..3] = p[0..3] + p4 * NC
    // Emulate uint128_t with hi:lo pairs
    u64x2 t;
    ulong carry;

    t = mul64_full(p4, NC0);
    ulong r0 = p0 + t.x;
    carry = t.y + ((r0 < p0) ? 1UL : 0UL);

    t = mul64_full(p4, NC1);
    ulong sum1 = p1 + t.x;
    ulong c_1 = (sum1 < p1) ? 1UL : 0UL;
    ulong r1 = sum1 + carry;
    ulong c_2 = (r1 < sum1) ? 1UL : 0UL;
    carry = t.y + c_1 + c_2;

    ulong sum2 = p2 + p4;
    ulong c_3 = (sum2 < p2) ? 1UL : 0UL;
    ulong r2 = sum2 + carry;
    ulong c_4 = (r2 < sum2) ? 1UL : 0UL;
    carry = c_3 + c_4;

    ulong r3 = p3 + carry;
    // Detect ACTUAL carry from the addition (not a bit of the result!)
    // The old code used (r3 >> 63) which is WRONG: it checks bit 63, but
    // valid scalars in [2^255, n-1) have bit 63 set. That caused false
    // reductions (subtracting n from values < n), corrupting ~50% of results.
    ulong final_carry = (r3 < p3) ? 1UL : 0UL;

    r->limbs[0] = r0;
    r->limbs[1] = r1;
    r->limbs[2] = r2;
    r->limbs[3] = r3;

    // Reduce: matches CUDA's secp256k1_scalar_reduce(r, c + check_overflow(r))
    // If final_carry=1 or result >= n, subtract n. If both, subtract n twice.
    ulong red = final_carry + (ulong)(scalar_check_overflow(r) ? 1 : 0);
    while (red > 0) {
        ulong borrow = 0;
        r->limbs[0] = sub_with_borrow(r->limbs[0], N_LIMB0, 0, &borrow);
        r->limbs[1] = sub_with_borrow(r->limbs[1], N_LIMB1, borrow, &borrow);
        r->limbs[2] = sub_with_borrow(r->limbs[2], N_LIMB2, borrow, &borrow);
        r->limbs[3] = sub_with_borrow(r->limbs[3], N_LIMB3, borrow, &borrow);
        red--;
    }
}

// Scalar multiplication: r = (a * b) mod n
// Direct port of libsecp256k1's secp256k1_scalar_mul_512 + reduce.
// Uses column-accumulation with a 192-bit accumulator (c2:c1:c0) to avoid
// any possibility of carry overflow. This is the same approach as the CUDA miner.
FORCE_INLINE void scalar_mul_mod_n(Scalar* r, const Scalar* a, const Scalar* b) {
    // Always use the portable 4x64 Comba path. The 8x32 PTX engine was
    // producing wrong s values (r from k*G matched libsecp; s did not).
    ulong product[8];
    ulong c0, c1, c2;

    ulong a0 = a->limbs[0], a1 = a->limbs[1], a2 = a->limbs[2], a3 = a->limbs[3];
    ulong b0 = b->limbs[0], b1 = b->limbs[1], b2 = b->limbs[2], b3 = b->limbs[3];

    c0 = 0; c1 = 0; c2 = 0;
    ACC_MULADD_FAST(a0, b0); ACC_EXTRACT_FAST(product[0]);
    ACC_MULADD(a0, b1); ACC_MULADD(a1, b0); ACC_EXTRACT(product[1]);
    ACC_MULADD(a0, b2); ACC_MULADD(a1, b1); ACC_MULADD(a2, b0); ACC_EXTRACT(product[2]);
    ACC_MULADD(a0, b3); ACC_MULADD(a1, b2); ACC_MULADD(a2, b1); ACC_MULADD(a3, b0); ACC_EXTRACT(product[3]);
    ACC_MULADD(a1, b3); ACC_MULADD(a2, b2); ACC_MULADD(a3, b1); ACC_EXTRACT(product[4]);
    ACC_MULADD(a2, b3); ACC_MULADD(a3, b2); ACC_EXTRACT(product[5]);
    ACC_MULADD_FAST(a3, b3); ACC_EXTRACT_FAST(product[6]);
    product[7] = c0;
    scalar_reduce_512(r, product);
}


// v59: fused scalar multiply-add for ECDSA finish.
// Computes r = (a*b + c) mod n with a single 512->256 reduction.
// On CUDA the 256-bit add is folded directly into the 16x32 product with
// one PTX carry chain, avoiding a reduced rd temporary + scalar_add_mod_n().
FORCE_INLINE void scalar_muladd_mod_n(Scalar* r, const Scalar* a, const Scalar* b, const Scalar* c) {
#ifdef __NV_CL_C_VERSION
    FieldElement aa, bb;
    aa.limbs[0] = a->limbs[0]; aa.limbs[1] = a->limbs[1];
    aa.limbs[2] = a->limbs[2]; aa.limbs[3] = a->limbs[3];
    bb.limbs[0] = b->limbs[0]; bb.limbs[1] = b->limbs[1];
    bb.limbs[2] = b->limbs[2]; bb.limbs[3] = b->limbs[3];

    uint t32[16];
    mul_256_comba32_ocl(&aa, &bb, t32);

    const uint c0=(uint)c->limbs[0], c1=(uint)(c->limbs[0]>>32);
    const uint c2=(uint)c->limbs[1], c3=(uint)(c->limbs[1]>>32);
    const uint c4=(uint)c->limbs[2], c5=(uint)(c->limbs[2]>>32);
    const uint c6=(uint)c->limbs[3], c7=(uint)(c->limbs[3]>>32);
    const uint z = 0U;
    asm volatile(
        "add.cc.u32      %0,  %0,  %16;\n\t"
        "addc.cc.u32     %1,  %1,  %17;\n\t"
        "addc.cc.u32     %2,  %2,  %18;\n\t"
        "addc.cc.u32     %3,  %3,  %19;\n\t"
        "addc.cc.u32     %4,  %4,  %20;\n\t"
        "addc.cc.u32     %5,  %5,  %21;\n\t"
        "addc.cc.u32     %6,  %6,  %22;\n\t"
        "addc.cc.u32     %7,  %7,  %23;\n\t"
        "addc.cc.u32     %8,  %8,  %24;\n\t"
        "addc.cc.u32     %9,  %9,  %24;\n\t"
        "addc.cc.u32    %10, %10,  %24;\n\t"
        "addc.cc.u32    %11, %11,  %24;\n\t"
        "addc.cc.u32    %12, %12,  %24;\n\t"
        "addc.cc.u32    %13, %13,  %24;\n\t"
        "addc.cc.u32    %14, %14,  %24;\n\t"
        "addc.u32       %15, %15,  %24;\n\t"
        : "+r"(t32[0]), "+r"(t32[1]), "+r"(t32[2]), "+r"(t32[3]),
          "+r"(t32[4]), "+r"(t32[5]), "+r"(t32[6]), "+r"(t32[7]),
          "+r"(t32[8]), "+r"(t32[9]), "+r"(t32[10]), "+r"(t32[11]),
          "+r"(t32[12]), "+r"(t32[13]), "+r"(t32[14]), "+r"(t32[15])
        : "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4), "r"(c5), "r"(c6), "r"(c7), "r"(z)
    );

    ulong product[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        product[i] = (ulong)t32[2*i] | ((ulong)t32[2*i + 1] << 32);
    }
    scalar_reduce_512(r, product);
#else
    Scalar tmp;
    scalar_mul_mod_n(&tmp, a, b);
    scalar_add_mod_n(r, &tmp, c);
#endif
}

// Scalar squaring: r = a² mod n
// Uses the same 192-bit accumulator approach as scalar_mul_mod_n.
// Simply calls mul with both operands the same — the compiler can optimize.
FORCE_INLINE void scalar_sqr_mod_n(Scalar* r, const Scalar* a) {
    scalar_mul_mod_n(r, a, a);
}

// Fermat k^{-1} = k^{n-2} mod n. safegcd inverse is still wrong on this toolchain.
static __device__ __noinline__ void scalar_inverse_fermat(Scalar* r, const Scalar* a)
{
    Scalar base = *a;
    Scalar acc;
    acc.limbs[0] = 1UL;
    acc.limbs[1] = 0UL;
    acc.limbs[2] = 0UL;
    acc.limbs[3] = 0UL;
    const ulong exp[4] = {
        0xBFD25E8CD036413FUL,
        0xBAAEDCE6AF48A03BUL,
        0xFFFFFFFFFFFFFFFEUL,
        0xFFFFFFFFFFFFFFFFUL
    };
#pragma unroll 1
    for (int i = 0; i < 256; ++i) {
        const uint limb = (uint)i >> 6;
        const uint shift = (uint)i & 63u;
        if ((exp[limb] >> shift) & 1UL) {
            Scalar tmp;
            scalar_mul_mod_n(&tmp, &acc, &base);
            acc = tmp;
        }
        if (i != 255) {
            Scalar sq;
            scalar_sqr_mod_n(&sq, &base);
            base = sq;
        }
    }
    *r = acc;
}

// =============================================================================
// Scalar inversion mod n — Direct port of libsecp256k1 modinv64 (safegcd)
// =============================================================================
// This is the SAME algorithm used by the CUDA miner. It uses the safegcd
// (divsteps) algorithm which only needs basic integer arithmetic (add/sub/shift),
// NOT exponentiation. This avoids the 255-iteration multiply loop that was
// causing OpenCL compiler issues.
//
// Requires: 128-bit signed integer emulation (since OpenCL has no int128_t).

// --- Signed 128-bit integer emulation ---
typedef struct { long lo; long hi; } s128;  // signed 128-bit as (hi:lo)

FORCE_INLINE s128 s128_mul(long a, long b) {
    // Signed 64x64→128 multiply
    // Split into unsigned multiply + sign correction
    ulong au = (ulong)a, bu = (ulong)b;
    ulong lo = au * bu;
    long hi = (long)mul_hi(au, bu);
    // Sign correction: if a < 0, subtract b from high; if b < 0, subtract a from high
    if (a < 0) hi -= b;
    if (b < 0) hi -= a;
    s128 r; r.lo = (long)lo; r.hi = hi;
    return r;
}

FORCE_INLINE void s128_accum_mul(s128* acc, long a, long b) {
    s128 prod = s128_mul(a, b);
    ulong old_lo = (ulong)acc->lo;
    acc->lo += prod.lo;
    ulong carry = ((ulong)acc->lo < old_lo) ? 1UL : 0UL;
    acc->hi += prod.hi + (long)carry;
}

FORCE_INLINE void s128_rshift(s128* r, int n) {
    // Arithmetic right shift by n (0 < n < 64)
    r->lo = (long)(((ulong)r->lo >> n) | ((ulong)r->hi << (64 - n)));
    r->hi = r->hi >> n;  // arithmetic shift
}

FORCE_INLINE ulong s128_to_u64(const s128* a) { return (ulong)a->lo; }
FORCE_INLINE long  s128_to_i64(const s128* a) { return a->lo; }

// --- Signed62 representation (5 limbs of 62 bits each) ---
typedef struct { long v[5]; } Signed62;

typedef struct { long u, v, q, r; } Trans2x2;

typedef struct {
    Signed62 modulus;
    ulong modulus_inv62;
} ModInfo;

// n in signed62 limbs
// n = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
// Converted to signed62:
//   v[0] = 0x3FD25E8CD0364141  (low 62 bits of n)
//   v[1] = 0x2ABB739ABD2280EE
//   v[2] = -0x15  = 0xFFFFFFFFFFFFFFEB (as signed)
//   v[3] = 0
//   v[4] = 256  (just encodes the bit length)
// modulus_inv62 = 0x34F20099AA774EC1

FORCE_INLINE long modinv64_divsteps_59(long zeta, ulong f0, ulong g0, Trans2x2* t) {
    ulong u = 8, v = 0, q = 0, r = 8;
    ulong f = f0, g = g0, x, y, z;
    ulong mask1, mask2;
    long c1;
    ulong c2;

    for (int i = 3; i < 62; ++i) {
        c1 = zeta >> 63;
        mask1 = (ulong)c1;
        c2 = g & 1;
        mask2 = (ulong)(-(long)c2);
        x = (f ^ mask1) - mask1;
        y = (u ^ mask1) - mask1;
        z = (v ^ mask1) - mask1;
        g += x & mask2;
        q += y & mask2;
        r += z & mask2;
        mask1 &= mask2;
        zeta = (zeta ^ (long)mask1) - 1;
        f += g & mask1;
        u += q & mask1;
        v += r & mask1;
        g >>= 1;
        u <<= 1;
        v <<= 1;
    }
    t->u = (long)u;
    t->v = (long)v;
    t->q = (long)q;
    t->r = (long)r;
    return zeta;
}

FORCE_INLINE void modinv64_update_de_62(Signed62* d, Signed62* e, const Trans2x2* t,
                                   long mod0, long mod1, long mod2, long mod3, long mod4,
                                   ulong mod_inv62) {
    const ulong M62 = 0x3FFFFFFFFFFFFFFFUL;
    const long d0=d->v[0], d1=d->v[1], d2=d->v[2], d3=d->v[3], d4=d->v[4];
    const long e0=e->v[0], e1=e->v[1], e2=e->v[2], e3=e->v[3], e4=e->v[4];
    const long u=t->u, v_=t->v, q=t->q, r_=t->r;
    long md, me;
    long sd = d4 >> 63;
    long se = e4 >> 63;
    md = (u & sd) + (v_ & se);
    me = (q & sd) + (r_ & se);
    s128 cd = s128_mul(u, d0); s128_accum_mul(&cd, v_, e0);
    s128 ce = s128_mul(q, d0); s128_accum_mul(&ce, r_, e0);
    md -= (long)((mod_inv62 * s128_to_u64(&cd) + (ulong)md) & M62);
    me -= (long)((mod_inv62 * s128_to_u64(&ce) + (ulong)me) & M62);
    s128_accum_mul(&cd, mod0, md);
    s128_accum_mul(&ce, mod0, me);
    s128_rshift(&cd, 62);
    s128_rshift(&ce, 62);

    s128_accum_mul(&cd, u, d1); s128_accum_mul(&cd, v_, e1);
    s128_accum_mul(&ce, q, d1); s128_accum_mul(&ce, r_, e1);
    if (mod1) { s128_accum_mul(&cd, mod1, md); s128_accum_mul(&ce, mod1, me); }
    d->v[0] = (long)(s128_to_u64(&cd) & M62); s128_rshift(&cd, 62);
    e->v[0] = (long)(s128_to_u64(&ce) & M62); s128_rshift(&ce, 62);

    s128_accum_mul(&cd, u, d2); s128_accum_mul(&cd, v_, e2);
    s128_accum_mul(&ce, q, d2); s128_accum_mul(&ce, r_, e2);
    if (mod2) { s128_accum_mul(&cd, mod2, md); s128_accum_mul(&ce, mod2, me); }
    d->v[1] = (long)(s128_to_u64(&cd) & M62); s128_rshift(&cd, 62);
    e->v[1] = (long)(s128_to_u64(&ce) & M62); s128_rshift(&ce, 62);

    s128_accum_mul(&cd, u, d3); s128_accum_mul(&cd, v_, e3);
    s128_accum_mul(&ce, q, d3); s128_accum_mul(&ce, r_, e3);
    if (mod3) { s128_accum_mul(&cd, mod3, md); s128_accum_mul(&ce, mod3, me); }
    d->v[2] = (long)(s128_to_u64(&cd) & M62); s128_rshift(&cd, 62);
    e->v[2] = (long)(s128_to_u64(&ce) & M62); s128_rshift(&ce, 62);

    s128_accum_mul(&cd, u, d4); s128_accum_mul(&cd, v_, e4);
    s128_accum_mul(&ce, q, d4); s128_accum_mul(&ce, r_, e4);
    s128_accum_mul(&cd, mod4, md);
    s128_accum_mul(&ce, mod4, me);
    d->v[3] = (long)(s128_to_u64(&cd) & M62); s128_rshift(&cd, 62);
    e->v[3] = (long)(s128_to_u64(&ce) & M62); s128_rshift(&ce, 62);

    d->v[4] = s128_to_i64(&cd);
    e->v[4] = s128_to_i64(&ce);
}

FORCE_INLINE void modinv64_update_fg_62(Signed62* f, Signed62* g, const Trans2x2* t) {
    const ulong M62 = 0x3FFFFFFFFFFFFFFFUL;
    const long f0=f->v[0], f1=f->v[1], f2=f->v[2], f3=f->v[3], f4=f->v[4];
    const long g0=g->v[0], g1=g->v[1], g2=g->v[2], g3=g->v[3], g4=g->v[4];
    const long u=t->u, v_=t->v, q=t->q, r_=t->r;
    s128 cf = s128_mul(u, f0); s128_accum_mul(&cf, v_, g0);
    s128 cg = s128_mul(q, f0); s128_accum_mul(&cg, r_, g0);
    s128_rshift(&cf, 62);
    s128_rshift(&cg, 62);

    s128_accum_mul(&cf, u, f1); s128_accum_mul(&cf, v_, g1);
    s128_accum_mul(&cg, q, f1); s128_accum_mul(&cg, r_, g1);
    f->v[0] = (long)(s128_to_u64(&cf) & M62); s128_rshift(&cf, 62);
    g->v[0] = (long)(s128_to_u64(&cg) & M62); s128_rshift(&cg, 62);

    s128_accum_mul(&cf, u, f2); s128_accum_mul(&cf, v_, g2);
    s128_accum_mul(&cg, q, f2); s128_accum_mul(&cg, r_, g2);
    f->v[1] = (long)(s128_to_u64(&cf) & M62); s128_rshift(&cf, 62);
    g->v[1] = (long)(s128_to_u64(&cg) & M62); s128_rshift(&cg, 62);

    s128_accum_mul(&cf, u, f3); s128_accum_mul(&cf, v_, g3);
    s128_accum_mul(&cg, q, f3); s128_accum_mul(&cg, r_, g3);
    f->v[2] = (long)(s128_to_u64(&cf) & M62); s128_rshift(&cf, 62);
    g->v[2] = (long)(s128_to_u64(&cg) & M62); s128_rshift(&cg, 62);

    s128_accum_mul(&cf, u, f4); s128_accum_mul(&cf, v_, g4);
    s128_accum_mul(&cg, q, f4); s128_accum_mul(&cg, r_, g4);
    f->v[3] = (long)(s128_to_u64(&cf) & M62); s128_rshift(&cf, 62);
    g->v[3] = (long)(s128_to_u64(&cg) & M62); s128_rshift(&cg, 62);

    f->v[4] = s128_to_i64(&cf);
    g->v[4] = s128_to_i64(&cg);
}

FORCE_INLINE void modinv64_normalize_62(Signed62* r, long sign,
                                   long mod0, long mod1, long mod2, long mod3, long mod4) {
    const long M62 = (long)0x3FFFFFFFFFFFFFFFUL;
    long r0=r->v[0], r1=r->v[1], r2=r->v[2], r3=r->v[3], r4=r->v[4];
    long cond_add, cond_negate;

    // Step 1: if negative, add modulus; then negate if requested
    cond_add = r4 >> 63;
    r0 += mod0 & cond_add;
    r1 += mod1 & cond_add;
    r2 += mod2 & cond_add;
    r3 += mod3 & cond_add;
    r4 += mod4 & cond_add;
    cond_negate = sign >> 63;
    r0 = (r0 ^ cond_negate) - cond_negate;
    r1 = (r1 ^ cond_negate) - cond_negate;
    r2 = (r2 ^ cond_negate) - cond_negate;
    r3 = (r3 ^ cond_negate) - cond_negate;
    r4 = (r4 ^ cond_negate) - cond_negate;
    // Propagate
    r1 += r0 >> 62; r0 &= M62;
    r2 += r1 >> 62; r1 &= M62;
    r3 += r2 >> 62; r2 &= M62;
    r4 += r3 >> 62; r3 &= M62;

    // Step 2: if still negative, add modulus again
    cond_add = r4 >> 63;
    r0 += mod0 & cond_add;
    r1 += mod1 & cond_add;
    r2 += mod2 & cond_add;
    r3 += mod3 & cond_add;
    r4 += mod4 & cond_add;
    r1 += r0 >> 62; r0 &= M62;
    r2 += r1 >> 62; r1 &= M62;
    r3 += r2 >> 62; r2 &= M62;
    r4 += r3 >> 62; r3 &= M62;

    r->v[0]=r0; r->v[1]=r1; r->v[2]=r2; r->v[3]=r3; r->v[4]=r4;
}

FORCE_INLINE void scalar_inverse_mod_n(Scalar* r, const Scalar* a) {
    const ulong M62 = 0x3FFFFFFFFFFFFFFFUL;

    // Modulus constants (n in signed62)
    const long mod0 = 0x3FD25E8CD0364141LL;
    const long mod1 = 0x2ABB739ABD2280EELL;
    const long mod2 = -0x15LL;
    const long mod3 = 0LL;
    const long mod4 = 256LL;
    const ulong mod_inv62 = 0x34F20099AA774EC1UL;

    // Convert scalar to signed62
    ulong a0 = a->limbs[0], a1 = a->limbs[1], a2 = a->limbs[2], a3 = a->limbs[3];
    Signed62 s;
    s.v[0] = (long)( a0                  & M62);
    s.v[1] = (long)((a0 >> 62 | a1 << 2) & M62);
    s.v[2] = (long)((a1 >> 60 | a2 << 4) & M62);
    s.v[3] = (long)((a2 >> 58 | a3 << 6) & M62);
    s.v[4] = (long)( a3 >> 56);

    // Run modinv64: d=0, e=1, f=modulus, g=x, zeta=-1
    Signed62 d; d.v[0]=0; d.v[1]=0; d.v[2]=0; d.v[3]=0; d.v[4]=0;
    Signed62 e; e.v[0]=1; e.v[1]=0; e.v[2]=0; e.v[3]=0; e.v[4]=0;
    Signed62 f; f.v[0]=mod0; f.v[1]=mod1; f.v[2]=mod2; f.v[3]=mod3; f.v[4]=mod4;
    Signed62 g = s;
    long zeta = -1L;

    // 10 iterations of 59 divsteps = 590 total (sufficient for 256-bit)
    for (int i = 0; i < 10; i++) {
        Trans2x2 t;
        zeta = modinv64_divsteps_59(zeta, (ulong)f.v[0], (ulong)g.v[0], &t);
        modinv64_update_de_62(&d, &e, &t, mod0, mod1, mod2, mod3, mod4, mod_inv62);
        modinv64_update_fg_62(&f, &g, &t);
    }

    // Normalize and convert back
    modinv64_normalize_62(&d, f.v[4], mod0, mod1, mod2, mod3, mod4);

    // Convert signed62 back to scalar (4 × 64-bit limbs)
    ulong d0 = (ulong)d.v[0], d1 = (ulong)d.v[1], d2 = (ulong)d.v[2];
    ulong d3 = (ulong)d.v[3], d4 = (ulong)d.v[4];
    r->limbs[0] = d0      | d1 << 62;
    r->limbs[1] = d1 >> 2 | d2 << 60;
    r->limbs[2] = d2 >> 4 | d3 << 58;
    r->limbs[3] = d3 >> 6 | d4 << 56;
}

// Scalar negation: r = n - a
FORCE_INLINE void scalar_negate(Scalar* r, const Scalar* a) {
    if (scalar_is_zero(a)) {
        *r = *a;
        return;
    }
    ulong borrow = 0;
    r->limbs[0] = sub_with_borrow(N_LIMB0, a->limbs[0], 0, &borrow);
    r->limbs[1] = sub_with_borrow(N_LIMB1, a->limbs[1], borrow, &borrow);
    r->limbs[2] = sub_with_borrow(N_LIMB2, a->limbs[2], borrow, &borrow);
    r->limbs[3] = sub_with_borrow(N_LIMB3, a->limbs[3], borrow, &borrow);
}

// Check if scalar is "high" (s > n/2)
FORCE_INLINE int scalar_is_high(const Scalar* s) {
    // n/2 = 7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0
    // Compare s > n/2
    Scalar half_n;
    half_n.limbs[3] = 0x7FFFFFFFFFFFFFFFUL;
    half_n.limbs[2] = 0xFFFFFFFFFFFFFFFFUL;
    half_n.limbs[1] = 0x5D576E7357A4501DUL;
    half_n.limbs[0] = 0xDFE92F46681B20A0UL;
    
    if (s->limbs[3] > half_n.limbs[3]) return 1;
    if (s->limbs[3] < half_n.limbs[3]) return 0;
    if (s->limbs[2] > half_n.limbs[2]) return 1;
    if (s->limbs[2] < half_n.limbs[2]) return 0;
    if (s->limbs[1] > half_n.limbs[1]) return 1;
    if (s->limbs[1] < half_n.limbs[1]) return 0;
    if (s->limbs[0] > half_n.limbs[0]) return 1;
    return 0;
}

// =============================================================================
// ECDSA Signing
// =============================================================================
// Given: secret key, message hash (both as 32-byte big-endian)
// Returns: DER-encoded signature in sig_out, length in sig_len
// Uses RFC 6979 deterministic nonce

// DER encode a scalar (big-endian 32 bytes) into DER integer format
// Returns number of bytes written
FORCE_INLINE int der_encode_integer(uchar* out, const uchar* val32) {
    int start = 0;
    // Skip leading zeros
    while (start < 32 && val32[start] == 0) start++;
    if (start == 32) {
        // Zero value
        out[0] = 0x02; out[1] = 0x01; out[2] = 0x00;
        return 3;
    }
    int need_pad = (val32[start] & 0x80) ? 1 : 0;
    int len = 32 - start + need_pad;
    out[0] = 0x02;
    out[1] = (uchar)len;
    int pos = 2;
    if (need_pad) out[pos++] = 0x00;
    for (int i = start; i < 32; i++) out[pos++] = val32[i];
    return pos;
}

// Full DER signature encoding: 0x30 <total_len> <r_der> <s_der>
FORCE_INLINE int der_encode_signature(uchar* out, const uchar* r32, const uchar* s32) {
    uchar r_der[35], s_der[35];
    int r_len = der_encode_integer(r_der, r32);
    int s_len = der_encode_integer(s_der, s32);
    
    out[0] = 0x30;
    out[1] = (uchar)(r_len + s_len);
    int pos = 2;
    for (int i = 0; i < r_len; i++) out[pos++] = r_der[i];
    for (int i = 0; i < s_len; i++) out[pos++] = s_der[i];
    return pos;
}

// =============================================================================
// Precomputed Generator Table - 20-bit windowed lookup for k*G
// =============================================================================
// Table layout: 13 groups x 1048576 entries = 13631488 affine points
//   table[group * 1048576 + value] = value * ((2^20)^group) * G
//   Each entry: 8 x uint64 (x[4 LE limbs] + y[4 LE limbs]) = 64 bytes
//   Entry for value=0 is unused (infinity); we skip it during lookup.
// Total: 13631488 * 64 = 872415232 bytes = 109051904 ulong values
//
// This replaces the per-thread wNAF computation (~256 doubles + ~85 adds)
// with ~96 mixed affine-Jacobian additions (no doubles at all).
// =============================================================================

// Load an affine point from the precomputed table buffer
FORCE_INLINE void ecmult_table_load_point(AffinePoint* p, const ulong* table, int index) {
    int off = index * 8;  // 8 ulongs per entry
    p->x.limbs[0] = table[off + 0];
    p->x.limbs[1] = table[off + 1];
    p->x.limbs[2] = table[off + 2];
    p->x.limbs[3] = table[off + 3];
    p->y.limbs[0] = table[off + 4];
    p->y.limbs[1] = table[off + 5];
    p->y.limbs[2] = table[off + 6];
    p->y.limbs[3] = table[off + 7];
}


/* ============================================================================
 * XYZZ fixed-base accumulator for W20 mining
 *
 * Coordinates:
 *   affine x = X / ZZ
 *   affine y = Y / ZZZ
 * where ZZ=Z^2 and ZZZ=Z^3.
 *
 * Mixed affine add avoids recomputing Z^2/Z^3 every window:
 *   U2   = x2 * ZZ
 *   S2   = y2 * ZZZ
 *   P    = U2 - X
 *   R    = S2 - Y
 *   PP   = P^2
 *   PPP  = P*PP
 *   Q    = X*PP
 *   X3   = R^2 - PPP - 2Q
 *   Y3   = R*(Q-X3) - Y*PPP
 *   ZZ3  = ZZ*PP
 *   ZZZ3 = ZZZ*PPP
 *
 * Compared with the Jacobian mixed-add hot path this trades one multiply for
 * two fewer squarings per addition.  It also outputs ZZ directly, so affine
 * X normalization after the batch inversion needs only X * ZZ^-1.
 * ========================================================================== */

typedef struct {
    FieldElement x;
    FieldElement y;
    FieldElement zz;
    FieldElement zzz;
    int infinity;
} XYZZPoint;

FORCE_INLINE void xyzz_from_affine(XYZZPoint* r, const AffinePoint* a) {
    r->x = a->x;
    r->y = a->y;
    r->zz.limbs[0] = 1UL;
    r->zz.limbs[1] = 0UL;
    r->zz.limbs[2] = 0UL;
    r->zz.limbs[3] = 0UL;
    r->zzz = r->zz;
    r->infinity = 0;
}

FORCE_INLINE void xyzz_add_mixed_unchecked(
    XYZZPoint* r,
    const XYZZPoint* p,
    const AffinePoint* q)
{
    /*
      Lower-live-range XYZZ mixed addition.

      We aggressively reuse temporaries so fewer FieldElement values remain
      simultaneously live.  This is aimed at NVIDIA register/private-memory
      pressure as much as arithmetic count.
    */
    FieldElement P, Rv, PP, PPP, Q, t0, t1;

    /* P = x2*ZZ - X */
    field_mul_impl(&P, &q->x, &p->zz);
    field_sub_impl(&P, &P, &p->x);

    /* R = y2*ZZZ - Y */
    field_mul_impl(&Rv, &q->y, &p->zzz);
    field_sub_impl(&Rv, &Rv, &p->y);

    /* PP=P^2, PPP=P^3, Q=X*PP */
    field_sqr_impl(&PP, &P);
    field_mul_impl(&PPP, &P, &PP);
    field_mul_impl(&Q, &p->x, &PP);

    /* X3 = R^2 - PPP - 2Q */
    field_sqr_impl(&t0, &Rv);
    field_sub_impl(&t0, &t0, &PPP);
    field_add_impl(&t1, &Q, &Q);
    field_sub_impl(&t0, &t0, &t1);
    r->x = t0;

    /* Y3 = R*(Q-X3) - Y*PPP */
    field_sub_impl(&t0, &Q, &r->x);
    field_mul_impl(&t0, &Rv, &t0);
    field_mul_impl(&t1, &p->y, &PPP);
    field_sub_impl(&r->y, &t0, &t1);

    /* ZZ3 = ZZ*PP */
    field_mul_impl(&r->zz, &p->zz, &PP);

    /* ZZZ3 = ZZZ*PPP */
    field_mul_impl(&r->zzz, &p->zzz, &PPP);

    r->infinity = 0;
}


FORCE_INLINE void ecmult_table_load_point_xyzz(
    AffinePoint* p,
    const ulong* table,
    int index)
{
    const int off = index << 3;
    ulong x0 = table[off + 0];
    ulong x1 = table[off + 1];
    ulong x2 = table[off + 2];
    ulong x3 = table[off + 3];
    ulong y0 = table[off + 4];
    ulong y1 = table[off + 5];
    ulong y2 = table[off + 6];
    ulong y3 = table[off + 7];

    p->x.limbs[0] = x0; p->x.limbs[1] = x1;
    p->x.limbs[2] = x2; p->x.limbs[3] = x3;
    p->y.limbs[0] = y0; p->y.limbs[1] = y1;
    p->y.limbs[2] = y2; p->y.limbs[3] = y3;
}


/* --------------------------------------------------------------------------
 * secp256k1 GLV split + shared W26 table
 *
 * k = r1 + lambda*r2 (mod n), with |r1|,|r2| < 2^128.
 * The same 5 fixed-base tables are used for both halves.  r2*lambda*G is
 * obtained through the curve endomorphism phi(x,y)=(beta*x,y), so a second
 * 9 GiB table is NOT required.
 *
 * Window offsets are 0,26,52,78,104.  The last window includes an implicit
 * zero bit 128, which absorbs the signed carry without a sixth point.
 * Thus each half contributes at most 5 affine points, so the combined k*G
 * needs at most 9 XYZZ mixed additions.
 * -------------------------------------------------------------------------- */

FORCE_INLINE void scalar_mul_shift_384_round(Scalar* r, const Scalar* a, const Scalar* b)
{
    ulong product[8];
    ulong c0=0UL,c1=0UL,c2=0UL;
    ulong a0=a->limbs[0], a1=a->limbs[1], a2=a->limbs[2], a3=a->limbs[3];
    ulong b0=b->limbs[0], b1=b->limbs[1], b2=b->limbs[2], b3=b->limbs[3];

    ACC_MULADD_FAST(a0,b0); ACC_EXTRACT_FAST(product[0]);
    ACC_MULADD(a0,b1); ACC_MULADD(a1,b0); ACC_EXTRACT(product[1]);
    ACC_MULADD(a0,b2); ACC_MULADD(a1,b1); ACC_MULADD(a2,b0); ACC_EXTRACT(product[2]);
    ACC_MULADD(a0,b3); ACC_MULADD(a1,b2); ACC_MULADD(a2,b1); ACC_MULADD(a3,b0); ACC_EXTRACT(product[3]);
    ACC_MULADD(a1,b3); ACC_MULADD(a2,b2); ACC_MULADD(a3,b1); ACC_EXTRACT(product[4]);
    ACC_MULADD(a2,b3); ACC_MULADD(a3,b2); ACC_EXTRACT(product[5]);
    ACC_MULADD_FAST(a3,b3); ACC_EXTRACT_FAST(product[6]);
    product[7]=c0;

    /* Round (a*b)/2^384 to nearest integer. Result is at most 128 bits. */
    ulong lo=product[6], hi=product[7];
    const ulong round=(product[5] >> 63) & 1UL;
    if (round) {
        lo += 1UL;
        if (lo == 0UL) hi += 1UL;
    }
    r->limbs[0]=lo; r->limbs[1]=hi; r->limbs[2]=0UL; r->limbs[3]=0UL;
}

FORCE_INLINE void scalar_split_lambda_glv(Scalar* r1, Scalar* r2, const Scalar* k)
{
    const Scalar minus_b1={{0x6F547FA90ABFE4C3UL,0xE4437ED6010E8828UL,0UL,0UL}};
    const Scalar minus_b2={{0xD765CDA83DB1562CUL,0x8A280AC50774346DUL,0xFFFFFFFFFFFFFFFEUL,0xFFFFFFFFFFFFFFFFUL}};
    const Scalar g1={{0xE893209A45DBB031UL,0x3DAA8A1471E8CA7FUL,0xE86C90E49284EB15UL,0x3086D221A7D46BCDUL}};
    const Scalar g2={{0x1571B4AE8AC47F71UL,0x221208AC9DF506C6UL,0x6F547FA90ABFE4C4UL,0xE4437ED6010E8828UL}};
    const Scalar lambda={{0xDF02967C1B23BD72UL,0x122E22EA20816678UL,0xA5261C028812645AUL,0x5363AD4CC05C30E0UL}};
    Scalar c1s,c2s,t;

    scalar_mul_shift_384_round(&c1s,k,&g1);
    scalar_mul_shift_384_round(&c2s,k,&g2);
    scalar_mul_mod_n(&c1s,&c1s,&minus_b1);
    scalar_mul_mod_n(&c2s,&c2s,&minus_b2);
    scalar_add_mod_n(r2,&c1s,&c2s);

    scalar_mul_mod_n(&t,r2,&lambda);
    scalar_negate(&t,&t);
    scalar_add_mod_n(r1,&t,k);
}

/* Convert split scalar modulo n to its <=128-bit absolute representative. */
FORCE_INLINE uint scalar_glv_abs128(Scalar* s)
{
    uint neg=(s->limbs[2] | s->limbs[3]) != 0UL;
    if (neg) {
        Scalar t;
        scalar_negate(&t,s);
        *s=t;
    }
    return neg;
}

FORCE_INLINE void field_mul_beta(FieldElement* x)
{
    /* beta = 0x7AE96A2B657C07106E64479EAC3434E99CF0497512F58995C1396C28719501EE */
    const FieldElement beta={{0xC1396C28719501EEUL,0x9CF0497512F58995UL,
                              0x6E64479EAC3434E9UL,0x7AE96A2B657C0710UL}};
    FieldElement t;
    field_mul_impl(&t,x,&beta);
    *x=t;
}

FORCE_INLINE uint scalar_extract_bits_128(const Scalar* s, uint bit, uint width)
{
    const uint limb=bit>>6;
    const uint shift=bit&63u;
    ulong raw=s->limbs[limb] >> shift;
    if (shift && (shift + width > 64u) && limb < 1u) {
        raw |= s->limbs[limb+1u] << (64u-shift);
    }
    return (uint)(raw & ((1UL<<width)-1UL));
}

FORCE_INLINE void glv_accum_half(
    XYZZPoint* acc, int* started, const Scalar* abs_s, uint scalar_neg, uint use_phi,
    const ulong* table0, const ulong* table1,
    const ulong* table2, const ulong* table3,
    const ulong* table4, const ulong* table5)
{
    const uint offsets[6]={0u,24u,48u,72u,96u,120u};
    const uint widths [6]={24u,24u,24u,24u,24u,9u};
    const ulong* tables[6]={table0,table1,table2,table3,table4,table5};
    uint carry=0u;

#pragma unroll
    for (int group=0; group<6; ++group) {
        uint w=widths[group];
        uint raw=scalar_extract_bits_128(abs_s,offsets[group],w);

        /* group 5 spans bits 120..128. */

        uint u=raw+carry;
        uint halfv=1u<<(w-1u);
        uint full=1u<<w;
        uint digit_neg=0u, mag;
        if (u>halfv) { mag=full-u; carry=1u; digit_neg=1u; }
        else { mag=u; carry=0u; }

        if (!mag) continue;

        AffinePoint ap;
        ecmult_table_load_point_xyzz(&ap,tables[group],(int)(mag-1u));
        if (use_phi) field_mul_beta(&ap.x);

        if (scalar_neg ^ digit_neg) {
            FieldElement zero={{0UL,0UL,0UL,0UL}};
            field_sub_impl(&ap.y,&zero,&ap.y);
        }

        if (!*started) {
            xyzz_from_affine(acc,&ap);
            *started=1;
        } else {
            XYZZPoint next;
            xyzz_add_mixed_unchecked(&next,acc,&ap);
            *acc=next;
        }
    }
}

static __device__ __noinline__ void scalar_mul_generator_double_add(
    FieldElement* out_x, FieldElement* out_zz, const Scalar* k)
{
    AffinePoint gen;
    get_generator(&gen);
    JacobianPoint acc;
    int started = 0;
#pragma unroll 1
    for (int bit = 255; bit >= 0; --bit) {
        if (started) {
            JacobianPoint doubled;
            point_double_impl(&doubled, &acc);
            acc = doubled;
        }
        const uint limb = (uint)bit >> 6;
        const uint shift = (uint)bit & 63u;
        if ((k->limbs[limb] >> shift) & 1UL) {
            if (!started) {
                point_from_affine(&acc, &gen);
                started = 1;
            } else {
                JacobianPoint added;
                point_add_mixed_impl(&added, &acc, &gen);
                acc = added;
            }
        }
    }
    if (!started) {
        for (int i = 0; i < 4; ++i) {
            out_x->limbs[i] = 0UL;
            out_zz->limbs[i] = 0UL;
        }
        return;
    }
    *out_x = acc.x;
    field_sqr_impl(out_zz, &acc.z);
}

FORCE_INLINE void scalar_mul_generator_precomp_xyzz(
    FieldElement* out_x, FieldElement* out_zz, const Scalar* k,
    const ulong* table0, const ulong* table1,
    const ulong* table2, const ulong* table3,
    const ulong* table4, const ulong* table5)
{
    if (!table0) {
        scalar_mul_generator_double_add(out_x, out_zz, k);
        return;
    }
    Scalar r1,r2;
    scalar_split_lambda_glv(&r1,&r2,k);
    uint neg1=scalar_glv_abs128(&r1);
    uint neg2=scalar_glv_abs128(&r2);

    XYZZPoint acc;
    int started=0;
    glv_accum_half(&acc,&started,&r1,neg1,0u,table0,table1,table2,table3,table4,table5);
    glv_accum_half(&acc,&started,&r2,neg2,1u,table0,table1,table2,table3,table4,table5);

    if (!started) {
        for(int i=0;i<4;++i){out_x->limbs[i]=0UL;out_zz->limbs[i]=0UL;}
        return;
    }
    *out_x=acc.x;
    *out_zz=acc.zz;
}

// Fast scalar multiplication k*G using precomputed 20-bit windowed table
// Uses 13 iterations of table lookup + mixed add
FORCE_INLINE void scalar_mul_generator_precomp(JacobianPoint* r, const Scalar* k, const ulong* table) {
    AffinePoint ap;
    int started = 0;

    #pragma unroll
    for (int group = 0; group < 13; group++) {
        // Extract 20-bit window from scalar k at position (group * 20).
        int bit = group * 20;
        int limb_idx = bit >> 6;
        int bit_idx  = bit & 63;
        ulong raw;

        if (bit_idx <= 44) {
            raw = k->limbs[limb_idx] >> bit_idx;
        } else {
            ulong lo = k->limbs[limb_idx] >> bit_idx;
            ulong hi = 0;
            if (limb_idx < 3) {
                hi = k->limbs[limb_idx + 1] << (64 - bit_idx);
            }
            raw = lo | hi;
        }

        int value = (int)(raw & 0xFFFFFUL);

        if (value != 0) {
            ecmult_table_load_point(&ap, table, group * 1048576 + value);
            if (!started) {
                point_from_affine(r, &ap);
                started = 1;
            } else {
                point_add_mixed_unchecked(r, r, &ap);
            }
        }
    }

    if (!started) {
        point_set_infinity(r);
    }
}

// =============================================================================
// ECDSA signing function (precomputed table k*G + optimized inverse)
// =============================================================================
// Uses scalar_mul_generator_precomp (20-bit windowed table) for k*G
// and scalar_inverse_mod_n (nibble-based, faster) for k^(-1).
FORCE_INLINE int ecdsa_sign(const uchar* seckey32, const uchar* msg32, uchar* sig_out,
                      const ulong* ecmult_table0,
                      const ulong* ecmult_table1,
                      const ulong* ecmult_table2,
                      const ulong* ecmult_table3,
                      const ulong* ecmult_table4,
                      const ulong* ecmult_table5) {
    // 1. Generate deterministic nonce k via RFC 6979
    uchar nonce32[32];
    rfc6979_generate_k(seckey32, msg32, nonce32);
    
    // 2. Load nonce as scalar k
    Scalar k;
    scalar_set_b32(&k, nonce32);
    if (scalar_is_zero(&k)) return 0;
    scalar_reduce(&k);
    if (scalar_is_zero(&k)) return 0;
    
    // 3. Compute R = k*G using GLV shared-W26 9-add XYZZ path.
    FieldElement R_x, R_zz, zz_inv, rx_fe;
    scalar_mul_generator_precomp_xyzz(&R_x, &R_zz, &k, ecmult_table0, ecmult_table1, ecmult_table2, ecmult_table3, ecmult_table4, ecmult_table5);
    if ((R_zz.limbs[0]|R_zz.limbs[1]|R_zz.limbs[2]|R_zz.limbs[3])==0UL) return 0;
    field_inv_impl(&zz_inv, &R_zz);
    field_mul_impl(&rx_fe, &R_x, &zz_inv);
    
    // 5. r = R.x mod n
    uchar rx_bytes[32];
    rx_bytes[0]  = (uchar)(rx_fe.limbs[3] >> 56); rx_bytes[1]  = (uchar)(rx_fe.limbs[3] >> 48);
    rx_bytes[2]  = (uchar)(rx_fe.limbs[3] >> 40); rx_bytes[3]  = (uchar)(rx_fe.limbs[3] >> 32);
    rx_bytes[4]  = (uchar)(rx_fe.limbs[3] >> 24); rx_bytes[5]  = (uchar)(rx_fe.limbs[3] >> 16);
    rx_bytes[6]  = (uchar)(rx_fe.limbs[3] >> 8);  rx_bytes[7]  = (uchar)(rx_fe.limbs[3]);
    rx_bytes[8]  = (uchar)(rx_fe.limbs[2] >> 56); rx_bytes[9]  = (uchar)(rx_fe.limbs[2] >> 48);
    rx_bytes[10] = (uchar)(rx_fe.limbs[2] >> 40); rx_bytes[11] = (uchar)(rx_fe.limbs[2] >> 32);
    rx_bytes[12] = (uchar)(rx_fe.limbs[2] >> 24); rx_bytes[13] = (uchar)(rx_fe.limbs[2] >> 16);
    rx_bytes[14] = (uchar)(rx_fe.limbs[2] >> 8);  rx_bytes[15] = (uchar)(rx_fe.limbs[2]);
    rx_bytes[16] = (uchar)(rx_fe.limbs[1] >> 56); rx_bytes[17] = (uchar)(rx_fe.limbs[1] >> 48);
    rx_bytes[18] = (uchar)(rx_fe.limbs[1] >> 40); rx_bytes[19] = (uchar)(rx_fe.limbs[1] >> 32);
    rx_bytes[20] = (uchar)(rx_fe.limbs[1] >> 24); rx_bytes[21] = (uchar)(rx_fe.limbs[1] >> 16);
    rx_bytes[22] = (uchar)(rx_fe.limbs[1] >> 8);  rx_bytes[23] = (uchar)(rx_fe.limbs[1]);
    rx_bytes[24] = (uchar)(rx_fe.limbs[0] >> 56); rx_bytes[25] = (uchar)(rx_fe.limbs[0] >> 48);
    rx_bytes[26] = (uchar)(rx_fe.limbs[0] >> 40); rx_bytes[27] = (uchar)(rx_fe.limbs[0] >> 32);
    rx_bytes[28] = (uchar)(rx_fe.limbs[0] >> 24); rx_bytes[29] = (uchar)(rx_fe.limbs[0] >> 16);
    rx_bytes[30] = (uchar)(rx_fe.limbs[0] >> 8);  rx_bytes[31] = (uchar)(rx_fe.limbs[0]);
    
    Scalar sig_r;
    scalar_set_b32(&sig_r, rx_bytes);
    scalar_reduce(&sig_r);
    if (scalar_is_zero(&sig_r)) return 0;
    
    // 6. s = k^(-1) * (msg + r * seckey) mod n
    Scalar sec, msg_scalar, n_val, sig_s;
    scalar_set_b32(&sec, seckey32);
    scalar_reduce(&sec);           // Ensure sec < n (matches CUDA scalar_set_b32 reduce)
    scalar_set_b32(&msg_scalar, msg32);
    scalar_reduce(&msg_scalar);    // Ensure msg < n (matches CUDA scalar_set_b32 reduce)
    
    scalar_mul_mod_n(&n_val, &sig_r, &sec);
    scalar_add_mod_n(&n_val, &n_val, &msg_scalar);
    // k_inv using OPTIMIZED nibble-based inverse (uses scalar_sqr_mod_n)
    Scalar k_inv;
    scalar_inverse_mod_n(&k_inv, &k);
    scalar_mul_mod_n(&sig_s, &k_inv, &n_val);
    
    if (scalar_is_zero(&sig_s)) return 0;
    
    // 7. Normalize s
    if (scalar_is_high(&sig_s)) {
        scalar_negate(&sig_s, &sig_s);
    }
    
    // 8. DER encode
    uchar r_bytes[32], s_bytes[32];
    scalar_get_b32(r_bytes, &sig_r);
    scalar_get_b32(s_bytes, &sig_s);
    
    return der_encode_signature(sig_out, r_bytes, s_bytes);
}


// =============================================================================
// Batch-of-32 ECDSA inversion path + seckey-precomputed RFC6979
// =============================================================================
#ifndef BTCW_SIGN_BATCH
#define BTCW_SIGN_BATCH 128
#endif
#define SIGN_BATCH BTCW_SIGN_BATCH
static_assert(SIGN_BATCH == 64 || SIGN_BATCH == 128,
              "BTCW_SIGN_BATCH must be 64 or 128");
#ifndef BTCW_HYBRID_RX
#define BTCW_HYBRID_RX 0
#endif

// v48: invert the 128 Z values as two 64-element sub-batches.  This keeps
// SIGN_BATCH=128 for RFC/kG/scalar batching, but halves the largest temporary
// field-prefix array from 128 to 64 entries.  Two field inversions are used
// per 128 signatures instead of one; the benchmark determines whether the
// smaller local frame/cache footprint outweighs the extra inversion.
#define FIELD_INV_SUBBATCH 64
FORCE_INLINE void field_batch_inverse64_inplace(FieldElement* vals) {
    FieldElement prefix[FIELD_INV_SUBBATCH];
    prefix[0] = vals[0];
#pragma unroll
    for (int j=1;j<FIELD_INV_SUBBATCH;j++)
        field_mul_impl(&prefix[j],&prefix[j-1],&vals[j]);
    FieldElement acc;
    field_inv_impl(&acc,&prefix[FIELD_INV_SUBBATCH-1]);
#pragma unroll
    for (int j=FIELD_INV_SUBBATCH-1;j>0;j--) {
        FieldElement original=vals[j];
        field_mul_impl(&vals[j],&acc,&prefix[j-1]);
        field_mul_impl(&acc,&acc,&original);
    }
    vals[0]=acc;
}

FORCE_INLINE void field_batch_inverse64x2_inplace(FieldElement* vals) {
    #pragma unroll
    for (int base = 0; base < SIGN_BATCH; base += FIELD_INV_SUBBATCH) {
        FieldElement prefix[FIELD_INV_SUBBATCH];
        prefix[0] = vals[base];
        #pragma unroll
        for (int j = 1; j < FIELD_INV_SUBBATCH; ++j) {
            field_mul_impl(&prefix[j], &prefix[j - 1], &vals[base + j]);
        }
        FieldElement acc;
        field_inv_impl(&acc, &prefix[FIELD_INV_SUBBATCH - 1]);
        #pragma unroll
        for (int j = FIELD_INV_SUBBATCH - 1; j > 0; --j) {
            FieldElement original = vals[base + j];
            field_mul_impl(&vals[base + j], &acc, &prefix[j - 1]);
            field_mul_impl(&acc, &acc, &original);
        }
        vals[base] = acc;
    }
}

// Same construction in scalar mod-n space. vals[] becomes k^-1 in place.
FORCE_INLINE void scalar_batch_inverse32_inplace(Scalar* vals) {
    Scalar prefix[SIGN_BATCH];
    prefix[0] = vals[0];
    #pragma unroll
    for (int i = 1; i < SIGN_BATCH; ++i) {
        scalar_mul_mod_n(&prefix[i], &prefix[i - 1], &vals[i]);
    }
    Scalar acc;
    scalar_inverse_mod_n(&acc, &prefix[SIGN_BATCH - 1]);
    #pragma unroll
    for (int i = SIGN_BATCH - 1; i > 0; --i) {
        Scalar original = vals[i];
        scalar_mul_mod_n(&vals[i], &acc, &prefix[i - 1]);
        scalar_mul_mod_n(&acc, &acc, &original);
    }
    vals[0] = acc;
}

// v27: batch-invert k directly in transposed global scratch.  This removes the
// private k_batch[128] array while preserving a single scalar inversion/batch.
FORCE_INLINE void scalar_batch_inverse_global(Scalar* vals, ulong stride, ulong gid) {
    Scalar prefix[SIGN_BATCH];
    prefix[0] = vals[gid];
    #pragma unroll
    for (int i = 1; i < SIGN_BATCH; ++i) {
        Scalar v = vals[(ulong)i * stride + gid];
        scalar_mul_mod_n(&prefix[i], &prefix[i - 1], &v);
    }
    Scalar acc;
    scalar_inverse_mod_n(&acc, &prefix[SIGN_BATCH - 1]);
    #pragma unroll
    for (int i = SIGN_BATCH - 1; i > 0; --i) {
        ulong idx = (ulong)i * stride + gid;
        Scalar original = vals[idx];
        Scalar inv_i;
        scalar_mul_mod_n(&inv_i, &acc, &prefix[i - 1]);
        vals[idx] = inv_i;
        scalar_mul_mod_n(&acc, &acc, &original);
    }
    vals[gid] = acc;
}

// =============================================================================
// uint256 Addition (for mud = hash_no_sig + nonce)
// =============================================================================
// Both are 32-byte little-endian arrays interpreted as 4x64-bit limbs

typedef struct {
    ulong limbs[4]; // little-endian
} uint256_t;

FORCE_INLINE void uint256_add(uint256_t* r, const uint256_t* a, const uint256_t* b) {
    ulong carry = 0;
    r->limbs[0] = add_with_carry(a->limbs[0], b->limbs[0], 0, &carry);
    r->limbs[1] = add_with_carry(a->limbs[1], b->limbs[1], carry, &carry);
    r->limbs[2] = add_with_carry(a->limbs[2], b->limbs[2], carry, &carry);
    r->limbs[3] = add_with_carry(a->limbs[3], b->limbs[3], carry, &carry);
}

FORCE_INLINE ulong bswap64_miner(ulong x) {
    return ((x & 0x00000000000000FFUL) << 56) |
           ((x & 0x000000000000FF00UL) << 40) |
           ((x & 0x0000000000FF0000UL) << 24) |
           ((x & 0x00000000FF000000UL) << 8)  |
           ((x & 0x000000FF00000000UL) >> 8)  |
           ((x & 0x0000FF0000000000UL) >> 24) |
           ((x & 0x00FF000000000000UL) >> 40) |
           ((x & 0xFF00000000000000UL) >> 56);
}
FORCE_INLINE void mud_le_to_msg_scalar(Scalar* out, const uint256_t* mud) {
    out->limbs[3] = bswap64_miner(mud->limbs[0]);
    out->limbs[2] = bswap64_miner(mud->limbs[1]);
    out->limbs[1] = bswap64_miner(mud->limbs[2]);
    out->limbs[0] = bswap64_miner(mud->limbs[3]);
    scalar_reduce(out);
}


FORCE_INLINE int ecdsa_prepare_batch32(const RFC6979_SECKEY_PRECOMP* rfc_pc,
                                       const uchar* seckey32,
                                       const uint256_t* mud,
                                       const ulong* ecmult_table0,
                                       const ulong* ecmult_table1,
                                       const ulong* ecmult_table2,
                                       const ulong* ecmult_table3,
                                       const ulong* ecmult_table4,
                                       const ulong* ecmult_table5,
                                       Scalar* k,
                                       FieldElement* R_x,
                                       FieldElement* R_z) {
    // v32: mud already contains the little-endian message bytes. Convert its
    // four limbs directly to the scalar instead of materializing mb[32] and
    // immediately parsing those same bytes again.
    Scalar msg_scalar;
    mud_le_to_msg_scalar(&msg_scalar, mud);

    uint nonce_words[8];
    rfc6979_generate_k_precomp_words(rfc_pc, seckey32, &msg_scalar, nonce_words);
    scalar_set_sha256_words(k, nonce_words);
    if (scalar_is_zero(k)) return 0;
    scalar_reduce(k);
    if (scalar_is_zero(k)) return 0;

    /* XYZZ path returns X and ZZ=Z^2 directly. */
    scalar_mul_generator_precomp_xyzz(R_x, R_z, k, ecmult_table0, ecmult_table1, ecmult_table2, ecmult_table3, ecmult_table4, ecmult_table5);
    if ((R_z->limbs[0] | R_z->limbs[1] |
         R_z->limbs[2] | R_z->limbs[3]) == 0UL) return 0;
    return 1;
}

FORCE_INLINE int ecdsa_finish_batch32_scalars(const Scalar* sec,
                                      const Scalar* msg_scalar,
                                      const Scalar* k,
                                      const FieldElement* R_x_affine,
                                      Scalar* sig_r,
                                      Scalar* sig_s) {
    // R_x_affine was normalized immediately after the field batch inversion.
    sig_r->limbs[0] = R_x_affine->limbs[0];
    sig_r->limbs[1] = R_x_affine->limbs[1];
    sig_r->limbs[2] = R_x_affine->limbs[2];
    sig_r->limbs[3] = R_x_affine->limbs[3];
    scalar_reduce(sig_r);
    if (scalar_is_zero(sig_r)) return 0;

    Scalar kinv, rd, sum;
    scalar_inverse_mod_n(&kinv, k);
    scalar_mul_mod_n(&rd, sig_r, sec);
    scalar_add_mod_n(&sum, &rd, msg_scalar);
    scalar_mul_mod_n(sig_s, &kinv, &sum);
    if (scalar_is_zero(sig_s)) return 0;
    if (scalar_is_high(sig_s)) scalar_negate(sig_s, sig_s);
    return 1;
}

// Build nonce || CompactSize(DER) || DER directly into the SHA input buffer,
// avoiding a separate DER signature buffer plus a second preimage copy.

// v192: specialized SHA256d for the BTCW nonce||CompactSize(DER)||DER preimage.
// The serialized signature preimage is always >64 and <=82 bytes, so SHA-256
// always consumes exactly two blocks.  Use the existing 16-word rolling SHA
// core directly: no SHA256_CTX, no generic update/final loops, no 64-word
// expanded schedule, and no 32-byte intermediate digest buffer.
FORCE_INLINE void btcw_double_sha256_preimage_2block(const uchar* preimage,
                                                      uint len,
                                                      uchar out32[32]) {
    uint s[8];
    s[0]=0x6a09e667U; s[1]=0xbb67ae85U; s[2]=0x3c6ef372U; s[3]=0xa54ff53aU;
    s[4]=0x510e527fU; s[5]=0x9b05688cU; s[6]=0x1f83d9abU; s[7]=0x5be0cd19U;

    uint w[16];
    #pragma unroll
    for (int i=0; i<16; ++i) w[i] = read_be32(preimage + 4*i);
    sha256_transform_words(s, w);

    #pragma unroll
    for (int i=0; i<16; ++i) w[i] = 0U;
    const uint tail = len - 64U;
    #pragma unroll 18
    for (uint i=0; i<18U; ++i) {
        if (i < tail) {
            const uint wi = i >> 2;
            const uint sh = 24U - ((i & 3U) << 3);
            w[wi] |= ((uint)preimage[64U+i]) << sh;
        }
    }
    {
        const uint wi = tail >> 2;
        const uint sh = 24U - ((tail & 3U) << 3);
        w[wi] |= 0x80U << sh;
    }
    w[15] = len << 3;
    sha256_transform_words(s, w);

    // Second SHA-256 hashes the 32-byte first digest.  The first digest is
    // already in big-endian SHA word form, so feed state words directly.
    uint d[8];
    d[0]=0x6a09e667U; d[1]=0xbb67ae85U; d[2]=0x3c6ef372U; d[3]=0xa54ff53aU;
    d[4]=0x510e527fU; d[5]=0x9b05688cU; d[6]=0x1f83d9abU; d[7]=0x5be0cd19U;
    #pragma unroll
    for (int i=0; i<8; ++i) w[i] = s[i];
    w[8] = 0x80000000U;
    #pragma unroll
    for (int i=9; i<15; ++i) w[i] = 0U;
    w[15] = 256U;
    sha256_transform_words(d, w);

    #pragma unroll
    for (int i=0; i<8; ++i) write_be32(out32 + 4*i, d[i]);
}

// NEWFORK hashes the DER signature itself (70 or 71 bytes), rather than the
// legacy nonce || CompactSize || DER preimage above.  Both valid lengths use
// exactly two blocks in the first SHA-256 and one block in the second SHA-256.
// Keeping this fixed-shape path out of SHA256_CTX removes the byte-at-a-time
// update/final machinery from the per-candidate hot loop.
FORCE_INLINE uint btcw_der_meets_fixed_target(const uchar* der, uint len, const uchar* target32) {
    uint s[8];
    s[0]=0x6a09e667U; s[1]=0xbb67ae85U; s[2]=0x3c6ef372U; s[3]=0xa54ff53aU;
    s[4]=0x510e527fU; s[5]=0x9b05688cU; s[6]=0x1f83d9abU; s[7]=0x5be0cd19U;

    uint w[16];
    #pragma unroll
    for (int i=0; i<16; ++i) w[i] = read_be32(der + 4*i);
    sha256_transform_words(s, w);

    #pragma unroll
    for (int i=0; i<16; ++i) w[i] = 0U;
    const uint tail = len - 64U; // 6 or 7 for accepted NEWFORK signatures
    #pragma unroll
    for (uint i=0; i<7U; ++i) {
        if (i < tail) {
            const uint wi = i >> 2;
            const uint sh = 24U - ((i & 3U) << 3);
            w[wi] |= ((uint)der[64U+i]) << sh;
        }
    }
    {
        const uint wi = tail >> 2;
        const uint sh = 24U - ((tail & 3U) << 3);
        w[wi] |= 0x80U << sh;
    }
    w[15] = len << 3;
    sha256_transform_words(s, w);

    uint d[8];
    d[0]=0x6a09e667U; d[1]=0xbb67ae85U; d[2]=0x3c6ef372U; d[3]=0xa54ff53aU;
    d[4]=0x510e527fU; d[5]=0x9b05688cU; d[6]=0x1f83d9abU; d[7]=0x5be0cd19U;
    #pragma unroll
    for (int i=0; i<8; ++i) w[i] = s[i];
    w[8] = 0x80000000U;
    #pragma unroll
    for (int i=9; i<15; ++i) w[i] = 0U;
    w[15] = 256U;
    sha256_transform_words(d, w);

    uchar out32[32];
#pragma unroll
    for (int i=0;i<8;++i) write_be32(out32 + 4*i, d[i]);
    return hash_meets_target_le(out32, target32);
}

FORCE_INLINE void btcw_hash_signature_direct(ulong nonce,
                                              const Scalar* sig_r,
                                              const Scalar* sig_s,
                                              uchar hashPoW[32]) {
    uchar r32[32], s32[32];
    scalar_get_b32(r32, sig_r);
    scalar_get_b32(s32, sig_s);

    int rs = 0; while (rs < 31 && r32[rs] == 0) rs++;
    int ss = 0; while (ss < 31 && s32[ss] == 0) ss++;
    int rpad = (r32[rs] & 0x80) ? 1 : 0;
    int spad = (s32[ss] & 0x80) ? 1 : 0;
    int rlen = 32 - rs + rpad;
    int slen = 32 - ss + spad;
    int sig_len = 6 + rlen + slen;

    uchar preimage[82];
    preimage[0]=(uchar)nonce; preimage[1]=(uchar)(nonce>>8);
    preimage[2]=(uchar)(nonce>>16); preimage[3]=(uchar)(nonce>>24);
    preimage[4]=(uchar)(nonce>>32); preimage[5]=(uchar)(nonce>>40);
    preimage[6]=(uchar)(nonce>>48); preimage[7]=(uchar)(nonce>>56);
    preimage[8]=(uchar)sig_len;
    int p=9;
    preimage[p++]=0x30; preimage[p++]=(uchar)(sig_len-2);
    preimage[p++]=0x02; preimage[p++]=(uchar)rlen;
    if (rpad) preimage[p++]=0;
    for (int i=rs;i<32;i++) preimage[p++]=r32[i];
    preimage[p++]=0x02; preimage[p++]=(uchar)slen;
    if (spad) preimage[p++]=0;
    for (int i=ss;i<32;i++) preimage[p++]=s32[i];

    btcw_double_sha256_preimage_2block(preimage, (uint)p, hashPoW);
}

// =============================================================================
// Mining Kernel
// =============================================================================
// Number of nonces each thread processes per kernel invocation.
// Amortizes global memory loads (key, hash_no_sig) across multiple hashes.
#define NONCES_PER_THREAD 128


/* --- Batch64 SHA/DER hot-path helpers --- */
FORCE_INLINE int scalar_der_bytes(const Scalar* s, uchar out33[33])
{
    uchar tmp[32];
    scalar_get_b32(tmp, s);
    int first = 0;
    while (first < 31 && tmp[first] == 0) ++first;
    int len = 32 - first;
    int p = 0;
    if (tmp[first] & 0x80) out33[p++] = 0;
    for (int i = first; i < 32; ++i) out33[p++] = tmp[i];
    return p;
}

FORCE_INLINE int build_btcw_sig_preimage(
    ulong nonce,
    const Scalar* r,
    const Scalar* s,
    uchar out[82])
{
    uchar rb[33], sb[33];
    int rlen = scalar_der_bytes(r, rb);
    int slen = scalar_der_bytes(s, sb);
    int derlen = 6 + rlen + slen;

#pragma unroll
    for (int i=0;i<8;i++) out[i]=(uchar)(nonce>>(8*i));
    out[8]=(uchar)derlen;

    int p=9;
    out[p++]=0x30;
    out[p++]=(uchar)(derlen-2);
    out[p++]=0x02;
    out[p++]=(uchar)rlen;
    for(int i=0;i<rlen;i++) out[p++]=rb[i];
    out[p++]=0x02;
    out[p++]=(uchar)slen;
    for(int i=0;i<slen;i++) out[p++]=sb[i];
    return p;
}

extern "C" __global__ void btcw_mine(
    const uchar* key_data,
    const uchar* hash_no_sig,
    volatile ulong* result_nonce,
    volatile uint* result_found,
    volatile uint* result_der_len,
    volatile uchar* result_der,
    volatile uint* hashrate_ctr,
    const ulong nonce_base,
    const uint gpu_num,
    const ulong* ecmult_table0,
    const ulong* ecmult_table1,
    const ulong* ecmult_table2,
    const ulong* ecmult_table3,
    const ulong* ecmult_table4,
    const ulong* ecmult_table5,
    const uchar* target32,
    Scalar* k_scratch,
    FieldElement* rx_scratch)
{
    const uint gid=(uint)(blockIdx.x*blockDim.x+threadIdx.x);
    uchar seckey[32], msg32[32];
    #pragma unroll
    for(int i=0;i<32;i++){seckey[i]=key_data[i];msg32[i]=hash_no_sig[i];}
    Scalar sec,msg_scalar;
    scalar_set_b32(&sec,seckey); scalar_reduce(&sec);
    scalar_set_b32(&msg_scalar,msg32); scalar_reduce(&msg_scalar);
    // The key and message are invariant for all 128 candidates handled by
    // this thread.  Reuse their RFC6979 prefix state and reduced message;
    // only LE32(test_case) changes in the hot loop.
    RFC6979_SECKEY_PRECOMP rfc_pc;
    rfc6979_precompute_seckey(seckey,&rfc_pc);
    uchar msgmod32[32];
    scalar_get_b32(msgmod32,&msg_scalar);

#if SIGN_BATCH == 128 && BTCW_HYBRID_RX
    // Preserve one 128-way scalar inversion, but keep only 64 EC points in
    // the thread-local frame at once.  Normalized X coordinates are parked
    // in a transposed global buffer between the EC and ECDSA-finish phases.
    for(int base_iter=0;base_iter<NONCES_PER_THREAD;base_iter+=SIGN_BATCH){
        int batch_ok=1;
#pragma unroll
        for(int ec_base=0;ec_base<SIGN_BATCH;ec_base+=FIELD_INV_SUBBATCH) {
            FieldElement rxj_batch[FIELD_INV_SUBBATCH], z_batch[FIELD_INV_SUBBATCH];
        #pragma unroll
            for(int j=0;j<FIELD_INV_SUBBATCH;j++){
            const int b=ec_base+j;
            ulong idx64=nonce_base+(ulong)gid*NONCES_PER_THREAD+(ulong)(base_iter+b);
            if(idx64==0UL) idx64=1UL;
            uint nw[8]; Scalar ktmp;
            rfc6979_generate_k_testcase_words(&rfc_pc,seckey,msgmod32,idx64,nw);
            scalar_set_sha256_words(&ktmp,nw); scalar_reduce(&ktmp);
            if(scalar_is_zero(&ktmp)){batch_ok=0;continue;}
            scalar_mul_generator_precomp_xyzz(&rxj_batch[j],&z_batch[j],&ktmp,
                ecmult_table0,ecmult_table1,ecmult_table2,ecmult_table3,ecmult_table4,ecmult_table5);
            if((z_batch[j].limbs[0]|z_batch[j].limbs[1]|z_batch[j].limbs[2]|z_batch[j].limbs[3])==0UL) batch_ok=0;
            k_scratch[(ulong)b*(ulong)get_global_size(0)+(ulong)gid]=ktmp;
        }
        if(!batch_ok) continue;
            field_batch_inverse64_inplace(z_batch);
        #pragma unroll
            for(int j=0;j<FIELD_INV_SUBBATCH;j++){
                FieldElement x; field_mul_impl(&x,&rxj_batch[j],&z_batch[j]);
                rx_scratch[(ulong)(ec_base+j)*(ulong)get_global_size(0)+(ulong)gid]=x;
            }
        }
        if(!batch_ok) continue;

        #pragma unroll
        for(int b=0;b<SIGN_BATCH;b++){
            ulong extra64=nonce_base+(ulong)gid*NONCES_PER_THREAD+(ulong)(base_iter+b);
            if(extra64==0UL) extra64=1UL;
            Scalar kinv=k_scratch[(ulong)b*(ulong)get_global_size(0)+(ulong)gid];
            FieldElement rx=rx_scratch[(ulong)b*(ulong)get_global_size(0)+(ulong)gid];
            Scalar r,sig_s;
            if(!ecdsa_finish_batch32_scalars(&sec,&msg_scalar,&kinv,&rx,&r,&sig_s)) continue;
            uchar rb[32],sb[32],der[73];
            scalar_get_b32(rb,&r); scalar_get_b32(sb,&sig_s);
            int len=der_encode_signature(der,rb,sb);
            if(len!=70 && len!=71) continue;
            if(btcw_der_meets_fixed_target(der,(uint)len,target32)){
                if(atomicCAS((uint*)result_found,0u,1u)==0u){
                    *result_nonce=extra64;
                    *result_der_len=(uint)len;
                    for(int i=0;i<len && i<72;i++) result_der[i]=der[i];
                }
            }
        }
    }
#else
    for(int base_iter=0;base_iter<NONCES_PER_THREAD;base_iter+=SIGN_BATCH){
        FieldElement rxj_batch[SIGN_BATCH], z_batch[SIGN_BATCH];
        int batch_ok=1;
#pragma unroll
        for(int b=0;b<SIGN_BATCH;b++){
            ulong idx64=nonce_base+(ulong)gid*NONCES_PER_THREAD+(ulong)(base_iter+b);
            if(idx64==0UL) idx64=1UL;
            uint nw[8]; Scalar ktmp;
            rfc6979_generate_k_testcase_words(&rfc_pc,seckey,msgmod32,idx64,nw);
            scalar_set_sha256_words(&ktmp,nw); scalar_reduce(&ktmp);
            if(scalar_is_zero(&ktmp)){batch_ok=0;continue;}
            scalar_mul_generator_precomp_xyzz(&rxj_batch[b],&z_batch[b],&ktmp,
                ecmult_table0,ecmult_table1,ecmult_table2,ecmult_table3,ecmult_table4,ecmult_table5);
            if((z_batch[b].limbs[0]|z_batch[b].limbs[1]|z_batch[b].limbs[2]|z_batch[b].limbs[3])==0UL)batch_ok=0;
            k_scratch[(ulong)b*(ulong)get_global_size(0)+(ulong)gid]=ktmp;
        }
        if(!batch_ok)continue;
        field_batch_inverse64x2_inplace(z_batch);
#pragma unroll
        for(int b=0;b<SIGN_BATCH;b++){FieldElement x;field_mul_impl(&x,&rxj_batch[b],&z_batch[b]);rxj_batch[b]=x;}
#pragma unroll
        for(int b=0;b<SIGN_BATCH;b++){
            ulong extra64=nonce_base+(ulong)gid*NONCES_PER_THREAD+(ulong)(base_iter+b); if(extra64==0UL)extra64=1UL;
            Scalar kinv=k_scratch[(ulong)b*(ulong)get_global_size(0)+(ulong)gid],r,sig_s;
            if(!ecdsa_finish_batch32_scalars(&sec,&msg_scalar,&kinv,&rxj_batch[b],&r,&sig_s))continue;
            uchar rb[32],sb[32],der[73]; scalar_get_b32(rb,&r);scalar_get_b32(sb,&sig_s);
            int len=der_encode_signature(der,rb,sb); if(len!=70&&len!=71)continue;
            if(btcw_der_meets_fixed_target(der,(uint)len,target32)){
                if(atomicCAS((uint*)result_found,0u,1u)==0u){
                    *result_nonce=extra64;
                    *result_der_len=(uint)len;
                    for(int i=0;i<len && i<72;i++) result_der[i]=der[i];
                }
            }
        }
    }
#endif
    atomicAdd((uint*)hashrate_ctr,(uint)NONCES_PER_THREAD);
}

// Startup guard for the optimized NEWFORK RFC6979 path.  Compare its nonce
// words with the original generic HMAC implementation for published-vector
// test cases before allocating the large generator tables or mining.
extern "C" __global__ void diagnostic_mining_ecdsa(
    const uchar* seckey32,
    const uchar* msg32,
    uint test_case,
    uchar* r_out,
    uchar* s_out,
    uint* flags_out)
{
    if (blockIdx.x || threadIdx.x) return;
    uint flags = 0;
    Scalar a, b, m;
    a.limbs[0] = 3UL; a.limbs[1] = 0UL; a.limbs[2] = 0UL; a.limbs[3] = 0UL;
    b.limbs[0] = 7UL; b.limbs[1] = 0UL; b.limbs[2] = 0UL; b.limbs[3] = 0UL;
    scalar_mul_mod_n(&m, &a, &b);
    if (m.limbs[0] == 21UL && m.limbs[1] == 0UL && m.limbs[2] == 0UL && m.limbs[3] == 0UL) flags |= 1u;

    Scalar two;
    two.limbs[0] = 2UL; two.limbs[1] = 0UL; two.limbs[2] = 0UL; two.limbs[3] = 0UL;
    Scalar inv2;
    scalar_inverse_fermat(&inv2, &two);
    scalar_mul_mod_n(&m, &inv2, &two);
    if (m.limbs[0] == 1UL && m.limbs[1] == 0UL && m.limbs[2] == 0UL && m.limbs[3] == 0UL) flags |= 2u;

    uchar seckey[32], msg[32], msgmod[32];
    for (int i = 0; i < 32; ++i) { seckey[i] = seckey32[i]; msg[i] = msg32[i]; }
    Scalar sec, msg_scalar;
    scalar_set_b32(&sec, seckey); scalar_reduce(&sec);
    scalar_set_b32(&msg_scalar, msg); scalar_reduce(&msg_scalar);
    scalar_get_b32(msgmod, &msg_scalar);
    RFC6979_SECKEY_PRECOMP pc;
    rfc6979_precompute_seckey(seckey, &pc);
    uint nw[8];
    Scalar k;
    rfc6979_generate_k_testcase_words(&pc, seckey, msgmod, test_case, nw);
    scalar_set_sha256_words(&k, nw); scalar_reduce(&k);
    FieldElement Rx, ZZ;
    scalar_mul_generator_precomp_xyzz(&Rx, &ZZ, &k, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    FieldElement zinv, xaff;
    field_inv_impl(&zinv, &ZZ);
    field_mul_impl(&xaff, &Rx, &zinv);
    Scalar sig_r, sig_s;
    if (ecdsa_finish_batch32_scalars(&sec, &msg_scalar, &k, &xaff, &sig_r, &sig_s)) {
        flags |= 4u;
        scalar_get_b32(r_out, &sig_r);
        scalar_get_b32(s_out, &sig_s);
    } else {
        for (int i = 0; i < 32; ++i) { r_out[i] = 0; s_out[i] = 0; }
    }
    *flags_out = flags;
}

extern "C" __global__ void diagnostic_rfc6979_testcase(uint* ok_out) {
    if (blockIdx.x || threadIdx.x) return;
    uchar sk[32], msg[32], msgmod[32];
#pragma unroll
    for (int i=0;i<32;i++) { sk[i]=0; msg[i]=(uchar)i; }
    sk[31]=1;
    Scalar ms;
    scalar_set_b32(&ms,msg); scalar_reduce(&ms); scalar_get_b32(msgmod,&ms);
    RFC6979_SECKEY_PRECOMP pc;
    rfc6979_precompute_seckey(sk,&pc);
    const uint cases[2]={1U,0x12345678U};
    uint ok=1U;
#pragma unroll
    for (int c=0;c<2;c++) {
        uint reference[8], fast[8];
        rfc6979_generate_k_testcase(sk,msg,cases[c],reference);
        rfc6979_generate_k_testcase_words(&pc,sk,msgmod,cases[c],fast);
#pragma unroll
        for (int i=0;i<8;i++) if (reference[i]!=fast[i]) ok=0U;
    }
    *ok_out=ok;
}

// =============================================================================
// Precompute Kernel: builds the 20-bit ecmult_gen table in parallel on the GPU
// =============================================================================
// Layout:
//   table[group * 1048576 + value] = value * 2^(20*group) * G
//   group = 0..12, value = 0..1048575
// Each entry is 8 ulongs (64 bytes), total 832 MiB.
//
// Unlike the old implementation, this kernel uses ONE work-item per table
// entry instead of one work-item generating all 13,631,488 entries serially.


/* Shared GLV fixed-base table.
 * Offsets: 0,26,52,78,104 bits.  First four groups use signed W26
 * (2^25 magnitudes, 2 GiB/group); final group uses a 25-bit window with an
 * implicit zero bit 128 (2^24 magnitudes, 1 GiB). Total VRAM: 9 GiB.
 */
__device__ __constant__ ulong GLV_BASE_X[6][4] = {
    {0x59F2815B16F81798UL,0x029BFCDB2DCE28D9UL,0x55A06295CE870B07UL,0x79BE667EF9DCBBACUL},
    {0xCB6115925232FCDAUL,0xB700DBFFA6C0E77BUL,0x6BF771C00BD548C7UL,0x723CBAA6E5DB996DUL},
    {0x57545CCC1A37B7C0UL,0xEC08D0F7BB11069FUL,0xA6E000935EF22151UL,0x53904FAA0B334CDDUL},
    {0xFFD959AF60C82A0AUL,0x0F9226C60F668832UL,0x6B06C9F1919413B1UL,0x0948BF809B1988A4UL},
    {0x32427E2840FB27B6UL,0xC76E3DB2BE430576UL,0x10F238AD61686AA5UL,0xFEA74E3DBE778B1BUL},
    {0xDDC07BBCC4E16070UL,0xF2A182031EFD6915UL,0x13BA48E51D567543UL,0xA301697BDFCD7043UL},
};
__device__ __constant__ ulong GLV_BASE_Y[6][4] = {
    {0x9C47D08FFB10D4B8UL,0xFD17B448A6855419UL,0x5DA4FBFC0E1108A8UL,0x483ADA7726A3C465UL},
    {0x01DC069D9EB39F5FUL,0x2660A06537794948UL,0xA921137488824D6EUL,0x96E867B5595CC498UL},
    {0x9DCB096B022771C8UL,0x13999981E1443469UL,0x88C9ECCAC20D3C1CUL,0x5BC087D0BC80106DUL},
    {0xD4CB7F88D8C8E589UL,0x6D4DFF08C97CD2BEUL,0xDC6B74C5D1C3418CUL,0x53A562856DCB6646UL},
    {0x701D3DB7F23CB96FUL,0x126B596B973F7B77UL,0x7CF674DECCB6AF93UL,0x6E0568DB9B0B1329UL},
    {0x0C0D1A041E177EA1UL,0x1735DBF7C0A11A13UL,0x081809FA25D40F9BUL,0x7370F91CFB67E4F5UL},
};

FORCE_INLINE void write_affine_to_table(ulong* table, int index,
                                         const JacobianPoint* jac) {
    FieldElement z_inv, z_inv2, z_inv3, ax, ay;
    field_inv_impl(&z_inv, &jac->z);
    field_sqr_impl(&z_inv2, &z_inv);
    field_mul_impl(&z_inv3, &z_inv, &z_inv2);
    field_mul_impl(&ax, &jac->x, &z_inv2);
    field_mul_impl(&ay, &jac->y, &z_inv3);

    int off = index * 8;
    table[off + 0] = ax.limbs[0]; table[off + 1] = ax.limbs[1];
    table[off + 2] = ax.limbs[2]; table[off + 3] = ax.limbs[3];
    table[off + 4] = ay.limbs[0]; table[off + 5] = ay.limbs[1];
    table[off + 6] = ay.limbs[2]; table[off + 7] = ay.limbs[3];
}

/* Host launches this once for each group. */
extern "C" __global__ void precompute_ecmult_gen_table(ulong* table, uint group, uint entries) {
    const uint slot=(uint)((uint)(blockIdx.x * blockDim.x + threadIdx.x));
    if (group>=6u || slot>=entries) return;
    const uint value=slot+1u;
    const int width=(group<5u)?24:9;

    AffinePoint base;
    for(int i=0;i<4;++i){base.x.limbs[i]=GLV_BASE_X[group][i];base.y.limbs[i]=GLV_BASE_Y[group][i];}
    JacobianPoint acc; int started=0;
    for(int bit=width-1;bit>=0;--bit){
        if(started){JacobianPoint d;point_double_impl(&d,&acc);acc=d;}
        if((value>>bit)&1u){
            if(!started){point_from_affine(&acc,&base);started=1;}
            else{JacobianPoint n;point_add_mixed_impl(&n,&acc,&base);acc=n;}
        }
    }
    write_affine_to_table(table,(int)slot,&acc);
}

// =============================================================================
// Diagnostic Kernel: verify ECDSA signing on GPU
// =============================================================================
// Signs a caller-provided (seckey, message) pair and writes the DER signature
// to the output buffer.  Runs a single work-item.
// The host code can then compare this against a known-good reference.
extern "C" __global__ void diagnostic_ecdsa_sign(
    const uchar*  seckey32,    // 32 bytes
    const uchar*  msg32,       // 32 bytes
    uchar*        sig_out,     // 73 bytes max DER
    int*          sig_len_out, // 1 int
    const ulong*  ecmult_table0,
    const ulong*  ecmult_table1,
    const ulong*  ecmult_table2,
    const ulong*  ecmult_table3,
    const ulong*  ecmult_table4,
    const ulong*  ecmult_table5
) {
    if (((uint)(blockIdx.x * blockDim.x + threadIdx.x)) != 0) return;

    uchar sk[32], m[32];
    for (int i = 0; i < 32; i++) { sk[i] = seckey32[i]; m[i] = msg32[i]; }

    uchar der[73];
    int len = ecdsa_sign(sk, m, der, ecmult_table0, ecmult_table1, ecmult_table2, ecmult_table3, ecmult_table4, ecmult_table5);

    *sig_len_out = len;
    for (int i = 0; i < len; i++) sig_out[i] = der[i];
}

// =============================================================================
// Diagnostic Kernel: test scalar arithmetic in isolation
// =============================================================================
// Writes results as hex limbs (little-endian) to output buffer.
// Tests:
//   [0..31]   = mul(3, 7)                    → expect {21, 0, 0, 0}
//   [32..63]  = inv(2)                       → expect (n+1)/2
//   [64..95]  = mul(2, inv(2))               → expect {1, 0, 0, 0}
//   [96..127] = mul(n-1, n-1)                → expect {1, 0, 0, 0}
//   [128]     = pass/fail byte (0xFF = all pass, else bit flags)
extern "C" __global__ void diagnostic_scalar_ops(
    uchar* output     // 129 bytes
) {
    if (((uint)(blockIdx.x * blockDim.x + threadIdx.x)) != 0) return;

    Scalar a, b, r;
    uchar flags = 0;

    // Helper: write scalar limbs (LE 64-bit) to output
    #define WRITE_SCALAR(off, s) \
        for (int _i = 0; _i < 4; _i++) \
            for (int _j = 0; _j < 8; _j++) \
                output[(off) + _i * 8 + _j] = (uchar)((s).limbs[_i] >> (_j * 8));

    // Test 1: 3 * 7 = 21
    a.limbs[0] = 3; a.limbs[1] = 0; a.limbs[2] = 0; a.limbs[3] = 0;
    b.limbs[0] = 7; b.limbs[1] = 0; b.limbs[2] = 0; b.limbs[3] = 0;
    scalar_mul_mod_n(&r, &a, &b);
    WRITE_SCALAR(0, r);
    if (r.limbs[0] == 21 && r.limbs[1] == 0 && r.limbs[2] == 0 && r.limbs[3] == 0)
        flags |= 0x01;

    // Test 2: inverse(2) → should be (n+1)/2
    // (n+1)/2 = 7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A1
    a.limbs[0] = 2; a.limbs[1] = 0; a.limbs[2] = 0; a.limbs[3] = 0;
    scalar_inverse_mod_n(&r, &a);
    WRITE_SCALAR(32, r);
    if (r.limbs[0] == 0xDFE92F46681B20A1UL &&
        r.limbs[1] == 0x5D576E7357A4501DUL &&
        r.limbs[2] == 0xFFFFFFFFFFFFFFFFUL &&
        r.limbs[3] == 0x7FFFFFFFFFFFFFFFUL)
        flags |= 0x02;

    // Test 3: 2 * inv(2) = 1
    Scalar two;
    two.limbs[0] = 2; two.limbs[1] = 0; two.limbs[2] = 0; two.limbs[3] = 0;
    Scalar identity;
    scalar_mul_mod_n(&identity, &two, &r);
    WRITE_SCALAR(64, identity);
    if (identity.limbs[0] == 1 && identity.limbs[1] == 0 &&
        identity.limbs[2] == 0 && identity.limbs[3] == 0)
        flags |= 0x04;

    // Test 4: (n-1) * (n-1) = 1  (since n-1 ≡ -1 mod n, (-1)^2 = 1)
    a.limbs[0] = N_LIMB0 - 1; // Wait, n-1 limbs: subtract 1 from n
    // n = {N_LIMB0, N_LIMB1, N_LIMB2, N_LIMB3}
    // n-1 = {N_LIMB0-1, N_LIMB1, N_LIMB2, N_LIMB3}  (since N_LIMB0 > 0)
    a.limbs[0] = 0xBFD25E8CD0364140UL;
    a.limbs[1] = 0xBAAEDCE6AF48A03BUL;
    a.limbs[2] = 0xFFFFFFFFFFFFFFFEUL;
    a.limbs[3] = 0xFFFFFFFFFFFFFFFFUL;
    scalar_mul_mod_n(&r, &a, &a);
    WRITE_SCALAR(96, r);
    if (r.limbs[0] == 1 && r.limbs[1] == 0 && r.limbs[2] == 0 && r.limbs[3] == 0)
        flags |= 0x08;

    // Test 5: 2^256 mod n via 8 chained squarings
    // 2^(2^8) = 2^256 mod n = 2^256 - n = NC
    // Expected: {0x402DA1732FC9BEBF, 0x4551231950B75FC4, 1, 0}
    a.limbs[0] = 2; a.limbs[1] = 0; a.limbs[2] = 0; a.limbs[3] = 0;
    for (int _sq = 0; _sq < 8; _sq++) {
        scalar_mul_mod_n(&r, &a, &a);
        a = r;
    }
    // a is now 2^256 mod n
    // Write at offset 129 (output[129..160])
    for (int _i = 0; _i < 4; _i++)
        for (int _j = 0; _j < 8; _j++)
            output[129 + _i * 8 + _j] = (uchar)(a.limbs[_i] >> (_j * 8));
    if (a.limbs[0] == 0x402DA1732FC9BEBFUL &&
        a.limbs[1] == 0x4551231950B75FC4UL &&
        a.limbs[2] == 1 && a.limbs[3] == 0)
        flags |= 0x10;

    // Test 6: single squaring of a mid-range value
    // (2^128)^2 = 2^256 mod n = NC
    a.limbs[0] = 0; a.limbs[1] = 0; a.limbs[2] = 1; a.limbs[3] = 0;  // 2^128
    scalar_mul_mod_n(&r, &a, &a);
    // Write at offset 161 (output[161..192])
    for (int _i = 0; _i < 4; _i++)
        for (int _j = 0; _j < 8; _j++)
            output[161 + _i * 8 + _j] = (uchar)(r.limbs[_i] >> (_j * 8));
    if (r.limbs[0] == 0x402DA1732FC9BEBFUL &&
        r.limbs[1] == 0x4551231950B75FC4UL &&
        r.limbs[2] == 1 && r.limbs[3] == 0)
        flags |= 0x20;

    output[128] = flags;
    #undef WRITE_SCALAR
}

#ifdef BTCW_MSVC_LONG_REMAP
#undef long
#undef BTCW_MSVC_LONG_REMAP
#endif
