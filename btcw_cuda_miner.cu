// BTCW.SPACE CUDA GPU Miner v45 - CUDA runtime host + v40 crypto kernel
#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <ctime>
#include <thread>
#include <atomic>
#include <vector>
#include <cstdlib>
#ifndef _WIN32
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif
#include <cerrno>
#include <csignal>

#include "btcw_cuda_kernels.cuh"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

static std::atomic<bool> g_running{true};
static void signal_handler(int){ g_running.store(false); }
static void print_timestamp(){ time_t now=time(nullptr); tm* lt=localtime(&now); char b[16]; strftime(b,sizeof(b),"%H:%M:%S",lt); printf("[%s] ",b); }
#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__); return 1; } } while(0)
#ifdef _WIN32
#define SHM_NAME "shared_mem"
#else
#define SHM_NAME "/shared_mem"
#endif
static const int CTX_SIZE_BYTES=8*20, KEY_SIZE_BYTES=32, HASH_NO_SIG_SIZE_BYTES=32;
// Keep the original node IPC contract unchanged: key[32] + ctx[160] + hash_no_sig[32].
static const int TOTAL_BYTES_SEND=CTX_SIZE_BYTES+KEY_SIZE_BYTES+HASH_NO_SIG_SIZE_BYTES;
static const uint64_t SENTINEL_NONCE=0x0707070707070707ULL;
struct SharedData { volatile uint64_t nonce; volatile uint8_t data[TOTAL_BYTES_SEND]; };

int main(int argc,char** argv){
 signal(SIGINT,signal_handler); signal(SIGTERM,signal_handler);
 int gpu_num=0; size_t user_work=0, user_block=0;
 if(argc>=2) gpu_num=atoi(argv[1]); if(argc>=3) user_work=strtoull(argv[2],nullptr,10); if(argc>=4) user_block=strtoull(argv[3],nullptr,10);
 int ndev=0; CUDA_CHECK(cudaGetDeviceCount(&ndev)); if(ndev<=0){fprintf(stderr,"No CUDA GPU found.\n");return 1;}
 print_timestamp(); printf("BTCW.SPACE CUDA GPU Miner 4060-v202-NEWFORK (RFC-extra fast path, DER-SHA fixed, sign-batch%d)\n",SIGN_BATCH);
 print_timestamp(); printf("Found %d CUDA device(s):\n",ndev);
 for(int i=0;i<ndev;i++){ cudaDeviceProp p{}; CUDA_CHECK(cudaGetDeviceProperties(&p,i)); printf("  Device %d: %s SMs=%d Mem=%zuMB MaxBlock=%d CC=%d.%d\n",i,p.name,p.multiProcessorCount,p.totalGlobalMem/(1024*1024),p.maxThreadsPerBlock,p.major,p.minor); }
 int dev=(gpu_num>0)?gpu_num-1:0; if(dev<0||dev>=ndev){fprintf(stderr,"GPU %d not found.\n",gpu_num);return 1;} CUDA_CHECK(cudaSetDevice(dev));
 cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop,dev));
 print_timestamp(); printf("Using device %d: %s (%d SMs, %zuMB)\n",dev,prop.name,prop.multiProcessorCount,prop.totalGlobalMem/(1024*1024));
 uint *d_rfc_ok=nullptr, h_rfc_ok=0; CUDA_CHECK(cudaMalloc((void**)&d_rfc_ok,sizeof(uint)));
 diagnostic_rfc6979_testcase<<<1,1>>>(d_rfc_ok); CUDA_CHECK(cudaGetLastError());
 CUDA_CHECK(cudaMemcpy(&h_rfc_ok,d_rfc_ok,sizeof(uint),cudaMemcpyDeviceToHost)); cudaFree(d_rfc_ok);
 if(!h_rfc_ok){fprintf(stderr,"Fatal: optimized RFC6979 extra-entropy self-test failed.\n");return 1;}
 print_timestamp(); printf("RFC6979 extra-entropy fast path self-test passed.\n");
 {
   const uchar sk[32]={0xf9,0x82,0x88,0xd3,0x43,0x7e,0x27,0x4c,0xf7,0xf5,0xc6,0x48,0x46,0x55,0x50,0x50,0x63,0x24,0xbb,0xc5,0x57,0x30,0x45,0x0e,0x06,0x05,0x5a,0xb0,0xbb,0x0e,0x08,0x79};
   const uchar msg[32]={0x61,0xd1,0x75,0xca,0x77,0xd5,0xc7,0x07,0x51,0xcb,0x1d,0x23,0x6d,0x73,0xea,0x8f,0xa4,0xab,0xe9,0x71,0x5f,0xf8,0xfa,0x2c,0x88,0xef,0x88,0xb0,0x8a,0xad,0xef,0xb4};
   const uchar exp_r[32]={0x8f,0xe4,0x4a,0x25,0xda,0x47,0x8b,0xa9,0x1f,0x0e,0x9c,0x21,0x73,0xa7,0x3e,0xf2,0x4b,0xab,0xa1,0x61,0xda,0xc0,0xae,0xd6,0x42,0x50,0x92,0x12,0x49,0x21,0x63,0x98};
   const uchar exp_s[32]={0x31,0xbd,0x1d,0xc5,0xf7,0x21,0x5b,0x84,0x57,0x70,0xba,0x90,0xd3,0x8b,0x33,0x16,0x10,0xc1,0xc5,0xb5,0x5e,0x14,0xd5,0x81,0x0e,0x8e,0x54,0x24,0x0a,0x2c,0x98,0xf2};
   uchar *dsk=nullptr,*dmsg=nullptr,*dr=nullptr,*ds=nullptr; uint *dflags=nullptr;
   CUDA_CHECK(cudaMalloc((void**)&dsk,32)); CUDA_CHECK(cudaMalloc((void**)&dmsg,32));
   CUDA_CHECK(cudaMalloc((void**)&dr,32)); CUDA_CHECK(cudaMalloc((void**)&ds,32)); CUDA_CHECK(cudaMalloc((void**)&dflags,sizeof(uint)));
   CUDA_CHECK(cudaMemcpy(dsk,sk,32,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dmsg,msg,32,cudaMemcpyHostToDevice));
   diagnostic_mining_ecdsa<<<1,1>>>(dsk,dmsg,0x240f9e24u,dr,ds,dflags);
   CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
   uchar hr[32],hs[32]; uint flags=0;
   CUDA_CHECK(cudaMemcpy(hr,dr,32,cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(hs,ds,32,cudaMemcpyDeviceToHost));
   CUDA_CHECK(cudaMemcpy(&flags,dflags,sizeof(uint),cudaMemcpyDeviceToHost));
   cudaFree(dsk); cudaFree(dmsg); cudaFree(dr); cudaFree(ds); cudaFree(dflags);
   print_timestamp();
   printf("ECDSA diag flags=%u mul3x7=%s fermat2=%s finish=%s r=%s s=%s\n",
          flags, (flags&1)?"ok":"FAIL", (flags&2)?"ok":"FAIL", (flags&4)?"ok":"FAIL",
          memcmp(hr,exp_r,32)==0?"ok":"FAIL", memcmp(hs,exp_s,32)==0?"ok":"FAIL");
   if(memcmp(hr,exp_r,32)!=0 || memcmp(hs,exp_s,32)!=0){
     printf("  got_r="); for(int i=0;i<32;i++) printf("%02x",hr[i]); printf("\n");
     printf("  got_s="); for(int i=0;i<32;i++) printf("%02x",hs[i]); printf("\n");
   }
   if(!(flags&1) || !(flags&2) || !(flags&4) || memcmp(hr,exp_r,32)!=0 || memcmp(hs,exp_s,32)!=0){
     fprintf(stderr,"Fatal: mining ECDSA path does not match libsecp256k1.\n");
     return 1;
   }
 }
 cudaFuncAttributes attr{}; CUDA_CHECK(cudaFuncGetAttributes(&attr,btcw_mine));
 print_timestamp(); printf("=== CUDA KERNEL RESOURCE PROFILE ===\n");
 printf("Max threads/block          : %d\n",attr.maxThreadsPerBlock); printf("Registers/thread           : %d\n",attr.numRegs); printf("Static shared memory       : %zu bytes\n",attr.sharedSizeBytes); printf("Local spill bytes/thread   : %zu bytes\n",attr.localSizeBytes); printf("Compute capability         : %d.%d\n",prop.major,prop.minor); printf("====================================\n");

 const size_t EN[6]={8388608ULL,8388608ULL,8388608ULL,8388608ULL,8388608ULL,256ULL};
 ulong* dtab[6]={};
 print_timestamp(); printf("Allocating/precomputing GLV W24/W9 generator table (~2560 MiB)...\n");
 for(int i=0;i<6;i++){ size_t bytes=EN[i]*8ULL*sizeof(ulong); CUDA_CHECK(cudaMalloc((void**)&dtab[i],bytes)); uint entries=(uint)EN[i]; int t=256; size_t b=(EN[i]+t-1)/t; printf("  group %d/6: %u entries (%zu MiB)\n",i+1,entries,bytes/(1024*1024)); precompute_ecmult_gen_table<<<(unsigned)b,t>>>(dtab[i],(uint)i,entries); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); }
 print_timestamp(); printf("Ecmult table ready (GLV W24/W9, 6 groups, ~2560 MiB).\n");

 uchar *dkey=nullptr,*dhash=nullptr,*dtarget=nullptr; ulong* dnonce=nullptr; uint *dfound=nullptr,*dctr=nullptr;
 CUDA_CHECK(cudaMalloc((void**)&dkey,32)); CUDA_CHECK(cudaMalloc((void**)&dhash,32)); CUDA_CHECK(cudaMalloc((void**)&dtarget,32)); CUDA_CHECK(cudaMalloc((void**)&dnonce,sizeof(ulong))); CUDA_CHECK(cudaMalloc((void**)&dfound,sizeof(uint))); CUDA_CHECK(cudaMalloc((void**)&dctr,sizeof(uint)));

#ifdef _WIN32
 HANDLE mapping=CreateFileMappingA(INVALID_HANDLE_VALUE,nullptr,PAGE_READWRITE,0,(DWORD)sizeof(SharedData),SHM_NAME);
 if(!mapping){fprintf(stderr,"CreateFileMapping('%s') failed (%lu)\n",SHM_NAME,GetLastError());return 1;}
 SharedData* shared=(SharedData*)MapViewOfFile(mapping,FILE_MAP_ALL_ACCESS,0,0,sizeof(SharedData));
 if(!shared){fprintf(stderr,"MapViewOfFile failed (%lu)\n",GetLastError());CloseHandle(mapping);return 1;}
#else
 int shm_fd=shm_open(SHM_NAME,O_RDWR,0666); if(shm_fd==-1){ shm_fd=shm_open(SHM_NAME,O_CREAT|O_RDWR,0666); if(shm_fd==-1){perror("shm_open");return 1;} if(ftruncate(shm_fd,sizeof(SharedData))==-1){perror("ftruncate");return 1;} }
 SharedData* shared=(SharedData*)mmap(nullptr,sizeof(SharedData),PROT_READ|PROT_WRITE,MAP_SHARED,shm_fd,0); if(shared==MAP_FAILED){perror("mmap");return 1;}
#endif
 print_timestamp(); printf("Shared memory '%s' mapped successfully.\n",SHM_NAME);

 size_t block=user_block?user_block:128; if(block>1024||block==0){fprintf(stderr,"Invalid CUDA block size %zu\n",block);return 1;}
 size_t work=user_work?user_work:(size_t)prop.multiProcessorCount*9728ULL; if(work<65536)work=65536; if(work>4194304)work=4194304; if(work%block)work=((work+block-1)/block)*block;
 print_timestamp(); printf("Work size: %zu%s\n",work,user_work?" (manual override)":" (v40 tuned mapping)"); print_timestamp(); printf("CUDA block size: %zu%s\n",block,user_block?" (manual override)":"");
 constexpr size_t SIGN_BATCH_HOST=SIGN_BATCH; size_t scratch_bytes=work*SIGN_BATCH_HOST*sizeof(Scalar); Scalar* dscratch=nullptr; FieldElement* drxscratch=nullptr; CUDA_CHECK(cudaMalloc((void**)&dscratch,scratch_bytes)); if(BTCW_HYBRID_RX)CUDA_CHECK(cudaMalloc((void**)&drxscratch,scratch_bytes)); print_timestamp(); printf("K/Rx scratch: %.2f / %.2f GiB global\n",(double)scratch_bytes/(1024.0*1024.0*1024.0),drxscratch?(double)scratch_bytes/(1024.0*1024.0*1024.0):0.0);

 // Fixed Stage-2 target: 28 leading zero bits in the displayed/arith256 hash.
 // hash_meets_target_le() compares the SHA256d bytes as a little-endian uint256,
 // so this is (2^228 - 1): ff..ff 0f 00 00 00.
 uint8_t hkey[32]={}, hhash[32]={}, prevhash[32]={};
 uint8_t htarget[32];
 memset(htarget, 0xFF, 28);
 htarget[28]=0x0F; htarget[29]=0; htarget[30]=0; htarget[31]=0;
 CUDA_CHECK(cudaMemcpy(dtarget,htarget,32,cudaMemcpyHostToDevice)); bool havehash=false,was_connected=false,conn_printed=false,disconnect_timing=false; auto disconnect_start=std::chrono::steady_clock::now(); const int DISCONNECT_SECONDS=3; int block_transitions=0; uint64_t nonce_prev=1234,hashlow=0,nonce_base=0; uint32_t throttle=0; auto session_start=std::chrono::steady_clock::now();
 print_timestamp(); printf("GPU initialized - waiting for block data...\n");
 while(g_running.load()){
   uint64_t changeCount=0; auto start=std::chrono::steady_clock::now();
   while(g_running.load() && std::chrono::steady_clock::now()-start<std::chrono::seconds(2)){
     if((throttle%3)==0){ memcpy(hkey,(const void*)&shared->data[0],32); memcpy(hhash,(const void*)&shared->data[192],32); if(!havehash){memcpy(prevhash,hhash,32);havehash=true;} else if(memcmp(hhash,prevhash,32)!=0){memcpy(prevhash,hhash,32);block_transitions++;nonce_base=0;shared->nonce=SENTINEL_NONCE;nonce_prev=SENTINEL_NONCE;print_timestamp();printf("New block data from node (block #%d this session)\n",block_transitions);} CUDA_CHECK(cudaMemcpyAsync(dkey,hkey,32,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpyAsync(dhash,hhash,32,cudaMemcpyHostToDevice)); }
     throttle++;
     CUDA_CHECK(cudaMemsetAsync(dnonce,0,sizeof(ulong))); CUDA_CHECK(cudaMemsetAsync(dfound,0,sizeof(uint))); CUDA_CHECK(cudaMemsetAsync(dctr,0,sizeof(uint)));
     size_t blocks=work/block;
     btcw_mine<<<(unsigned)blocks,(unsigned)block>>>(dkey,dhash,dnonce,dfound,dctr,(ulong)nonce_base,(uint)gpu_num,dtab[0],dtab[1],dtab[2],dtab[3],dtab[4],dtab[5],dtarget,dscratch,drxscratch);
     CUDA_CHECK(cudaGetLastError());
     uint result_found=0,ctr=0; ulong result_nonce=0; CUDA_CHECK(cudaMemcpy(&result_found,dfound,sizeof(uint),cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(&result_nonce,dnonce,sizeof(ulong),cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(&ctr,dctr,sizeof(uint),cudaMemcpyDeviceToHost)); changeCount+=ctr;
     if(result_found){shared->nonce=result_nonce;nonce_prev=result_nonce;}
     nonce_base = (nonce_base + work*128ULL) & 0xFFFFFFFFULL; if(nonce_base==0) nonce_base=1;
     memcpy(&hashlow,(const void*)&shared->data[192],8);
     if(hashlow==0){ if(!disconnect_timing){disconnect_start=std::chrono::steady_clock::now();disconnect_timing=true;} if(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now()-disconnect_start).count()>=DISCONNECT_SECONDS){ if(conn_printed||!was_connected){print_timestamp();printf("!!! NOT CONNECTED TO BTCW NODE WALLET !!! Make sure your wallet has at least 1 utxo.\n");conn_printed=false;} std::this_thread::sleep_for(std::chrono::seconds(1)); }} else { if(!was_connected||!conn_printed){print_timestamp();printf("Connected to BTCW node wallet\n");conn_printed=true;} disconnect_timing=false;was_connected=true; }
   }
   double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(); double mh=(elapsed>0)?(double)changeCount/elapsed/1e6:0; auto up=std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now()-session_start).count(); print_timestamp(); printf("Mining | %.2f MH/s | Nonce: %016llx | Blocks: %d | Up: %02lld:%02lld:%02lld\n",mh,(unsigned long long)shared->nonce,block_transitions,(long long)(up/3600),(long long)((up/60)%60),(long long)(up%60)); fflush(stdout);
 }
 print_timestamp(); printf("Shutting down...\n");
 cudaFree(drxscratch); cudaFree(dscratch); cudaFree(dkey); cudaFree(dhash); cudaFree(dtarget); cudaFree(dnonce); cudaFree(dfound); cudaFree(dctr); for(auto p:dtab)cudaFree(p);
#ifdef _WIN32
 UnmapViewOfFile((void*)shared); CloseHandle(mapping);
#else
 munmap(shared,sizeof(SharedData)); close(shm_fd);
#endif
 print_timestamp();printf("Goodbye.\n"); return 0;
}
