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
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <csignal>

#include "btcw_cuda_kernels.cuh"

static std::atomic<bool> g_running{true};
static void signal_handler(int){ g_running.store(false); }
static void print_timestamp(){ time_t now=time(nullptr); tm* lt=localtime(&now); char b[16]; strftime(b,sizeof(b),"%H:%M:%S",lt); printf("[%s] ",b); }
#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__); return 1; } } while(0)
#define SHM_NAME "/shared_mem"
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

 int shm_fd=shm_open(SHM_NAME,O_RDWR,0666); if(shm_fd==-1){ shm_fd=shm_open(SHM_NAME,O_CREAT|O_RDWR,0666); if(shm_fd==-1){perror("shm_open");return 1;} if(ftruncate(shm_fd,sizeof(SharedData))==-1){perror("ftruncate");return 1;} }
 SharedData* shared=(SharedData*)mmap(nullptr,sizeof(SharedData),PROT_READ|PROT_WRITE,MAP_SHARED,shm_fd,0); if(shared==MAP_FAILED){perror("mmap");return 1;}
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
 cudaFree(drxscratch); cudaFree(dscratch); cudaFree(dkey); cudaFree(dhash); cudaFree(dtarget); cudaFree(dnonce); cudaFree(dfound); cudaFree(dctr); for(auto p:dtab)cudaFree(p); munmap(shared,sizeof(SharedData)); close(shm_fd); print_timestamp();printf("Goodbye.\n"); return 0;
}
