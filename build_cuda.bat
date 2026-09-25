@echo off
setlocal
cd /d "%~dp0"
if not exist release mkdir release
if "%CUDA_ARCH%"=="" set CUDA_ARCH=sm_120
if "%CUDA_MAX_REGS%"=="" set CUDA_MAX_REGS=160
if "%CUDA_SIGN_BATCH%"=="" set CUDA_SIGN_BATCH=128
set OUTPUT=release\btcw_cuda_miner_batch%CUDA_SIGN_BATCH%.exe
echo Building %OUTPUT% for %CUDA_ARCH% maxrregcount=%CUDA_MAX_REGS% sign_batch=%CUDA_SIGN_BATCH%
nvcc -O3 -std=c++17 -arch=%CUDA_ARCH% --maxrregcount %CUDA_MAX_REGS% -DBTCW_SIGN_BATCH=%CUDA_SIGN_BATCH% -Xptxas=-v,-warn-spills btcw_cuda_miner.cu -o "%OUTPUT%"
if errorlevel 1 exit /b 1
copy /y "%OUTPUT%" release\btcw_cuda_miner.exe >nul
echo Built %OUTPUT%
endlocal
