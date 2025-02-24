// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target
// REQUIRES: libgcc
// UNSUPPORTED: system-windows

// RUN: %clang -c -### -target x86_64-unknown-linux-gnu --cuda-gpu-arch=gfx900 \
// RUN:   -std=c++11 --rocm-path=%S/Inputs/rocm -nogpulib %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=COMMON,CLANG,HIP %s

// RUN: %clang -c -### -target x86_64-unknown-linux-gnu --cuda-gpu-arch=gfx900 \
// RUN:   -std=c++11 --rocm-path=%S/Inputs/rocm -nobuiltininc -nogpulib %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=COMMON,NOCLANG,HIP %s

// RUN: %clang -c -### -target x86_64-unknown-linux-gnu --cuda-gpu-arch=gfx900 \
// RUN:   -std=c++11 --rocm-path=%S/Inputs/rocm -nogpuinc -nogpulib %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=COMMON,CLANG,NOHIP %s

// COMMON: InstalledDir: [[ROOT:[^"]*]]/bin

// COMMON-LABEL: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CLANG-SAME: "-internal-isystem" "[[ROOT]]/lib/clang/{{.*}}/include/cuda_wrappers"
// NOCLANG-NOT: "[[ROOT]]/lib/clang/{{.*}}/include/cuda_wrappers"
// HIP-SAME: "-internal-isystem" "{{.*}}Inputs/rocm/include"
// NOHIP-NOT: "{{.*}}Inputs/rocm/include"
// HIP-SAME: "-include" "__clang_hip_runtime_wrapper.h"
// NOHIP-NOT: "-include" "__clang_hip_runtime_wrapper.h"
// skip check of standard C++ include path
// CLANG-SAME: "-internal-isystem" "[[ROOT]]/lib/clang/{{.*}}/include"
// NOCLANG-NOT: "[[ROOT]]/lib/clang/{{.*}}/include"

// COMMON-LABEL: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CLANG-SAME: "-internal-isystem" "[[ROOT]]/lib/clang/{{.*}}/include/cuda_wrappers"
// NOCLANG-NOT: "[[ROOT]]/lib/clang/{{.*}}/include/cuda_wrappers"
// HIP-SAME: "-internal-isystem" "{{.*}}Inputs/rocm/include"
// NOHIP-NOT: "{{.*}}Inputs/rocm/include"
// HIP-SAME: "-include" "__clang_hip_runtime_wrapper.h"
// NOHIP-NOT: "-include" "__clang_hip_runtime_wrapper.h"
// skip check of standard C++ include path
// CLANG-SAME: "-internal-isystem" "[[ROOT]]/lib/clang/{{.*}}/include"
// NOCLANG-NOT: "[[ROOT]]/lib/clang/{{.*}}/include"

// RUN: %clang -c -### -target x86_64-unknown-linux-gnu --cuda-gpu-arch=gfx900 \
// RUN:   -std=c++11 --rocm-path=%S/Inputs/rocm -nogpulib %s 2>&1 \
// RUN:   --hip-version=3.5 | FileCheck -check-prefixes=ROCM35 %s

// ROCM35: InstalledDir: [[ROOT:[^"]*]]/bin

// ROCM35-LABEL: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// ROCM35-NOT: "[[ROOT]]/lib/clang/{{.*}}/include/cuda_wrappers"
// ROCM35-SAME: "-internal-isystem" "[[ROOT]]/lib/clang/{{[^"]*}}"
// ROCM35-SAME: "-internal-isystem" "{{[^"]*}}Inputs/rocm/include"
// ROCM35-NOT: "-include" "__clang_hip_runtime_wrapper.h"
// skip check of standard C++ include path
// ROCM35-SAME: "-internal-isystem" "[[ROOT]]/lib/clang/{{.*}}/include"
