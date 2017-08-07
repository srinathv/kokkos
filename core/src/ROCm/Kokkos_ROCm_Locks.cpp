/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <Kokkos_Macros.hpp>

#ifdef KOKKOS_ENABLE_ROCM

#include <ROCm/Kokkos_ROCm_Locks.hpp>
//#include <ROCm/Kokkos_ROCm_Error.hpp>
#include <Kokkos_Core.hpp>


namespace Kokkos {
namespace Impl {

void * rocm_device_allocate(int);
void * rocm_hostpinned_allocate(int);
void rocm_device_free(void * );
void rocm_device_synchronize();
void rocm_copy( const void *, void *, size_t );

//ROCmLockArrays * g_device_rocm_lock_arrays;
ROCmLockArrays g_host_rocm_lock_arrays = { nullptr, nullptr, 0 };

void initialize_host_rocm_lock_arrays() {
  int rocm_concurrency = Kokkos::Experimental::ROCm::concurrency();

  if (g_host_rocm_lock_arrays.atomic != nullptr) return;
  hc::accelerator acc;
  g_device_rocm_lock_arrays = 
         (ROCmLockArrays *) hc::am_alloc( sizeof(ROCmLockArrays), acc, 2 );
//         (__attribute__((address_space (3))) ROCmLockArrays *) rocm_device_allocate( sizeof(ROCmLockArrays) );
printf("g_device_rocm_lock_arrays = %p\n",g_device_rocm_lock_arrays);
  g_host_rocm_lock_arrays.atomic = 
         ( std::int32_t *) rocm_device_allocate( 
                             sizeof(int) * (ROCM_SPACE_ATOMIC_MASK+1));
printf("atomic                    = %p\n",g_host_rocm_lock_arrays.atomic);
  g_host_rocm_lock_arrays.scratch = 
         ( std::int32_t *) rocm_device_allocate( 
                             sizeof(int) * rocm_concurrency);
  g_host_rocm_lock_arrays.n = rocm_concurrency;
//  KOKKOS_COPY_ROCM_LOCK_ARRAYS_TO_DEVICE();
 Kokkos::Impl::rocm_copy( 
        & Kokkos::Impl::g_host_rocm_lock_arrays ,
        Kokkos::Impl::g_device_rocm_lock_arrays ,
        sizeof(Kokkos::Impl::ROCmLockArrays) ) ;

  rocm_device_synchronize();
  hc::parallel_for_each(hc::extent<1>(ROCM_SPACE_ATOMIC_MASK+1), 
      [=] (const hc::index<1> & idx) __HC__ {
     unsigned i = idx[0];
     if(i<ROCM_SPACE_ATOMIC_MASK+1) 
        Kokkos::Impl::g_device_rocm_lock_arrays->atomic[i] = 0x55;
  });
  hc::parallel_for_each(hc::extent<1>(rocm_concurrency), 
      [=] (const hc::index<1> & idx) __HC__ {
     unsigned i = idx[0];
     if(i<ROCM_SPACE_ATOMIC_MASK+1) 
        Kokkos::Impl::g_device_rocm_lock_arrays->scratch[i] = 0xaa;
  });
  rocm_device_synchronize();
printf("atomic                    = %p\n",g_device_rocm_lock_arrays->atomic);
printf("scratch                   = %p\n",g_device_rocm_lock_arrays->scratch);
}

void finalize_host_rocm_lock_arrays() {
  if (g_host_rocm_lock_arrays.atomic == nullptr) return;
  rocm_device_free(g_host_rocm_lock_arrays.atomic);
  g_host_rocm_lock_arrays.atomic = nullptr;
  rocm_device_free(g_host_rocm_lock_arrays.scratch);
  g_host_rocm_lock_arrays.scratch = nullptr;
  g_host_rocm_lock_arrays.n = 0;

  KOKKOS_COPY_ROCM_LOCK_ARRAYS_TO_DEVICE();
  rocm_device_synchronize();
}

} // namespace Impl

} // namespace Kokkos

#else

void KOKKOS_CORE_SRC_ROCM_ROCM_LOCKS_PREVENT_LINK_ERROR() {}

#endif
