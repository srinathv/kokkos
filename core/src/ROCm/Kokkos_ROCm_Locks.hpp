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

#ifndef KOKKOS_ROCM_LOCKS_HPP
#define KOKKOS_ROCM_LOCKS_HPP

#include <Kokkos_Macros.hpp>

#ifdef KOKKOS_ENABLE_ROCM
#include <hc.hpp>

#include <cstdint>

//#include <ROCm/Kokkos_ROCm_Error.hpp>

namespace Kokkos {
namespace Impl {

struct ROCmLockArrays {
  std::int32_t* atomic;
  std::int32_t* scratch;
  std::size_t n;
} ;

/// \brief This global variable in Host space is the central definition
///        of these arrays.
extern Kokkos::Impl::ROCmLockArrays g_host_rocm_lock_arrays ;

/// \brief After this call, the g_host_rocm_lock_arrays variable has
///        valid, initialized arrays.
///
/// This call is idempotent.
void initialize_host_rocm_lock_arrays();

/// \brief After this call, the g_host_rocm_lock_arrays variable has
///        all null pointers, and all array memory has been freed.
///
/// This call is idempotent.
void finalize_host_rocm_lock_arrays();

} // namespace Impl
} // namespace Kokkos

#if defined( __HCC__ )

namespace Kokkos {
namespace Impl {

/// \brief This global variable in ROCM space is what kernels use
///        to get access to the lock arrays.
///
/// When relocatable device code is enabled, there can be one single
/// instance of this global variable for the entire executable,
/// whose definition will be in Kokkos_ROCm_Locks.cpp (and whose declaration
/// here must then be extern.
/// This one instance will be initialized by initialize_host_rocm_lock_arrays
/// and need not be modified afterwards.
///
/// When relocatable device code is disabled, an instance of this variable
/// will be created in every translation unit that sees this header file
/// (we make this clear by marking it static, meaning no other translation
///  unit can link to it).
/// Since the Kokkos_ROCm_Locks.cpp translation unit cannot initialize the
/// instances in other translation units, we must update this ROCM global
/// variable based on the Host global variable prior to running any kernels
/// that will use it.
/// That is the purpose of the KOKKOS_ENSURE_ROCM_LOCK_ARRAYS_ON_DEVICE macro.
#if !defined(ROCM_LOCK_ARRAYS_DEFINED)
ROCmLockArrays * g_device_rocm_lock_arrays ;
#define ROCM_LOCK_ARRAYS_DEFINED
#endif
#define ROCM_SPACE_ATOMIC_MASK 0x1FFFF

/// \brief Aquire a lock for the address
///
/// This function tries to aquire the lock for the hash value derived
/// from the provided ptr. If the lock is successfully aquired the
/// function returns true. Otherwise it returns false.
KOKKOS_INLINE_FUNCTION
bool lock_address_rocm_space(void* ptr) {
ROCmLockArrays * gsptr = (ROCmLockArrays *)hc::get_group_segment_base_pointer();
  std::size_t offset = (std::size_t)(ptr);
  offset = offset >> 2;
  offset = offset & ROCM_SPACE_ATOMIC_MASK;
//  return (0 == hc::atomic_compare_exchange(&Kokkos::Impl::g_device_rocm_lock_arrays->atomic[offset],0,1));
  return (0 == hc::atomic_compare_exchange(&gsptr->atomic[offset],0,1));
  return (true);
}

/// \brief Release lock for the address
///
/// This function releases the lock for the hash value derived
/// from the provided ptr. This function should only be called
/// after previously successfully aquiring a lock with
/// lock_address.
KOKKOS_INLINE_FUNCTION
void unlock_address_rocm_space(void* ptr) {
ROCmLockArrays * gsptr = (ROCmLockArrays *)hc::get_group_segment_base_pointer();
  std::size_t offset = (std::size_t)(ptr);
  offset = offset >> 2;
  offset = offset & ROCM_SPACE_ATOMIC_MASK;
//  hc::atomic_exchange(&Kokkos::Impl::g_device_rocm_lock_arrays->atomic[ offset ], 0);
  hc::atomic_exchange(&gsptr->atomic[ offset ], 0);
}
// the following subroutine demonstrates how to get around a compiler issue
// The compiler (1.6.1) won't properly handle
// Kokkos::Impl::g_device_rocm_lock_arrays->atomic[ offset ]
// because it doesn't dereference the host container g_device_rocm_lock_arrays
// which contains the gpu address for the lock arrays structure
// So we have to use local data store to store the lock array pointers 
// (which are allocated in device memory) when the kernel is started up
//  in Kokkos_ROCm_Parallel.hpp.
// 
KOKKOS_INLINE_FUNCTION
std::size_t return_address_rocm_space(void* ptr) {
void * gsptr = hc::get_group_segment_base_pointer();
  ROCmLockArrays ** iptr = ((ROCmLockArrays **)gsptr);
//  std::int32_t* atomic_lock_array = (std::int32_t*)*iptr;  // this works
  std::int32_t** atomic_lock_array = (std::int32_t**)*iptr;  // this works
//  void * atomic_lock_array = (void*)*iptr;  // this works
//  std::int32_t* scratch_lock_array = (std::int32_t*)*(iptr+sizeof(std::int32_t *));  // this works
//  std::size_t lock_array_n = (std::size_t)*(iptr+2*sizeof(std::int32_t *));  // this works
//  return (std::size_t)atomic_lock_array;  //as expected
std::size_t * val = (std::size_t *)*atomic_lock_array;
//  return (std::size_t)atomic_lock_array;  
//std::size_t val = (std::size_t)&(*((std::int32_t *)atomic_lock_array));
  return (std::size_t)val;
//  return (std::size_t)scratch_lock_array;
//  return lock_array_n;
//  std::size_t offset = (std::size_t)(ptr);
//  offset = offset >> 2;
//  offset = offset & ROCM_SPACE_ATOMIC_MASK;
//  return (std::size_t)(atomic_lock_array + offset);
//  return (std::size_t)*(atomic_lock_array + offset);
// if the following line worked, we wouldn't need the contortions above.
//  return (std::size_t)&Kokkos::Impl::g_device_rocm_lock_arrays->atomic[ offset ];
//  return offset;
}

} // namespace Impl
} // namespace Kokkos

/* Dan Ibanez: it is critical that this code be a macro, so that it will
   capture the right address for Kokkos::Impl::g_device_rocm_lock_arrays!
   putting this in an inline function will NOT do the right thing! */
#define KOKKOS_COPY_ROCM_LOCK_ARRAYS_TO_DEVICE() \
{ \
        Kokkos::Impl::rocm_copy( \
        & Kokkos::Impl::g_host_rocm_lock_arrays , \
        Kokkos::Impl::g_device_rocm_lock_arrays , \
        sizeof(Kokkos::Impl::ROCmLockArrays) ) ; \
}

#define KOKKOS_ENSURE_ROCM_LOCK_ARRAYS_ON_DEVICE() KOKKOS_COPY_ROCM_LOCK_ARRAYS_TO_DEVICE()

#endif /* defined( __HCC__ ) */

#endif /* defined( KOKKOS_ENABLE_ROCM ) */

#endif /* #ifndef KOKKOS_ROCM_LOCKS_HPP */
