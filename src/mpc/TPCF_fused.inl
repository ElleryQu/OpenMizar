/*
 * TPCF.inl
 */

#pragma once

#include "TPCF.h"

#include <bitset>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>

#include "../gpu/bitwise.cuh"
#include "../gpu/convolution.cuh"
#include "../gpu/conv.cuh"
#include "../gpu/DeviceData.h"
#include "../gpu/functors.cuh"
#include "../gpu/matrix.cuh"
#include "../gpu/gemm.cuh"
#include "../gpu/StridedRange.cuh"
#include "../globals.h"
#include "Precompute.h"
#include "../util/functors.h"
#include "../util/Profiler.h"

extern Precompute PrecomputeObject;
extern Profiler comm_profiler;
extern Profiler func_profiler;


// TPCF class implementation 

template<typename T, typename I>
TPCFBase<T, I>::TPCFBase(DeviceData<T, I> *a) : 
                share_(a) {}

template<typename T, typename I>
void TPCFBase<T, I>::set(DeviceData<T, I> *a) {
    share_ = a;
}

template<typename T, typename I>
size_t TPCFBase<T, I>::size() const {
    return share_->size();
}

template<typename T, typename I>
void TPCFBase<T, I>::zero() {
    share_->zero();
};

template<typename T, typename I>
void TPCFBase<T, I>::fill(T val) {
    share_->fill(partyNum == PARTY_A ? val : 0);
}

template<typename T, typename I>
void TPCFBase<T, I>::setPublic(std::vector<double> &v) {
    std::vector<T> shifted_vals;
    for (double f : v) {
        shifted_vals.push_back((T) (f * (1 << FLOAT_PRECISION)));
    }

    switch (partyNum) {
        case PARTY_A:
            thrust::copy(shifted_vals.begin(), shifted_vals.end(), share_->begin());
            break;
        case PARTY_B:
            share_->zero();
            break;
    }
};

template<typename T, typename I>
DeviceData<T, I> *TPCFBase<T, I>::getShare(int i) {
    switch (i) {
        case 0:
            return share_;
        default:
            return nullptr;
    }
}

template<typename T, typename I>
const DeviceData<T, I> *TPCFBase<T, I>::getShare(int i) const {
    switch (i) {
        case 0:
            return share_;
        default:
            return nullptr;
    }
}

template<typename T, typename I>
TPCFBase<T, I> &TPCFBase<T, I>::operator+=(const T rhs) {
    auto op_ = [rhs] __host__ __device__ (T input) {
        T res = input + rhs;
        return res - (res >= TPCFPRIME) * TPCFPRIME;
    };
    if (partyNum == PARTY_A) apply(*this->getShare(0), op_);
    return *this;
}

template<typename T, typename I>
TPCFBase<T, I> &TPCFBase<T, I>::operator-=(const T rhs) {
    auto op_ = [rhs] __host__ __device__ (T input) {
        T res = input + TPCFPRIME - rhs;
        return res - (res >= TPCFPRIME) * TPCFPRIME;
    };
    if (partyNum == PARTY_A) apply(*this->getShare(0), op_);
    return *this;
}

template<typename T, typename I>
TPCFBase<T, I> &TPCFBase<T, I>::operator*=(const T rhs) {
    auto op_ = [rhs] __host__ __device__ (T input) {
        T res = input * rhs;
        res = (res >> TPCFPRIMEBW) + (res & TPCFPRIME);
        return res - (res >= TPCFPRIME) * TPCFPRIME;
    };
    apply(*this->getShare(0), op_);
    return *this;
}

template<typename T, typename I>
TPCFBase<T, I> &TPCFBase<T, I>::operator>>=(const T rhs) {
    *share_ >>= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
TPCFBase<T, I> &TPCFBase<T, I>::operator+=(const DeviceData<T, I2> &rhs) {
    auto op_ = [] __host__ __device__ (T lhs, T rhs) {
        T res = lhs + rhs;
        return res - (res >= TPCFPRIME) * TPCFPRIME;
    };
    if (partyNum == PARTY_A) apply(*this->getShare(0), rhs, op_);
    return *this;
}

template<typename T, typename I>
template<typename I2>
TPCFBase<T, I> &TPCFBase<T, I>::operator-=(const DeviceData<T, I2> &rhs) {
    auto op_ = [] __host__ __device__ (T lhs, T rhs) {
        T res = lhs + TPCFPRIME - rhs;
        return res - (res >= TPCFPRIME) * TPCFPRIME;
    };
    if (partyNum == PARTY_A) apply(*this->getShare(0), rhs, op_);
    return *this;
}

template<typename T, typename I>
template<typename I2>
TPCFBase<T, I> &TPCFBase<T, I>::operator*=(const DeviceData<T, I2> &rhs) {
    auto op_ = [] __host__ __device__ (T lhs, T rhs) {
        T res = lhs * rhs;
        res = (res >> TPCFPRIMEBW) + (res & TPCFPRIME);
        return res - (res >= TPCFPRIME) * TPCFPRIME;
    };
    apply(*this->getShare(0), rhs, op_);
    return *this;
}

template<typename T, typename I>
template<typename I2>
TPCFBase<T, I> &TPCFBase<T, I>::operator^=(const DeviceData<T, I2> &rhs) {
    if (partyNum == PARTY_A) {
        *share_ ^= rhs;
    }
    return *this;
}

template<typename T, typename I>
template<typename I2>
TPCFBase<T, I> &TPCFBase<T, I>::operator&=(const DeviceData<T, I2> &rhs) {
    *share_ &= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
TPCFBase<T, I> &TPCFBase<T, I>::operator>>=(const DeviceData<T, I2> &rhs) {
    *share_ >>= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
TPCFBase<T, I> &TPCFBase<T, I>::operator<<=(const DeviceData<T, I2> &rhs) {
    *share_ <<= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
TPCFBase<T, I> &TPCFBase<T, I>::operator+=(const TPCFBase<T, I2> &rhs) {
    auto op_ = [] __host__ __device__ (T lhs, T rhs) {
        T res = lhs + rhs;
        return res - (res >= TPCFPRIME) * TPCFPRIME;
    };
    apply(*this->getShare(0), *rhs.getShare(0), op_);
    return *this;
}

template<typename T, typename I>
template<typename I2>
TPCFBase<T, I> &TPCFBase<T, I>::operator-=(const TPCFBase<T, I2> &rhs) {
    auto op_ = [] __host__ __device__ (T lhs, T rhs) {
        T res = lhs + TPCFPRIME - rhs;
        return res - (res >= TPCFPRIME) * TPCFPRIME;
    };
    apply(*this->getShare(0), *rhs.getShare(0), op_);
    return *this;
}

template<typename T, typename I>
template<typename I2>
TPCFBase<T, I> &TPCFBase<T, I>::operator*=(const TPCFBase<T, I2> &rhs) {

    throw std::runtime_error("Not implemented");
 
    return *this;
}

template<typename T, typename I>
template<typename I2>
TPCFBase<T, I> &TPCFBase<T, I>::operator^=(const TPCFBase<T, I2> &rhs) {
    *share_ ^= *rhs.getShare(0);
    return *this;
}

template<typename T, typename I>
template<typename I2>
TPCFBase<T, I> &TPCFBase<T, I>::operator&=(const TPCFBase<T, I2> &rhs) {

    throw std::runtime_error("Not implemented");
 
    return *this;
}

//TO_BE_DONE
template<typename T, typename I>
int TPCFBase<T, I>::otherParty(int party) {
	switch(party) {
        case PARTY_A:
            return PARTY_B;
        default: // PARTY_B
            return PARTY_A;
    }	
}

template<typename T, typename I>
int TPCFBase<T, I>::numShares() {
    return 1;
}

template<typename T, typename I>
TPCF<T, I>::TPCF(DeviceData<T, I> *a) : TPCFBase<T, I>(a) {}

template<typename T>
TPCF<T, BufferIterator<T> >::TPCF(DeviceData<T> *a) :
    TPCFBase<T, BufferIterator<T> >(a) {}

template<typename T>
TPCF<T, BufferIterator<T> >::TPCF(size_t n) :
    share_buffer_(n),
    TPCFBase<T, BufferIterator<T> >(&share_buffer_) {}

template<typename T>
TPCF<T, BufferIterator<T> >::TPCF(std::initializer_list<double> il, bool convertToFixedPoint) :
    share_buffer_(il.size()),
    TPCFBase<T, BufferIterator<T> >(&share_buffer_) {

    std::vector<T> shifted_vals;
    for (double f : il) {
        if (convertToFixedPoint) {
            shifted_vals.push_back((T) (f * (1 << FLOAT_PRECISION)));
        } else {
            shifted_vals.push_back((T) f);
        }
    }

    switch (partyNum) {
        case TPCF<T>::PARTY_A:
            thrust::copy(shifted_vals.begin(), shifted_vals.end(), share_buffer_.begin());
            break;
        case TPCF<T>::PARTY_B:
            // nothing
            break;
    }
}

