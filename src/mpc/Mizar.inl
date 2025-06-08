#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>

#include "Mizar.h"
#include "MizarTPC.h"
#include "MizarKernels.cuh"
#include "../gpu/bitwise.cuh"
#include "../gpu/convolution.cuh"
#include "../gpu/conv.cuh"
#include "../gpu/DeviceData.h"
#include "../gpu/functors.cuh"
#include "../gpu/matrix.cuh"
#include "../gpu/gemm.cuh"
#include "../gpu/StridedRange.cuh"
#include "../globals.h"
#include "../util/functors.h"
#include "../util/Profiler.h"

extern Profiler comm_profiler;
extern Profiler func_profiler;

extern nlohmann::json piranha_config;

template<typename T, typename I>
MizarBase<T, I>::MizarBase(DeviceData<T, I> *a, DeviceData<T, I> *b) : 
                share_m_(a), share_r_(b) {}

template<typename T, typename I>
void MizarBase<T, I>::set(DeviceData<T, I> *a, DeviceData<T, I> *b) {
    share_m_ = a;
    share_r_ = b; 
}

template<typename T, typename I>
size_t MizarBase<T, I>::size() const {
    return share_m_->size();
}

template<typename T, typename I>
void MizarBase<T, I>::zero() {
    share_m_->zero();
    share_r_->zero();
};

template<typename T, typename I>
void MizarBase<T, I>::fill(T val) {
    share_m_->fill(partyNum == PARTY_A ? val : 0);
    share_r_->fill(partyNum == PARTY_K ? val : 0);
}

template<typename T, typename I>
void MizarBase<T, I>::setPublic(std::vector<double> &v) {
    std::vector<T> shifted_vals;
    for (double f : v) {
        shifted_vals.push_back((T) (f * (1 << FLOAT_PRECISION)));
    }

    switch (partyNum) {
        case Mizar<T>::PARTY_A:
            thrust::copy(shifted_vals.begin(), shifted_vals.end(), share_m_->begin());
            share_r_->zero();
            break;
        case Mizar<T>::PARTY_B:
            thrust::copy(shifted_vals.begin(), shifted_vals.end(), share_m_->begin());
            share_r_->zero();
            break;
        case Mizar<T>::PARTY_K:
            share_m_->zero();
            share_r_->zero();
            break;
    }
};

template<typename T, typename I>
DeviceData<T, I> *MizarBase<T, I>::getShare(int i) {
    switch (i) {
        case 0:
            return share_m_;
        case 1:
            return share_r_;
        default:
            return nullptr;
    }
}

template<typename T, typename I>
const DeviceData<T, I> *MizarBase<T, I>::getShare(int i) const {
    switch (i) {
        case 0:
            return share_m_;
        case 1:
            return share_r_;
        default:
            return nullptr;
    }
}

template<typename T, typename I>
MizarBase<T, I> &MizarBase<T, I>::operator+=(const T rhs) {
    if (partyNum == PARTY_A or partyNum == PARTY_B) {
        *share_m_ += rhs;
    }
    return *this;
}

template<typename T, typename I>
MizarBase<T, I> &MizarBase<T, I>::operator-=(const T rhs) {
    if (partyNum == PARTY_A or partyNum == PARTY_B) {
        *share_m_ -= rhs;
    }
    return *this;
}

template<typename T, typename I>
MizarBase<T, I> &MizarBase<T, I>::operator*=(const T rhs) {
    *share_m_ *= rhs;
    *share_r_ *= rhs;
    return *this;
}

template<typename T, typename I>
MizarBase<T, I> &MizarBase<T, I>::operator>>=(const T rhs) {
    *share_m_ >>= rhs;
    *share_r_ >>= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
MizarBase<T, I> &MizarBase<T, I>::operator+=(const DeviceData<T, I2> &rhs) {
    if (partyNum == PARTY_A or partyNum == PARTY_B) {
        *share_m_ += rhs;
    }
    return *this;
}

template<typename T, typename I>
template<typename I2>
MizarBase<T, I> &MizarBase<T, I>::operator-=(const DeviceData<T, I2> &rhs) {
    if (partyNum == PARTY_A or partyNum == PARTY_B) {
        *share_m_ -= rhs;
    }
    return *this;
}

template<typename T, typename I>
template<typename I2>
MizarBase<T, I> &MizarBase<T, I>::operator*=(const DeviceData<T, I2> &rhs) {
    *share_m_ *= rhs;
    *share_r_ *= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
MizarBase<T, I> &MizarBase<T, I>::operator^=(const DeviceData<T, I2> &rhs) {
    if (partyNum == PARTY_A or partyNum == PARTY_B) {
        *share_m_ ^= rhs;
    }
    return *this;
}

template<typename T, typename I>
template<typename I2>
MizarBase<T, I> &MizarBase<T, I>::operator&=(const DeviceData<T, I2> &rhs) {
    *share_m_ &= rhs;
    *share_r_ &= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
MizarBase<T, I> &MizarBase<T, I>::operator>>=(const DeviceData<T, I2> &rhs) {
    *share_m_ >>= rhs;
    *share_r_ >>= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
MizarBase<T, I> &MizarBase<T, I>::operator<<=(const DeviceData<T, I2> &rhs) {
    *share_m_ <<= rhs;
    *share_r_ <<= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
MizarBase<T, I> &MizarBase<T, I>::operator+=(const MizarBase<T, I2> &rhs) {
    *share_m_ += *rhs.getShare(0);
    *share_r_ += *rhs.getShare(1);
    return *this;
}

template<typename T, typename I>
template<typename I2>
MizarBase<T, I> &MizarBase<T, I>::operator-=(const MizarBase<T, I2> &rhs) {
    *share_m_ -= *rhs.getShare(0);
    *share_r_ -= *rhs.getShare(1);
    return *this;
}

template<typename T, typename I>
template<typename I2>
MizarBase<T, I> &MizarBase<T, I>::operator*=(const MizarBase<T, I2> &rhs) {
    auto size = rhs.size();
    DeviceData<T> r(size);

    // offline
    #ifdef ENABLE_OFFLINE 
    if (partyNum == PARTY_K) {
        *share_m_ += *share_r_;
        thrust::copy(rhs.getShare(0)->begin(), rhs.getShare(0)->end(), r.begin());
        r += *rhs.getShare(1);
        *share_m_ *= r;
        share_r_->zero();   // TODO: PRG instead. Generate [0].
        *share_m_ -= *share_r_;
        share_m_->transmit(PARTY_A);
        share_m_->join();
        share_r_->transmit(PARTY_B);
        share_r_->join();
        share_m_->zero(); // TODO: PRG instead. Generate [rc] with P0.
        share_r_->zero(); // TODO: PRG instead. Generate [rc] with P1.
    } else {
        r.receive(PARTY_K);
        r.join();
    }
    #endif

    // online
    if (partyNum != PARTY_K) {
        DeviceData<T> temp(size);
        thrust::copy(share_m_->begin(), share_m_->end(), temp.begin());
        temp *= *rhs.getShare(1);
        *share_m_ *= *rhs.getShare(0);
        if (partyNum != PARTY_A) share_m_->zero();
        *share_r_ *= *rhs.getShare(0);
        *share_m_ -= temp;
        *share_m_ -= *share_r_;
        *share_m_ += r;
        share_r_->zero(); // TODO: PRG instead. Generate [rc] with Pk.
        *share_m_ += *share_r_;
        share_m_->transmit(1 - partyNum);
        r.receive(1 - partyNum);
        share_m_->join();
        r.join();
        *share_m_ += r;
    }
    return *this;
}

template<typename T, typename I>
template<typename I2>
MizarBase<T, I> &MizarBase<T, I>::operator^=(const MizarBase<T, I2> &rhs) {
    *share_m_ ^= *rhs.getShare(0);
    *share_r_ ^= *rhs.getShare(1);
    return *this;
}

template<typename T, typename I>
template<typename I2>
MizarBase<T, I> &MizarBase<T, I>::operator&=(const MizarBase<T, I2> &rhs) {
    auto size = rhs.size();
    DeviceData<T> r(size);

    // offline
    #ifdef ENABLE_OFFLINE 
    if (partyNum == PARTY_K) {
        *share_m_ ^= *share_r_;
        thrust::copy(rhs.getShare(0)->begin(), rhs.getShare(0)->end(), r.begin());
        r ^= *rhs.getShare(1);
        *share_m_ &= r;
        share_r_->zero();   // TODO: PRG instead. Generate [0].
        *share_m_ ^= *share_r_;
        share_m_->transmit(PARTY_A);
        share_m_->join();
        share_r_->transmit(PARTY_B);
        share_r_->join();
        share_m_->zero(); // TODO: PRG instead. Generate [rc] with P0.
        share_r_->zero(); // TODO: PRG instead. Generate [rc] with P1.
    } else {
        r.receive(PARTY_K);
        r.join();
    }
    #endif

    // online
    if (partyNum != PARTY_K) {
        DeviceData<T> temp(size);
        thrust::copy(share_m_->begin(), share_m_->end(), temp.begin());
        temp &= *rhs.getShare(1);
        *share_m_ &= *rhs.getShare(0);
        if (partyNum != PARTY_A) share_m_->zero();
        *share_r_ &= *rhs.getShare(0);
        *share_m_ ^= temp;
        *share_m_ ^= *share_r_;
        *share_m_ ^= r;
        share_r_->zero(); // TODO: PRG instead. Generate [rc] with Pk.
        *share_m_ ^= *share_r_;
        share_m_->transmit(1 - partyNum);
        r.receive(1 - partyNum);
        share_m_->join();
        r.join();
        *share_m_ ^= r;
    }
    return *this;
}

template<typename T, typename I>
int MizarBase<T, I>::nextParty(int party) {
	switch(party) {
        case PARTY_A:
            return PARTY_B;
        case PARTY_B:
            return PARTY_K;
        default: // PARTY_K 
            return PARTY_A;
    }	
}

template<typename T, typename I>
int MizarBase<T, I>::prevParty(int party) {
	switch(party) {
        case PARTY_A:
            return PARTY_K;
        case PARTY_B:
            return PARTY_A;
        default: // PARTY_K
            return PARTY_B;
	}	
}

template<typename T, typename I>
int MizarBase<T, I>::numShares() {
    return 2;
}

template<typename T, typename I>
Mizar<T, I>::Mizar(DeviceData<T, I> *a, DeviceData<T, I> *b) : MizarBase<T, I>(a, b) {}

template<typename T>
Mizar<T, BufferIterator<T> >::Mizar(DeviceData<T> *a, DeviceData<T> *b) :
    MizarBase<T, BufferIterator<T> >(a, b) {}
template<typename T>
Mizar<T, BufferIterator<T> >::Mizar(size_t n) :
    share_m_buffer_(n),
    share_r_buffer_(n),
    MizarBase<T, BufferIterator<T> >(&share_m_buffer_, &share_r_buffer_) {}

template<typename T>
Mizar<T, BufferIterator<T> >::Mizar(std::initializer_list<double> il, bool convertToFixedPoint) :
    share_m_buffer_(il.size()),
    share_r_buffer_(il.size()),
    MizarBase<T, BufferIterator<T> >(&share_m_buffer_, &share_r_buffer_) {

    std::vector<T> shifted_vals;
    for (double f : il) {
        if (convertToFixedPoint) {
            shifted_vals.push_back((T) (f * (1 << FLOAT_PRECISION)));
        } else {
            shifted_vals.push_back((T) f);
        }
    }

    switch (partyNum) {
        case Mizar<T>::PARTY_A:
            thrust::copy(shifted_vals.begin(), shifted_vals.end(), share_m_buffer_.begin());
            share_r_buffer_.zero();
            break;
        case Mizar<T>::PARTY_B:
            thrust::copy(shifted_vals.begin(), shifted_vals.end(), share_m_buffer_.begin());
            share_r_buffer_.zero();
            break;
        case Mizar<T>::PARTY_K:
            share_m_buffer_.zero();
            share_r_buffer_.zero();
            break;
    }
}

template<typename T>
void Mizar<T, BufferIterator<T> >::resize(size_t n) {
    share_m_buffer_.resize(n);
    share_r_buffer_.resize(n); 
}


template<typename T, typename I>
void dividePublic(Mizar<T, I> &a, T denominator) {

    // Mizar<T> r(a.size()), rPrime(a.size());
    // PrecomputeObject.getDividedShares<T, Mizar<T> >(r, rPrime, denominator, a.size()); 
    // a -= rPrime;
    
    *a.getShare(0) /= denominator;
    *a.getShare(1) /= denominator;
}

template<typename T, typename I, typename I2>
void dividePublic(Mizar<T, I> &a, DeviceData<T, I2> &denominators) {

    assert(denominators.size() == a.size() && "Mizar dividePublic powers size mismatch");

    // Mizar<T> r(a.size()), rPrime(a.size());
    // PrecomputeObject.getDividedShares<T, I2, Mizar<T> >(r, rPrime, denominators, a.size()); 

    *a.getShare(0) /= denominators;
    *a.getShare(1) /= denominators;
}


template<typename T, typename I, typename I2>
void reconstruct(Mizar<T, I> &in, DeviceData<T, I2> &out) {
    size_t size = in.size();

    if (partyNum == Mizar<T>::PARTY_K) {
        in.getShare(1)->transmit(Mizar<T>::PARTY_A);
        in.getShare(1)->join();
        in.getShare(0)->transmit(Mizar<T>::PARTY_B);
        in.getShare(0)->join();
        out.receive(Mizar<T>::PARTY_A);
        out.join();
        out -= *in.getShare(0);
        out -= *in.getShare(1);
    } else if (partyNum == Mizar<T>::PARTY_A) {
        in.getShare(0)->transmit(Mizar<T>::PARTY_K);
        in.getShare(0)->join();
        out.receive(Mizar<T>::PARTY_K);
        out.join();
        out += *in.getShare(1);
        out *= static_cast<T>(-1);
        out += *in.getShare(0);
    } else {
        out.receive(Mizar<T>::PARTY_K);
        out.join();
        out += *in.getShare(1);
        out *= static_cast<T>(-1);
        out += *in.getShare(0);
    }
}

template<typename T>
void localMatMul(const Mizar<T> &A, const Mizar<T> &B, Mizar<T> &C,
        int M, int N, int K,
        bool transpose_a, bool transpose_b, bool transpose_c) 
{   
    int a_rows = transpose_a ? K : M; int a_cols = transpose_a ? M : K;
    int b_rows = transpose_b ? N : K; int b_cols = transpose_b ? K : N;

    DeviceData<T> R(C.size()); 

    // offline
    #ifdef ENABLE_OFFLINE
    if (partyNum == Mizar<T>::PARTY_K) {
        DeviceData<T> X(A.size()), Y(B.size());
        thrust::copy(A.getShare(0)->begin(), A.getShare(0)->end(), X.begin());
        thrust::copy(B.getShare(0)->begin(), B.getShare(0)->end(), Y.begin());
        X += *A.getShare(1);
        Y += *B.getShare(1);
        {
            auto z_ptr = C.getShare(0);
            auto zero_shr_ptr = C.getShare(1);
            gpu::gemm(M, N, K, 
                &X, transpose_a, &Y, transpose_b, 
                z_ptr, transpose_c);
            zero_shr_ptr->zero(); // TODO: PRG instead. Generate [0].
            *z_ptr -= *zero_shr_ptr;
            z_ptr->transmit(Mizar<T>::PARTY_A); 
            z_ptr->join();
            zero_shr_ptr->transmit(Mizar<T>::PARTY_B);
            zero_shr_ptr->join();
        }
        C.getShare(0)->fill(0); // TODO: PRG instead. Generate [rc] with P0.
        C.getShare(1)->fill(0); // TODO: PRG instead. Generate [rc] with P1.
    } else {
        R.receive(Mizar<T>::PARTY_K);
        R.join();
    }
    #endif

    if (partyNum != Mizar<T>::PARTY_K) {
        {
            DeviceData<T> x1y0(C.size()); 
            auto x0y0_ptr = C.getShare(0);
            auto x0y1_ptr = C.getShare(1);
            if (partyNum == Mizar<T>::PARTY_A) {
                gpu::gemm(M, N, K, A.getShare(0), transpose_a, B.getShare(0), transpose_b, x0y0_ptr, transpose_c); // x0y0
            }
            gpu::gemm(M, N, K, A.getShare(0), transpose_a, B.getShare(1), transpose_b, x0y1_ptr, transpose_c);     // x0y1
            gpu::gemm(M, N, K, A.getShare(1), transpose_a, B.getShare(0), transpose_b, &x1y0, transpose_c);        // x1y0   
            *x0y0_ptr -= *x0y1_ptr;
            *x0y0_ptr -= x1y0;
        }
        *C.getShare(0) += R;
        C.getShare(1)->zero(); // TODO: PRG instead. Generate [rc] with Pk.
        *C.getShare(0) += *C.getShare(1);
        C.getShare(0)->transmit(1 - partyNum);
        R.receive(1 - partyNum);
        C.getShare(0)->join();
        R.join();
        *C.getShare(0) += R;
    }
}

template<typename T>
void matmul(const Mizar<T> &a, const Mizar<T> &b, Mizar<T> &c,
        int M, int N, int K,
        bool transpose_a, bool transpose_b, bool transpose_c, T truncation)
{
    localMatMul(a, b, c, M, N, K, transpose_a, transpose_b, transpose_c);

    dividePublic(c, (T)1 << truncation);
}

template<typename T>
void localFprop(const Mizar<T> &A, const Mizar<T> &B, Mizar<T> &C,
        int batchSize, int imageHeight, int imageWidth, int Din,
        int Dout, int filterHeight, int filterWidth,
        int paddingHeight, int paddingWidth,
        int stride, int dilation) {
    DeviceData<T> R(C.size()); 

    // offline
    #ifdef ENABLE_OFFLINE
    if (partyNum == Mizar<T>::PARTY_K) {
        DeviceData<T> X(A.size()), Y(B.size());
        thrust::copy(A.getShare(0)->begin(), A.getShare(0)->end(), X.begin());
        thrust::copy(B.getShare(0)->begin(), B.getShare(0)->end(), Y.begin());
        X += *A.getShare(1);
        Y += *B.getShare(1);
        {
            auto z_ptr = C.getShare(0);
            auto zero_shr_ptr = C.getShare(1);
            gpu::conv_fprop(&X, &Y, z_ptr, 
                batchSize, imageHeight, imageWidth, Din,
                Dout, filterHeight, filterWidth,
                paddingHeight, paddingWidth,
                stride, dilation);
            zero_shr_ptr->zero(); // TODO: PRG instead. Generate [0].
            *z_ptr -= *zero_shr_ptr;
            z_ptr->transmit(Mizar<T>::PARTY_A); 
            z_ptr->join();
            zero_shr_ptr->transmit(Mizar<T>::PARTY_B);
            zero_shr_ptr->join();
        }
        C.getShare(0)->fill(0); // TODO: PRG instead. Generate [rc].
        C.getShare(1)->fill(0); // TODO: PRG instead. Generate [rc].
    } else {
        R.receive(Mizar<T>::PARTY_K);
        R.join();
    }
    #endif

    if (partyNum != Mizar<T>::PARTY_K) {
        {
            DeviceData<T> x1y0(C.size()); 
            auto x0y0_ptr = C.getShare(0);
            auto x0y1_ptr = C.getShare(1);
            if (partyNum == Mizar<T>::PARTY_A) {
                gpu::conv_fprop(A.getShare(0), B.getShare(0), x0y0_ptr, 
                    batchSize, imageHeight, imageWidth, Din,
                    Dout, filterHeight, filterWidth,
                    paddingHeight, paddingWidth,
                    stride, dilation);
            }
            gpu::conv_fprop(A.getShare(0), B.getShare(1), x0y1_ptr, 
                    batchSize, imageHeight, imageWidth, Din,
                    Dout, filterHeight, filterWidth,
                    paddingHeight, paddingWidth,
                    stride, dilation);
            gpu::conv_fprop(A.getShare(1), B.getShare(0), &x1y0, 
                    batchSize, imageHeight, imageWidth, Din,
                    Dout, filterHeight, filterWidth,
                    paddingHeight, paddingWidth,
                    stride, dilation);
            *x0y0_ptr -= *x0y1_ptr;
            *x0y0_ptr -= x1y0;
        }
        *C.getShare(0) += R;
        C.getShare(1)->zero(); // TODO: PRG instead. Generate [rc] with Pk.
        *C.getShare(0) += *C.getShare(1);
        C.getShare(0)->transmit(1 - partyNum);
        R.receive(1 - partyNum);
        C.getShare(0)->join();
        R.join();
        *C.getShare(0) += R;
    }
    cudaDeviceSynchronize();
}

template<typename T>
void localDgrad(const Mizar<T> &A, const Mizar<T> &B, Mizar<T> &C,
        int batchSize, int outputHeight, int outputWidth, int Dout,
        int filterHeight, int filterWidth, int Din,
        int paddingHeight, int paddingWidth, int stride, int dilation,
        int imageHeight, int imageWidth) {
    DeviceData<T> R(C.size()); 

    // offline
    #ifdef ENABLE_OFFLINE
    if (partyNum == Mizar<T>::PARTY_K) {
        DeviceData<T> X(A.size()), Y(B.size());
        thrust::copy(A.getShare(0)->begin(), A.getShare(0)->end(), X.begin());
        thrust::copy(B.getShare(0)->begin(), B.getShare(0)->end(), Y.begin());
        X += *A.getShare(1);
        Y += *B.getShare(1);
        {
            auto z_ptr = C.getShare(0);
            auto zero_shr_ptr = C.getShare(1);
            gpu::conv_dgrad(&X, &Y, z_ptr, 
                batchSize, outputHeight, outputWidth, Dout,
                filterHeight, filterWidth, Din,
                paddingHeight, paddingWidth, stride, dilation,
                imageHeight, imageWidth);
            zero_shr_ptr->zero(); // TODO: PRG instead. Generate [0].
            *z_ptr -= *zero_shr_ptr;
            z_ptr->transmit(Mizar<T>::PARTY_A); 
            z_ptr->join();
            zero_shr_ptr->transmit(Mizar<T>::PARTY_B);
            zero_shr_ptr->join();
        }
        C.getShare(0)->fill(0); // TODO: PRG instead. Generate [rc].
        C.getShare(1)->fill(0); // TODO: PRG instead. Generate [rc].
    } else {
        R.receive(Mizar<T>::PARTY_K);
        R.join();
    }
    #endif

    if (partyNum != Mizar<T>::PARTY_K) {
        {
            DeviceData<T> x1y0(C.size()); 
            auto x0y0_ptr = C.getShare(0);
            auto x0y1_ptr = C.getShare(1);
            if (partyNum == Mizar<T>::PARTY_A) {
                gpu::conv_dgrad(A.getShare(0), B.getShare(0), x0y0_ptr, 
                    batchSize, outputHeight, outputWidth, Dout,
                    filterHeight, filterWidth, Din,
                    paddingHeight, paddingWidth, stride, dilation,
                    imageHeight, imageWidth);
            }
            gpu::conv_dgrad(A.getShare(0), B.getShare(1), x0y1_ptr, 
                    batchSize, outputHeight, outputWidth, Dout,
                    filterHeight, filterWidth, Din,
                    paddingHeight, paddingWidth, stride, dilation,
                    imageHeight, imageWidth);
            gpu::conv_dgrad(A.getShare(1), B.getShare(0), &x1y0,
                    batchSize, outputHeight, outputWidth, Dout,
                    filterHeight, filterWidth, Din,
                    paddingHeight, paddingWidth, stride, dilation,
                    imageHeight, imageWidth);
            *x0y0_ptr -= *x0y1_ptr;
            *x0y0_ptr -= x1y0;
        }
        *C.getShare(0) += R;
        C.getShare(1)->zero(); // TODO: PRG instead. Generate [rc] with Pk.
        *C.getShare(0) += *C.getShare(1);
        C.getShare(0)->transmit(1 - partyNum);
        R.receive(1 - partyNum);
        C.getShare(0)->join();
        R.join();
        *C.getShare(0) += R;
    }
    cudaDeviceSynchronize();
}

template<typename T>
void localWgrad(const Mizar<T> &A, const Mizar<T> &B, Mizar<T> &C,
        int batchSize, int outputHeight, int outputWidth, int Dout,
        int imageHeight, int imageWidth, int Din,
        int filterHeight, int filterWidth,
        int paddingHeight, int paddingWidth, int stride, int dilation) {
    DeviceData<T> R(C.size());

    // offline
    #ifdef ENABLE_OFFLINE
    if (partyNum == Mizar<T>::PARTY_K) {
        DeviceData<T> X(A.size()), Y(B.size());
        thrust::copy(A.getShare(0)->begin(), A.getShare(0)->end(), X.begin());
        thrust::copy(B.getShare(0)->begin(), B.getShare(0)->end(), Y.begin());
        X += *A.getShare(1);
        Y += *B.getShare(1);
        {
            auto z_ptr = C.getShare(0);
            auto zero_shr_ptr = C.getShare(1);
            gpu::conv_wgrad(&X, &Y, z_ptr, 
                batchSize, outputHeight, outputWidth, Dout,
                imageHeight, imageWidth, Din,
                filterHeight, filterWidth,
                paddingHeight, paddingWidth,
                stride, dilation);
            zero_shr_ptr->zero(); // TODO: PRG instead. Generate [0].
            *z_ptr -= *zero_shr_ptr;
            z_ptr->transmit(Mizar<T>::PARTY_A); 
            z_ptr->join();
            zero_shr_ptr->transmit(Mizar<T>::PARTY_B);
            zero_shr_ptr->join();
        }
    } else {
        R.receive(Mizar<T>::PARTY_K);
        R.join();
    }
    #endif

    if (partyNum != Mizar<T>::PARTY_K) {
        {
            DeviceData<T> x1y0(C.size()); 
            auto x0y0_ptr = C.getShare(0);
            auto x0y1_ptr = C.getShare(1);
            if (partyNum == Mizar<T>::PARTY_A) {
                gpu::conv_wgrad(A.getShare(0), B.getShare(0), x0y0_ptr, 
                    batchSize, outputHeight, outputWidth, Dout,
                    imageHeight, imageWidth, Din,
                    filterHeight, filterWidth,
                    paddingHeight, paddingWidth,
                    stride, dilation);
            }
            gpu::conv_wgrad(A.getShare(0), B.getShare(1), x0y1_ptr, 
                    batchSize, outputHeight, outputWidth, Dout,
                    imageHeight, imageWidth, Din,
                    filterHeight, filterWidth,
                    paddingHeight, paddingWidth,
                    stride, dilation);
            gpu::conv_wgrad(A.getShare(1), B.getShare(0), &x1y0,
                    batchSize, outputHeight, outputWidth, Dout,
                    imageHeight, imageWidth, Din,
                    filterHeight, filterWidth,
                    paddingHeight, paddingWidth,
                    stride, dilation);
            *x0y0_ptr -= *x0y1_ptr;
            *x0y0_ptr -= x1y0;
        }
        *C.getShare(0) += R;
        C.getShare(1)->zero(); // TODO: PRG instead. Generate [rc] with Pk.
        *C.getShare(0) += *C.getShare(1);
        C.getShare(0)->transmit(1 - partyNum);
        R.receive(1 - partyNum);
        C.getShare(0)->join();
        R.join();
        *C.getShare(0) += R;
    }
    cudaDeviceSynchronize();
}

template<typename T>
void convolution(const Mizar<T> &A, const Mizar<T> &B, Mizar<T> &C,
        cutlass::conv::Operator op,
        int batchSize, int imageHeight, int imageWidth, int filterSize,
        int Din, int Dout, int stride, int padding, int truncation) {

    int outputHeight = (imageHeight + 2 * padding - filterSize) / stride + 1; 
    int outputWidth = (imageWidth + 2 * padding - filterSize) / stride + 1; 
    C.zero();
    // DeviceData<T> localResult(C.size());

    switch (op) {
        case cutlass::conv::Operator::kFprop:
            localFprop(A, B, C,
                    batchSize, imageHeight, imageWidth, Din,
                    Dout, filterSize, filterSize,
                    padding, padding,
                    stride, (T)1);
            break;
        case cutlass::conv::Operator::kDgrad:
            localDgrad(A, B, C,
                    batchSize, outputHeight, outputWidth, Dout,
                    filterSize, filterSize, Din,
                    padding, padding, stride, (T)1,
                    imageHeight, imageWidth);
            break;
        case cutlass::conv::Operator::kWgrad:
            localWgrad(A, B, C,
                    batchSize, outputHeight, outputWidth, Dout,
                    imageHeight, imageWidth, Din,
                    filterSize, filterSize,
                    padding, padding, stride, (T)1);
            break;
    }

    // *C.getShare(0) += localResult;
    dividePublic(C, (T)1 << truncation);
}

template<typename T, typename I, typename I2, typename I3, typename I4>
void selectShare(const Mizar<T, I> &x, const Mizar<T, I2> &y, const Mizar<T, I3> &b, Mizar<T, I4> &z) 
{
    assert(x.size() == y.size() && x.size() == b.size() && x.size() == z.size() && "Mizar selectShare size mismatch");
    thrust::copy(y.getShare(0)->begin(), y.getShare(0)->end(), z.getShare(0)->begin());
    thrust::copy(y.getShare(1)->begin(), y.getShare(1)->end(), z.getShare(1)->begin());
    z -= x;
    z *= b;
    z += x;
}

template<typename T, typename I, typename I2>
void dReLU(const Mizar<T, I> &input, Mizar<T, I2> &result) 
{
    #if     defined(AEGIS)
    #if     defined(AEGISOPT)
    dReLUFromAegisOpt(input, result);
    #else
    dReLUFromAegis(input, result);
    #endif
    #elif   defined(MIZAR)
    dReLUFromMizarOpt(input, result);
    #else
    #warning "No Mizar implementation selected."    \ 
             "Please define AEGIS or MIZAR explicitly."
    dReLUFromMizarOpt(input, result);
    #endif
}

template<typename T, typename I, typename I2>
void dReLUFromAegis(const Mizar<T, I> &input, Mizar<T, I2> &result)
{
    size_t size = input.size();
    size_t bvsize = size * sizeof(T) * 8;
    using U = uint8_t;
    
    if (partyNum != Mizar<T>::PARTY_K)  // P1 or P2
    {
        MizarTPC<T> rp(size) , rz(size);   
        rp.zero(), rz.zero();           // PRF, private

        // Offline
        //- // std::cout << "offline;" << std::endl;
        T Delta = 0;                    // PRF, 0 or 1
        DeviceData<T> Gamma(size); MizarTPC<U> rhat_bv(bvsize);      
        Gamma.zero(), rhat_bv.zero();
        {
            #ifdef ENABLE_OFFLINE
            // Gamma = \Delta + rp - 2 * \Delta * rp + rz
            MizarTPC<T> ss_Gamma(size); ss_Gamma.zero();
            ss_Gamma += rp;
            ss_Gamma *= TPCFPRIME - 2;
            ss_Gamma += 1;  // 1 - 2 * rp
            ss_Gamma *= Delta;
            ss_Gamma += rp;
            ss_Gamma += rz;
            ss_Gamma += ss_Gamma;

            ss_Gamma.getShare(0)->transmit(1 - partyNum);
            Gamma.receive(1 - partyNum);
            ss_Gamma.getShare(0)->join(); Gamma.join();
            Gamma += *ss_Gamma.getShare(0);

            DeviceData<U> somewhat_awkward(bvsize);
            shareFromKing(somewhat_awkward, rhat_bv);
            #endif
        }

        // Online
        // step 1 & 2.
        DeviceData<U> mhat_bv(bvsize); mhat_bv.zero();
        //- // std::cout << "online; step 1 & 2" << std::endl;
        {
            DeviceData<T> mhat(size); mhat.zero();
            mhat += *input.getShare(0); 
            mhat &= (1ull << (sizeof(T) * 8 - 1)) - 1;         // b0 b1 b2 ... -> 0 b1 b2 ...
            mhat ^= 1ull << (sizeof(T) * 8 - 1);               // 0 b1 b2 ... -> 1 b1 b2 ...
            gpu::bitexpand(&mhat, &mhat_bv);
        }

        // step 3.
        //- // std::cout << "online; step 3" << std::endl;
        MizarTPC<U> m(bvsize); m.zero();
        {
            m += rhat_bv; m *= TPCFPRIME - 2; m += 1; m *= mhat_bv; m += rhat_bv;
        }
        
        // step 4.
        //- // std::cout << "online; step 4" << std::endl;
        DeviceData<U> w(bvsize), wp(bvsize);
        w.fill(1), wp.fill(1);

        // step 5.
        // --- mp
        //- // std::cout << "online; step 5" << std::endl;
        MizarTPC<U> mp(bvsize); mp.zero();
        {
            //- // std::cout << "online; step 5; before inclusive scan" << std::endl;
            MizarTPC<U> prefix_sum_m(bvsize); prefix_sum_m.zero();
            auto range_iter = thrust::make_transform_iterator(thrust::make_counting_iterator(0ull), select_range_functor<U>(sizeof(T) * 8));
            thrust::inclusive_scan_by_key(range_iter, range_iter + bvsize, m.getShare(0)->begin(), prefix_sum_m.getShare(0)->begin());      // t = 1 or t = 0?

            //- // std::cout << "online; step 5; after inclusive scan" << std::endl;
            mp += m; mp *= TPCFPRIME - 2; mp += 1; mp += prefix_sum_m;
        }
        
        // --- u
        //- // std::cout << "online; step 5, u;" << std::endl;
        MizarTPC<U> u(bvsize); u.zero();        
        {
            DeviceData<U> msbm(bvsize); DeviceData<T> temp(size); DeviceData<U> temp2(size);
            msbm.zero(); temp.zero(); temp2.zero();
            temp += *input.getShare(0);
            temp &= 1ull << (sizeof(T) * 8 - 1);
            logic_rshift(temp, temp, static_cast<T>(sizeof(T) * 8 - 1));
            thrust::copy(temp.begin(), temp.end(), temp2.begin());
            gpu::repeat(temp2, msbm);
            msbm ^= Delta; msbm ^= mhat_bv;
            u += mp; u *= w; u += msbm;
        }

        // --- up
        //- // std::cout << "online; step 5, up;" << std::endl;
        MizarTPC<U> up(bvsize); up.zero();
        {
            up += mp; up *= w; up += 1; up *= wp;
        }
        
        // step 6.
        //- // std::cout << "online; step 6" << std::endl;
        MizarTPC<U> uhat(bvsize); uhat.zero();  
        MizarTPC<U> uphat(bvsize); uphat.zero();  
        // Permute...
        uhat += u; uphat += up;

        // step 7.
        //- // std::cout << "online; step 7" << std::endl;
        uhat.getShare(0)->transmit(Mizar<T>::PARTY_K);
        uhat.getShare(0)->join();
        uphat.getShare(0)->transmit(Mizar<T>::PARTY_K);
        uphat.getShare(0)->join();
        
        // finally.
        //- // std::cout << "online; finally" << std::endl;
        Delta = (Delta * (TPCFPRIME - 2) + 1) % TPCFPRIME;
        DeviceData<T> mp_(size); mp_.zero();
        mp_.receive(Mizar<T>::PARTY_K);
        mp_.join();
        mp_ *= Delta;
        mp_ += Gamma;
        thrust::copy(mp_.begin(), mp_.end(), result.getShare(0)->begin());
    } else {        // P0
        // Offline
        //- // std::cout << "dReLU from dealer;" << std::endl;
        //- // std::cout << "offline;" << std::endl;
        DeviceData<T> rp(size); Mizar<T> rz(size); DeviceData<T> sign_neg_r(size);
        rp.zero(); rz.zero(); sign_neg_r.zero();
        {
            #ifdef ENABLE_OFFLINE
            DeviceData<T> rhat(size); DeviceData<U> rhat_bv(bvsize);
            rhat.zero(), rhat_bv.zero();

            // rhat = 2^{l-1} + sign(-rx) * 2^{l-1} + rx
            rhat += *input.getShare(0); rhat += *input.getShare(1);
            sign_neg_r += rhat;
            sign_neg_r &= 1ull << (sizeof(T) * 8 - 1);            // b0 b1 b2 ... -> b0 0 0 ...
            sign_neg_r ^= 1ull << (sizeof(T) * 8 - 1);            // b0 0 0 ... -> not(b0) 0 0 ...
            rhat += 1ull << (sizeof(T) * 8 - 1); rhat += sign_neg_r;    // assert that the msb of rhat is 0.
            gpu::bitexpand(&rhat, &rhat_bv);
            MizarTPC<U> somewhat_awkward(bvsize);
            shareFromKing(rhat_bv, somewhat_awkward);
            #endif
        }

        // Online
        //- // std::cout << "online; gather" << std::endl;
        DeviceData<U> u0(bvsize), up0(bvsize), u1(bvsize), up1(bvsize);
        {
            u0.receive(Mizar<T>::PARTY_A), u1.receive(Mizar<T>::PARTY_B);
            u0.join(), u1.join(); u0 += u1; 
            up0.receive(Mizar<T>::PARTY_A), up1.receive(Mizar<T>::PARTY_B);
            up0.join(), up1.join(); up0 += up1;
        }
        
        //- // std::cout << "online; compute flag" << std::endl;
        DeviceData<T> mp0_(size), mp1_(size), mp_(size); 
        {
            mp0_.zero(), mp1_.zero();
            logic_rshift(sign_neg_r, sign_neg_r, static_cast<T>(sizeof(T) * 8 - 1));
            mp0_ += sign_neg_r; mp0_ -= rp;
            mp1_ += sign_neg_r; mp1_ ^= 1ull; mp1_ -= rp;
        }

        auto zip_mp = thrust::make_zip_iterator(mp0_.begin(), mp1_.begin());
        DeviceData<U> flags(bvsize);
        auto check_u_binary_op = [] __host__ __device__ (U ui, U upi) -> U {
            if (ui == 0 and upi != 0) return 0;
            else return 1;
        };
        thrust::transform(u0.begin(), u0.end(), up0.begin(), flags.begin(), check_u_binary_op);
        DeviceData<U> reduced_flags(size);
        thrust::equal_to<U> binary_pred;
        thrust::multiplies<U> mult_op;
        auto range_iter = thrust::make_transform_iterator(thrust::make_counting_iterator(0), select_range_functor<U>(sizeof(T) * 8));
        thrust::reduce_by_key(
            range_iter, range_iter + bvsize, flags.begin(), flags.begin() /* only for placeholder */, reduced_flags.begin(), binary_pred, mult_op);
        reduced_flags ^= static_cast<U>(1);

        auto select_op = [] __host__  __device__ (thrust::tuple<U, U> x, U y) -> U {
            if (y == 0) return thrust::get<0>(x);
            else return thrust::get<1>(x);
        };
        thrust::transform(zip_mp, zip_mp + size, reduced_flags.begin(), mp_.begin(), select_op);

        //- // std::cout << "online; broadcast" << std::endl;
        mp_.transmit(Mizar<T>::PARTY_A);
        mp_.join();
        mp_.transmit(Mizar<T>::PARTY_B);
        mp_.join();

        thrust::copy(rz.getShare(0)->begin(), rz.getShare(0)->end(), result.getShare(0)->begin());
        thrust::copy(rz.getShare(1)->begin(), rz.getShare(1)->end(), result.getShare(1)->begin());
    }
}

template<typename T, typename I, typename I2>
void dReLUFromAegisOpt(const Mizar<T, I> &input, Mizar<T, I2> &result)
{
    size_t size = input.size();
    size_t bvsize = size * sizeof(T) * 8;
    using U = uint8_t;
    
    if (partyNum != Mizar<T>::PARTY_K)  // P1 or P2
    {
        MizarTPC<T> rp(size) , rz(size);   
        rp.zero(), rz.zero();           // PRF, private

        // Offline
        //- // std::cout << "offline;" << std::endl;
        T Delta = 0;                    // PRF, 0 or 1
        DeviceData<T> Gamma(size); MizarTPC<U> rhat_bv(bvsize);      
        Gamma.zero(), rhat_bv.zero();
        {
            #ifdef ENABLE_OFFLINE
            // Gamma = \Delta + rp - 2 * \Delta * rp + rz
            MizarTPC<T> ss_Gamma(size); ss_Gamma.zero();
            ss_Gamma += rp;
            ss_Gamma *= TPCFPRIME - 2;
            ss_Gamma += 1;  // 1 - 2 * rp
            ss_Gamma *= Delta;
            ss_Gamma += rp;
            ss_Gamma += rz;
            ss_Gamma += ss_Gamma;

            ss_Gamma.getShare(0)->transmit(1 - partyNum);
            Gamma.receive(1 - partyNum);
            ss_Gamma.getShare(0)->join(); Gamma.join();
            Gamma += *ss_Gamma.getShare(0);

            DeviceData<U> somewhat_awkward(bvsize);
            shareFromKing(somewhat_awkward, rhat_bv);
            #endif
        }

        // Online
        // step 1 & 2.
        DeviceData<U> mhat_bv(bvsize); mhat_bv.zero();
        //- // std::cout << "online; step 1 & 2" << std::endl;
        {
            DeviceData<T> mhat(size); mhat.zero();
            mhat += *input.getShare(0); 
            mhat &= (1ull << (sizeof(T) * 8 - 1)) - 1;         // b0 b1 b2 ... -> 0 b1 b2 ...
            mhat ^= 1ull << (sizeof(T) * 8 - 1);               // 0 b1 b2 ... -> 1 b1 b2 ...
            gpu::bitexpand(&mhat, &mhat_bv);
        }

        // step 3.
        //- // std::cout << "online; step 3" << std::endl;
        MizarTPC<U> m(bvsize); m.zero();
        {
            m += rhat_bv; m *= TPCFPRIME - 2; m += 1; m *= mhat_bv; m += rhat_bv;
        }
        
        // step 4.
        //- // std::cout << "online; step 4" << std::endl;
        DeviceData<U> w(bvsize), wp(bvsize);
        w.fill(1), wp.fill(1);

        // step 5.
        // --- mp
        //- // std::cout << "online; step 5" << std::endl;
        MizarTPC<U> mp(bvsize); mp.zero();
        {
            //- // std::cout << "online; step 5; before inclusive scan" << std::endl;
            MizarTPC<U> prefix_sum_m(bvsize); prefix_sum_m.zero();
            auto range_iter = thrust::make_transform_iterator(thrust::make_counting_iterator(0ull), select_range_functor<U>(sizeof(T) * 8));
            thrust::inclusive_scan_by_key(range_iter, range_iter + bvsize, m.getShare(0)->begin(), prefix_sum_m.getShare(0)->begin());      // t = 1 or t = 0?

            //- // std::cout << "online; step 5; after inclusive scan" << std::endl;
            mp += m; mp *= TPCFPRIME - 2; mp += 1; mp += prefix_sum_m;
        }
        
        // --- u
        //- // std::cout << "online; step 5, u;" << std::endl;
        MizarTPC<U> u(bvsize); u.zero(); 
        {
            DeviceData<U> msbm(bvsize); DeviceData<T> temp(size); DeviceData<U> temp2(size); DeviceData<U> msbmp1(bvsize);
            msbm.zero(); temp.zero(); temp2.zero(); msbmp1.zero();
            temp += *input.getShare(0);
            temp &= 1ull << (sizeof(T) * 8 - 1);
            logic_rshift(temp, temp, static_cast<T>(sizeof(T) * 8 - 1));
            thrust::copy(temp.begin(), temp.end(), temp2.begin());
            gpu::repeat(temp2, msbm);
            msbm ^= Delta; msbm ^= mhat_bv;
            msbmp1 ^= msbm; msbmp1 ^= 1;
            u += mp; u *= w; u *= msbmp1;
            msbm *= w; u += msbm;
        }
        
        // step 6.
        //- // std::cout << "online; step 6" << std::endl;
        MizarTPC<U> uhat(bvsize); uhat.zero();  
        // Permute...
        uhat += u; 

        // step 7.
        //- // std::cout << "online; step 7" << std::endl;
        uhat.getShare(0)->transmit(Mizar<T>::PARTY_K);
        uhat.getShare(0)->join();
        
        // finally.
        //- // std::cout << "online; finally" << std::endl;
        Delta = (Delta * (TPCFPRIME - 2) + 1) % TPCFPRIME;
        DeviceData<T> mp_(size); mp_.zero();
        mp_.receive(Mizar<T>::PARTY_K);
        mp_.join();
        mp_ *= Delta;
        mp_ += Gamma;
        thrust::copy(mp_.begin(), mp_.end(), result.getShare(0)->begin());
    } else {        // P0
        // Offline
        //- // std::cout << "dReLU from dealer;" << std::endl;
        //- // std::cout << "offline;" << std::endl;
        DeviceData<T> rp(size); Mizar<T> rz(size); DeviceData<T> sign_neg_r(size);
        rp.zero(); rz.zero(); sign_neg_r.zero();
        {
            #ifdef ENABLE_OFFLINE
            DeviceData<T> rhat(size); DeviceData<U> rhat_bv(bvsize);
            rhat.zero(), rhat_bv.zero();

            // rhat = 2^{l-1} + sign(-rx) * 2^{l-1} + rx
            rhat += *input.getShare(0); rhat += *input.getShare(1);
            sign_neg_r += rhat;
            sign_neg_r &= 1ull << (sizeof(T) * 8 - 1);            // b0 b1 b2 ... -> b0 0 0 ...
            sign_neg_r ^= 1ull << (sizeof(T) * 8 - 1);            // b0 0 0 ... -> not(b0) 0 0 ...
            rhat += 1ull << (sizeof(T) * 8 - 1); rhat += sign_neg_r;    // assert that the msb of rhat is 0.
            gpu::bitexpand(&rhat, &rhat_bv);
            MizarTPC<U> somewhat_awkward(bvsize);
            shareFromKing(rhat_bv, somewhat_awkward);
            #endif
        }

        // Online
        //- // std::cout << "online; gather" << std::endl;
        DeviceData<U> u0(bvsize), up0(bvsize), u1(bvsize), up1(bvsize);
        {
            u0.receive(Mizar<T>::PARTY_A), u1.receive(Mizar<T>::PARTY_B);
            u0.join(), u1.join(); u0 += u1; 
        }
        
        //- // std::cout << "online; compute flag" << std::endl;
        DeviceData<T> mp0_(size), mp1_(size), mp_(size); 
        {
            mp0_.zero(), mp1_.zero();
            logic_rshift(sign_neg_r, sign_neg_r, static_cast<T>(sizeof(T) * 8 - 1));
            mp0_ += sign_neg_r;
            mp1_ += sign_neg_r; mp1_ ^= 1ull;
        }

        auto zip_mp = thrust::make_zip_iterator(mp0_.begin(), mp1_.begin());
        DeviceData<U> flags(bvsize);
        auto check_u_binary_op = [] __host__ __device__ (U ui) -> U {
            if (ui == 0) return 0;
            else return 1;
        };
        thrust::transform(u0.begin(), u0.end(), flags.begin(), check_u_binary_op);
        DeviceData<U> reduced_flags(size);
        thrust::equal_to<U> binary_pred;
        thrust::multiplies<U> mult_op;
        auto range_iter = thrust::make_transform_iterator(thrust::make_counting_iterator(0), select_range_functor<U>(sizeof(T) * 8));
        thrust::reduce_by_key(
            range_iter, range_iter + bvsize, flags.begin(), flags.begin() /* only for placeholder */, reduced_flags.begin(), binary_pred, mult_op);
        reduced_flags ^= static_cast<U>(1);

        auto select_op = [] __host__  __device__ (thrust::tuple<U, U> x, U y) -> U {
            if (y == 0) return thrust::get<0>(x);
            else return thrust::get<1>(x);
        };
        thrust::transform(zip_mp, zip_mp + size, reduced_flags.begin(), mp_.begin(), select_op);

        //- // std::cout << "online; broadcast" << std::endl;
        mp_.transmit(Mizar<T>::PARTY_A);
        mp_.join();
        mp_.transmit(Mizar<T>::PARTY_B);
        mp_.join();

        thrust::copy(rz.getShare(0)->begin(), rz.getShare(0)->end(), result.getShare(0)->begin());
        thrust::copy(rz.getShare(1)->begin(), rz.getShare(1)->end(), result.getShare(1)->begin());
    }
}


template<typename T, typename I, typename I2>
void dReLUFromMizar(const Mizar<T, I> &input, Mizar<T, I2> &result)
{
    size_t size = input.size();
    size_t bvsize = size * sizeof(T) * 8;
    using U = uint8_t;  // TODO: use uint8_t with naive mizar will get wrong result.
    
    if (partyNum != Mizar<T>::PARTY_K)  // P1 or P2
    {
        MizarTPC<T> rp(size) , rz(size);   
        rp.zero(), rz.zero();           // PRF, private

        // Offline
        // std::cout << "offline;" << std::endl;
        T Delta = 0;                    // PRF, 0 or 1
        DeviceData<T> Gamma(size); MizarTPC<U> rhat_bv(bvsize);      
        Gamma.zero(), rhat_bv.zero();
        {
            #ifdef ENABLE_OFFLINE
            // Gamma = \Delta + rp - 2 * \Delta * rp + rz
            MizarTPC<T> ss_Gamma(size); ss_Gamma.zero();
            ss_Gamma += rp;
            ss_Gamma *= TPCFPRIME - 2;
            ss_Gamma += 1;  // 1 - 2 * rp
            ss_Gamma *= Delta;
            ss_Gamma += rp;
            ss_Gamma += rz; //

            ss_Gamma.getShare(0)->transmit(1 - partyNum);
            Gamma.receive(1 - partyNum);
            ss_Gamma.getShare(0)->join(); Gamma.join();
            Gamma += *ss_Gamma.getShare(0);

            DeviceData<U> somewhat_awkward(bvsize);
            shareFromKing(somewhat_awkward, rhat_bv);
            #endif
        }

        // Online
        // step 1 & 2.
        DeviceData<U> mhat_bv(bvsize); mhat_bv.zero();
        // std::cout << "online; step 1 & 2" << std::endl;
        {
            DeviceData<T> mhat(size); mhat.zero();
            mhat += *input.getShare(0); 
            mhat &= (1ull << (sizeof(T) * 8 - 1)) - 1;         // b0 b1 b2 ... -> 0 b1 b2 ...
            mhat ^= 1ull << (sizeof(T) * 8 - 1);               // 0 b1 b2 ... -> 1 b1 b2 ...
            gpu::bitexpand(&mhat, &mhat_bv);
        }

        // step 3.
        // std::cout << "online; step 3" << std::endl;
        MizarTPC<U> m(bvsize); m.zero();
        {
            m += rhat_bv; m *= TPCFPRIME - 2; m += 1; m *= mhat_bv; m += rhat_bv;
        }
        
        // step 4.
        // std::cout << "online; step 4" << std::endl;
        DeviceData<U> w(bvsize);
        w.fill(1);

        // step 5.
        // --- mp, a.k.a. t
        // std::cout << "online; step 5" << std::endl;
        MizarTPC<U> mp(bvsize); mp.zero();
        {
            // std::cout << "online; step 5; before inclusive scan" << std::endl;
            MizarTPC<U> prefix_sum_m(bvsize); prefix_sum_m.zero();
            auto range_iter = thrust::make_transform_iterator(thrust::make_counting_iterator(0ull), select_range_functor<U>(sizeof(T) * 8));
            thrust::inclusive_scan_by_key(range_iter, range_iter + bvsize, m.getShare(0)->begin(), prefix_sum_m.getShare(0)->begin());      // t = 1 or t = 0?

            // std::cout << "online; step 5; after inclusive scan" << std::endl;
            mp += m; mp *= TPCFPRIME - 2; mp += 1; mp += prefix_sum_m;
        }
        
        // --- v
        // std::cout << "online; step 5, v;" << std::endl;
        MizarTPC<U> v(bvsize); v.zero();        
        {
            DeviceData<U> msbm(bvsize); DeviceData<T> temp(size); DeviceData<U> temp2(size);
            msbm.zero(); temp.zero(); temp2.zero();
            temp += *input.getShare(0);
            temp &= 1ull << (sizeof(T) * 8 - 1);
            logic_rshift(temp, temp, static_cast<T>(sizeof(T) * 8 - 1));
            thrust::copy(temp.begin(), temp.end(), temp2.begin());
            gpu::repeat(temp2, msbm);
            msbm ^= Delta; msbm ^= mhat_bv;
            v += mp; v += msbm; v *= w;
        }
        
        // step 6.
        // std::cout << "online; step 6" << std::endl;
        MizarTPC<U> vhat(bvsize); vhat.zero();  
        // Permute...
        vhat += v; 

        // step 7.
        // std::cout << "online; step 7" << std::endl;
        vhat.getShare(0)->transmit(Mizar<T>::PARTY_K);
        vhat.getShare(0)->join();
        
        // finally.
        // std::cout << "online; finally" << std::endl;
        Delta = (Delta * (TPCFPRIME - 2) + 1) % TPCFPRIME;
        DeviceData<T> mp_(size); mp_.zero();
        mp_.receive(Mizar<T>::PARTY_K);
        mp_.join();
        mp_ *= Delta;
        mp_ += Gamma;
        thrust::copy(mp_.begin(), mp_.end(), result.getShare(0)->begin());
    } else {        // P0
        // Offline
        // std::cout << "dReLU from dealer;" << std::endl;
        // std::cout << "offline;" << std::endl;
        DeviceData<T> rp(size); Mizar<T> rz(size); DeviceData<T> sign_neg_r(size);
        rp.zero(); rz.zero(); sign_neg_r.zero();
        {
            #ifdef ENABLE_OFFLINE
            DeviceData<T> rhat(size); DeviceData<U> rhat_bv(bvsize);
            rhat.zero(), rhat_bv.zero();

            // rhat = 2^{l-1} + sign(-rx) * 2^{l-1} + rx
            rhat += *input.getShare(0); rhat += *input.getShare(1);
            sign_neg_r += rhat;
            sign_neg_r &= 1ull << (sizeof(T) * 8 - 1);            // b0 b1 b2 ... -> b0 0 0 ...
            sign_neg_r ^= 1ull << (sizeof(T) * 8 - 1);            // b0 0 0 ... -> not(b0) 0 0 ...
            rhat += 1ull << (sizeof(T) * 8 - 1); rhat += sign_neg_r;    // assert that the msb of rhat is 0.
            gpu::bitexpand(&rhat, &rhat_bv);
            MizarTPC<U> somewhat_awkward(bvsize);
            shareFromKing(rhat_bv, somewhat_awkward);
            #endif
        }

        // Online
        // std::cout << "online; gather" << std::endl;
        DeviceData<U> v0(bvsize), v1(bvsize);
        {
            v0.receive(Mizar<T>::PARTY_A), v1.receive(Mizar<T>::PARTY_B);
            v0.join(), v1.join(); v0 += v1; 
        }
        
        // std::cout << "online; compute flag" << std::endl;
        DeviceData<T> mp0_(size), mp1_(size), mp_(size); 
        {
            mp0_.zero(), mp1_.zero();
            logic_rshift(sign_neg_r, sign_neg_r, static_cast<T>(sizeof(T) * 8 - 1));
            mp0_ += sign_neg_r; mp0_ -= rp;
            mp1_ += sign_neg_r; mp1_ ^= 1ull; mp1_ -= rp;
        }

        auto zip_mp = thrust::make_zip_iterator(mp0_.begin(), mp1_.begin());
        DeviceData<U> flags(bvsize);
        auto check_v_op = [] __host__ __device__ (U vi) -> U {
            if (vi == 0) return 0;
            else return 1;
        };
        thrust::transform(v0.begin(), v0.end(), flags.begin(), check_v_op);
        DeviceData<U> reduced_flags(size);
        thrust::equal_to<U> binary_pred;
        thrust::multiplies<U> mult_op;
        auto range_iter = thrust::make_transform_iterator(thrust::make_counting_iterator(0), select_range_functor<U>(sizeof(T) * 8));
        thrust::reduce_by_key(
            range_iter, range_iter + bvsize, flags.begin(), flags.begin() /* only for placeholder */, reduced_flags.begin(), binary_pred, mult_op);
        reduced_flags ^= static_cast<U>(1);

        auto select_op = [] __host__ __device__ (thrust::tuple<U, U> x, U y) -> U {
            if (y == 0) return thrust::get<0>(x);
            else return thrust::get<1>(x);
        };
        thrust::transform(zip_mp, zip_mp + size, reduced_flags.begin(), mp_.begin(), select_op);

        // std::cout << "online; broadcast" << std::endl;
        mp_.transmit(Mizar<T>::PARTY_A);
        mp_.join();
        mp_.transmit(Mizar<T>::PARTY_B);
        mp_.join();

        thrust::copy(rz.getShare(0)->begin(), rz.getShare(0)->end(), result.getShare(0)->begin());
        thrust::copy(rz.getShare(1)->begin(), rz.getShare(1)->end(), result.getShare(1)->begin());
    }
}

template<typename T, typename I, typename I2>
void dReLUFromMizarOpt(const Mizar<T, I> &input, Mizar<T, I2> &result)
{
    size_t size = input.size();
    size_t bvsize = size * sizeof(T) * 8;
    using U = uint8_t;
    
    if (partyNum != Mizar<T>::PARTY_K)  // P1 or P2
    {
        MizarTPC<T> rp(size) , rz(size);   
        rp.zero(), rz.zero();           // PRF, private

        // Offline
        //- // std::cout << "offline;" << std::endl;
        T Delta = 0;                    // PRF, 0 or 1
        DeviceData<T> Gamma(size); MizarTPC<U> rhat_bv(bvsize);      
        Gamma.zero(), rhat_bv.zero();
        {
            #ifdef ENABLE_OFFLINE
            MizarTPC<T> ss_Gamma(size); ss_Gamma.zero();
            thrust::transform(rp.getShare(0)->begin(), rp.getShare(0)->end(), rz.getShare(0)->begin(), ss_Gamma.getShare(0)->begin(), compute_Gamma_server<T>(Delta));

            ss_Gamma.getShare(0)->transmit(1 - partyNum);
            Gamma.receive(1 - partyNum);
            ss_Gamma.getShare(0)->join(); Gamma.join();
            Gamma += *ss_Gamma.getShare(0);

            DeviceData<U> temp_1_{};
            shareFromKing(temp_1_, rhat_bv);
            #endif
        }

        // Online
        // step 1 & 2.
        DeviceData<U> mhat_bv(bvsize); 
        //- // std::cout << "online; step 1 & 2" << std::endl;
        {
            DeviceData<T> mhat(size); 
            thrust::transform(input.getShare(0)->begin(), input.getShare(0)->end(), mhat.begin(), compute_mhat_server<T>());
            gpu::bitexpand(&mhat, &mhat_bv);
        }

        // step 3.
        //- // std::cout << "online; step 3" << std::endl;
        MizarTPC<U> m(bvsize); 
        {            
            thrust::transform(rhat_bv.getShare(0)->begin(), rhat_bv.getShare(0)->end(), mhat_bv.begin(), m.getShare(0)->begin(), compute_m_server<U>());
        }
        
        // step 4.
        //- // std::cout << "online; step 4" << std::endl;
        DeviceData<U> w(bvsize);
        w.fill(1);  // TODO: PRG instead. Random elements in Fp*.

        // step 5.
        // --- mp, a.k.a. t
        //- // std::cout << "online; step 5" << std::endl;
        MizarTPC<U> mp(bvsize);
        {
            //- // std::cout << "online; step 5; before inclusive scan" << std::endl;
            MizarTPC<U> prefix_sum_m(bvsize); prefix_sum_m.zero();
            auto range_iter = thrust::make_transform_iterator(thrust::make_counting_iterator(0ull), select_range_functor<U>(sizeof(T) * 8));
            thrust::inclusive_scan_by_key(range_iter, range_iter + bvsize, m.getShare(0)->begin(), prefix_sum_m.getShare(0)->begin());      // t = 1 or t = 0?

            //- // std::cout << "online; step 5; after inclusive scan" << std::endl;
            thrust::transform(m.getShare(0)->begin(), m.getShare(0)->end(), prefix_sum_m.getShare(0)->begin(), mp.getShare(0)->begin(), compute_mp_server<U>());
        }
        
        // --- v
        //- // std::cout << "online; step 5, v;" << std::endl;
        MizarTPC<U>& v = mp; 
        {
            DeviceData<U> temp(size), msbm(bvsize);
            thrust::transform(input.getShare(0)->begin(), input.getShare(0)->end(), temp.begin(), compute_temp_server<T, U>());
            gpu::repeat(temp, msbm);  // v.getShare(0) = msbm

            thrust::transform(msbm.begin(), msbm.end(), mhat_bv.begin(), msbm.begin(), compute_msbm_server<U>(Delta));
            v += msbm; v *= w;    // TODO: fuse this kernel.
        }
        
        // step 6.
        MizarTPC<U> vhat(bvsize); vhat.zero();  
        // TODO: Use real permute.
        vhat += v; 

        // step 7.
        vhat.getShare(0)->transmit(Mizar<T>::PARTY_K);
        vhat.getShare(0)->join();
        
        // finally.
        Delta = (Delta * (TPCFPRIME - 2) + 1) % TPCFPRIME;
        result.getShare(0)->receive(Mizar<T>::PARTY_K);
        result.getShare(0)->join(); // result.getShare(0) is mp;
        
        thrust::transform(result.getShare(0)->begin(), result.getShare(0)->end(), Gamma.begin(), result.getShare(0)->begin(), compute_mp2_server<T>(Delta));
    } else {        // P0
        // Offline
        DeviceData<T> rp(size); Mizar<T> rz(size); DeviceData<T> sign_neg_r(size);
        rp.zero(); rz.zero(); sign_neg_r.zero();
        {
            #ifdef ENABLE_OFFLINE
            DeviceData<T> rhat(size); DeviceData<U> rhat_bv(bvsize);
            
            thrust::transform(input.getShare(0)->begin(), input.getShare(0)->end(), input.getShare(1)->begin(), rhat.begin(), compute_rhat_dealer<T>());
            thrust::transform(input.getShare(0)->begin(), input.getShare(0)->end(), input.getShare(1)->begin(), sign_neg_r.begin(), compute_sign_neg_r_dealer<T>());

            gpu::bitexpand(&rhat, &rhat_bv);
            MizarTPC<U> temp_2_{};
            shareFromKing(rhat_bv, temp_2_);
            #endif
        }

        // Online
        //- // std::cout << "online; gather" << std::endl;
        DeviceData<U> v0(bvsize), v1(bvsize);
        {
            v0.receive(Mizar<T>::PARTY_A), v1.receive(Mizar<T>::PARTY_B);
            v0.join(), v1.join(); v0 += v1; 
        }
        
        //- // std::cout << "online; compute flag" << std::endl;
        DeviceData<T> mp0_(size), mp1_(size), mp_(size); 
        {
            thrust::transform(sign_neg_r.begin(), sign_neg_r.end(), rp.begin(), mp0_.begin(), compute_mp0_server<T>());
            thrust::transform(sign_neg_r.begin(), sign_neg_r.end(), rp.begin(), mp1_.begin(), compute_mp1_server<T>());
        }

        DeviceData<U> reduced_flags(size);
        thrust::equal_to<U> binary_pred;
        auto range_iter = thrust::make_transform_iterator(thrust::make_counting_iterator(0), select_range_functor<U>(sizeof(T) * 8));
        thrust::reduce_by_key(
            range_iter, range_iter + bvsize, v0.begin(), v1.begin() /* only for placeholder */, reduced_flags.begin(), binary_pred, compute_flags_server<U>());
        reduced_flags ^= static_cast<U>(1);

        auto select_op = []  __device__ (thrust::tuple<U, U> x, U y) -> U {
            if (y == 0) return thrust::get<0>(x);
            else return thrust::get<1>(x);
        };
        auto zip_mp = thrust::make_zip_iterator(mp0_.begin(), mp1_.begin());
        thrust::transform(zip_mp, zip_mp + size, reduced_flags.begin(), mp_.begin(), select_op);

        //- // std::cout << "online; broadcast" << std::endl;
        mp_.transmit(Mizar<T>::PARTY_A);
        mp_.join();
        mp_.transmit(Mizar<T>::PARTY_B);
        mp_.join();

        thrust::copy(rz.getShare(0)->begin(), rz.getShare(0)->end(), result.getShare(0)->begin());
        thrust::copy(rz.getShare(1)->begin(), rz.getShare(1)->end(), result.getShare(1)->begin());
    }
}

template<typename T, typename I, typename I2, typename I3>
void ReLU(const Mizar<T, I> &input, Mizar<T, I2> &result, Mizar<T, I3> &dresult)
{
    dReLU(input, dresult);
    result.zero();
    result += input;
    result *= dresult;
}

template<typename T, typename I, typename I2>
void sqrt(const Mizar<T, I> &in, Mizar<T, I2> &out) {
    /*
     * Approximations:
     *   > sqrt(x) = 0.424 + 0.584(x)
     *     sqrt(x) = 0.316 + 0.885(x) - 0.202(x^2)
     */
    taylorSeries(in, out, 0.424, 0.584, 0.0, sqrt_lambda());
}

template<typename T, typename I, typename I2>
void inverse(const Mizar<T, I> &in, Mizar<T, I2> &out) {
    /*
     * Approximations:
     *     1/x = 2.838 - 1.935(x)
     *   > 1/x = 4.245 - 5.857(x) + 2.630(x^2)
     */
    taylorSeries(in, out, 4.245, -5.857, 2.630, inv_lambda());
}

template<typename T, typename I, typename I2>
void sigmoid(const Mizar<T, I> &in, Mizar<T, I2> &out) {
    /*
     * Approximation:
     *   > sigmoid(x) = 0.494286 + 0.275589(x) + -0.038751(x^2)
     */
    // taylorSeries(in, out, 0.494286, 0.275589, -0.038751, sigmoid_lambda());
    // f(x) = 0.5 + 0.125x if -4 <= x <= 4
    //        1            if       x > 4
    //        0            if  -4 > x
    DeviceData<T> knot{-4., 4.};
    DeviceData<T> coeff{0., 0., 0.5, 0.125, 1., 0.};
    evaluate_spline(in, out, knot, coeff, (T)FLOAT_PRECISION);
}

template<typename T, typename I, typename I2>
void GeLU(const Mizar<T, I> &in, Mizar<T, I2> &out) {
    DeviceData<T> knot{-4., -1.95, 3.};
    DeviceData<T> coeff{
        0., 0., 0., 0., 0., 0., 0.,
        -0.5054031199708174, -0.42226581151983866, -0.11807612951181953, -0.011034134030615728, 0., 0., 0.,
        0.008526321541038084,  0.5, 0.3603292692789629, 0.0, -0.037688200365904236, 0.0, 0.0018067462606141187,
        0., 1., 0., 0., 0., 0., 0.};
    evaluate_spline(in, out, knot, coeff, (T)FLOAT_PRECISION);
}

template<typename T, typename I, typename I2, typename I3>
void maxpool(const Mizar<T, I> &input, Mizar<T, I2> &result, Mizar<T, I3> &dresult, int k) {

    // d(Maxpool) setup
    dresult.fill(1);

    // split input into even, odd
    using SRIterator = typename StridedRange<I>::iterator;

    int stride = 2;
    int offset = 1;

    func_profiler.start();
    StridedRange<I> even0Range(input.getShare(0)->begin(), input.getShare(0)->end(), stride);
    DeviceData<T, SRIterator> even0(even0Range.begin(), even0Range.end());
    StridedRange<I> even1Range(input.getShare(1)->begin(), input.getShare(1)->end(), stride);
    DeviceData<T, SRIterator> even1(even1Range.begin(), even1Range.end());
    Mizar<T, SRIterator> even(&even0, &even1);

    StridedRange<I> odd0Range(input.getShare(0)->begin() + offset, input.getShare(0)->end(), stride);
    DeviceData<T, SRIterator> odd0(odd0Range.begin(), odd0Range.end());
    StridedRange<I> odd1Range(input.getShare(1)->begin() + offset, input.getShare(1)->end(), stride);
    DeviceData<T, SRIterator> odd1(odd1Range.begin(), odd1Range.end());
    Mizar<T, SRIterator> odd(&odd0, &odd1);
    func_profiler.accumulate("range creation");

    //printf("func-maxpool-post-rangecreate\n");
    //printMemUsage();

    while(k > 2) {

        // -- MP --

        // diff = even - odd
        func_profiler.start();
        Mizar<T> diff(even.size());
        diff.zero();
        diff += even;
        diff -= odd;
        func_profiler.accumulate("maxpool-diff");

        //printf("func-maxpool-post-diff-k=%d\n", k);
        //printMemUsage();

        // DRELU diff -> b
        func_profiler.start();
        Mizar<T> b(even.size());
        dReLU(diff, b);
        func_profiler.accumulate("maxpool-drelu");

        //printf("func-maxpool-post-drelu-k=%d\n", k);
        //printMemUsage();
        
        selectShare(odd, even, b, even);

        // unzip even -> into even, odd
        stride *= 2;

        //printf("func-maxpool-pre-rangeupdate-k=%d\n", k);
        //printMemUsage();

        func_profiler.start();
        even0Range.set(input.getShare(0)->begin(), input.getShare(0)->end(), stride);
        even0.set(even0Range.begin(), even0Range.end());
        even1Range.set(input.getShare(1)->begin(), input.getShare(1)->end(), stride);
        even1.set(even1Range.begin(), even1Range.end());
        even.set(&even0, &even1);

        odd0Range.set(input.getShare(0)->begin() + stride/2, input.getShare(0)->end(), stride);
        odd0.set(odd0Range.begin(), odd0Range.end());
        odd1Range.set(input.getShare(1)->begin() + stride/2, input.getShare(1)->end(), stride);
        odd1.set(odd1Range.begin(), odd1Range.end());
        odd.set(&odd0, &odd1);
        func_profiler.accumulate("maxpool-unzip");
        
        // -- dMP --

        //printf("func-maxpool-pre-expand-k=%d\n", k);
        //printMemUsage();

        // expandCompare b -> expandedB
        func_profiler.start();
        Mizar<T> negated(b.size());
        negated.fill(1);
        negated -= b;
        Mizar<T> expandedB(input.size());

        gpu::expandCompare(*b.getShare(0), *negated.getShare(0), *expandedB.getShare(0));
        gpu::expandCompare(*b.getShare(1), *negated.getShare(1), *expandedB.getShare(1));

        func_profiler.accumulate("maxpool-expandCompare");

        //printf("func-maxpool-post-expand-k=%d\n", k);
        //printMemUsage();
        
        // dresult &= expandedB
        func_profiler.start();
        dresult &= expandedB;
        func_profiler.accumulate("maxpool-dcalc");

        k /= 2;
    }

    // Fencepost - don't unzip the final results after the last comparison and finish
    // calculating derivative.
    
    // -- MP --
    
    // diff = even - odd
    func_profiler.start();
    Mizar<T> diff(even.size());
    diff.zero();
    diff += even;
    diff -= odd;
    func_profiler.accumulate("maxpool-z-diff");

    // DRELU diff -> b
    func_profiler.start();
    Mizar<T> b(even.size());
    dReLU(diff, b);
    func_profiler.accumulate("maxpool-z-drelu");
    
    // b * even + 1-b * odd
    selectShare(odd, even, b, even);

    func_profiler.start();
    //even *= b;
    //odd *= negated;
    //even += odd;

    result.zero();
    result += even;
    func_profiler.accumulate("maxpool-z-calc");

    // -- dMP --

    // expandCompare b -> expandedB
    func_profiler.start();
    Mizar<T> negated(b.size());
    negated.fill(1);
    negated -= b;
    Mizar<T> expandedB(input.size());
    gpu::expandCompare(*b.getShare(0), *negated.getShare(0), *expandedB.getShare(0));
    gpu::expandCompare(*b.getShare(1), *negated.getShare(1), *expandedB.getShare(1));
    func_profiler.accumulate("maxpool-z-expandCompare");
    
    // dresult &= expandedB
    func_profiler.start();
    dresult &= expandedB;
    func_profiler.accumulate("maxpool-z-dcalc");
}

template<typename T, typename I, typename I2>
void shareFromKing(const DeviceData<T, I> &input, MizarTPC<T, I2> &result)
{
    if (partyNum == Mizar<T>::PARTY_K) {
        //- // std::cout << "shareFromKing from dealer;" << std::endl;
        DeviceData<T> temp(input.size());
        thrust::copy(input.begin(), input.end(), temp.begin());
        DeviceData<T> share(input.size());
        share.zero(); // TODO: PRF instead
        temp -= share;
        temp.transmit(Mizar<T>::PARTY_A);
        temp.join();
        share.transmit(Mizar<T>::PARTY_B);
        share.join();
    } else {
        //- // std::cout << "shareFromKing from parties;" << std::endl;
        result.getShare(0)->receive(Mizar<T>::PARTY_K);
        result.getShare(0)->join();
    }
}

template<typename T, typename I, typename I2>
void getPowers(const Mizar<T, I> &in, DeviceData<T, I2> &pow) {

    Mizar<T> powers(pow.size()); // accumulates largest power yet tested that is less than the input val
    Mizar<T> currentPowerBit(in.size()); // current power
    Mizar<T> diff(in.size());
    Mizar<T> comparisons(in.size());

    for (int bit = 0; bit < sizeof(T) * 8; bit++) {
        currentPowerBit.fill(bit);

        diff.zero();
        diff += in;
        diff -= (((T)1) << bit);

        comparisons.zero();
        dReLU(diff, comparisons); // 0 -> current power is larger than input val, 1 -> input val is larger than current power

        // 0 -> keep val, 1 -> update to current known largest power less than input
        currentPowerBit -= powers;
        currentPowerBit *= comparisons;
        powers += currentPowerBit;
    }

    reconstruct(powers, pow);
}

template<typename T, typename I, typename I2, typename Functor>
void taylorSeries(const Mizar<T, I> &in, Mizar<T, I2> &out,
        double a0, double a1, double a2,
        Functor fn) {

    out.zero();
    Mizar<T> scratch(out.size());

    DeviceData<T> pow(out.size());
    getPowers(in, pow);
    pow += 1;

    DeviceData<T> ones(pow.size());
    ones.fill(1);
    ones <<= pow;

    if (a2 != 0.0) {
        DeviceData<T> a2Coeff(out.size());
        thrust::transform(
            pow.begin(), pow.end(), a2Coeff.begin(),
            tofixed_variable_precision_functor<T>(a2));

        scratch.zero();
        scratch += in;
        scratch *= in;
        dividePublic(scratch, ones);

        scratch *= a2Coeff;
        dividePublic(scratch, ones);
        out += scratch;
    }

    if (a1 != 0.0) {

        DeviceData<T> a1Coeff(out.size());
        thrust::transform(
            pow.begin(), pow.end(), a1Coeff.begin(),
            tofixed_variable_precision_functor<T>(a1));

        scratch.zero();
        scratch += in;
        scratch *= a1Coeff;
        dividePublic(scratch, ones);

        out += scratch;
    }

    DeviceData<T> a0Coeff(out.size());
    thrust::transform(
        pow.begin(), pow.end(), a0Coeff.begin(),
        tofixed_variable_precision_functor<T>(a0));
    out += a0Coeff;

    DeviceData<T> powCoeff(out.size());
    thrust::transform(
        pow.begin(), pow.end(), powCoeff.begin(),
        calc_fn<T, Functor>(fn));
    out *= powCoeff;

    dividePublic(out, ones);

    // turn values back to base (e.g. 20 bit) precision

    pow -= FLOAT_PRECISION;

    DeviceData<T> positivePow(pow.size());
    thrust::transform(
        pow.begin(), pow.end(), positivePow.begin(),
        filter_positive_powers<T>());

    ones.fill(1);
    ones <<= positivePow;

    dividePublic(out, ones);

    DeviceData<T> negativePow(pow.size());
    thrust::transform(
        pow.begin(), pow.end(), negativePow.begin(),
        filter_negative_powers<T>());

    for (int share = 0; share < Mizar<T>::numShares(); share++) {
        thrust::transform(
            out.getShare(share)->begin(), out.getShare(share)->end(), negativePow.begin(), out.getShare(share)->begin(),
            lshift_functor<T>()); 
    }
}