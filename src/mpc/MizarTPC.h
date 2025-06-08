#pragma once

#include <cstddef>
#include <initializer_list>

#include "TPCF.h"
#include "../gpu/DeviceData.h"
#include "../globals.h"

template<typename T, typename I = BufferIterator<T> >
class MizarTPC : public TPCFBase<T, I> {
public:
    MizarTPC(DeviceData<T, I> *a);
};

template<typename T>
class MizarTPC<T, BufferIterator<T> > : public TPCFBase<T, BufferIterator<T> > {
public:
    MizarTPC(DeviceData<T> *a);
    MizarTPC(size_t n);
    MizarTPC(std::initializer_list<double> il, bool convertToFixedPoint = true);

    void resize(size_t n);

private:
    DeviceData<T> share_buffer_;
};

template<typename T, typename I>
MizarTPC<T, I>::MizarTPC(DeviceData<T, I> *a) : TPCFBase<T, I>(a) {}

template<typename T>
MizarTPC<T, BufferIterator<T> >::MizarTPC(DeviceData<T> *a) :
    TPCFBase<T, BufferIterator<T> >(a) {}

template<typename T>
MizarTPC<T, BufferIterator<T> >::MizarTPC(size_t n) :
    share_buffer_(n),
    TPCFBase<T, BufferIterator<T> >(&share_buffer_) {}

template<typename T>
MizarTPC<T, BufferIterator<T> >::MizarTPC(std::initializer_list<double> il, bool convertToFixedPoint) :
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
        case 1:
            thrust::copy(shifted_vals.begin(), shifted_vals.end(), share_buffer_.begin());
            break;
        case 2:
            break;
    }
}

template<typename T>
void MizarTPC<T, BufferIterator<T> >::resize(size_t n) {
    share_buffer_.resize(n);
}