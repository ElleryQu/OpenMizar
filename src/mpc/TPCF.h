/*
 * TPCF.h
 */

#pragma once

#include <cstddef>
#include <initializer_list>

#include <cutlass/conv/convolution.h>

#include "../gpu/DeviceData.h"
#include "../globals.h"

#define TPCFPRIME   127
#define TPCFPRIMEBW 7

/**
 * Additive secret-sharing over field F_{127}. Only supports operation with element in F_{127}.
 */
template <typename T, typename I>
class TPCFBase {

    protected:
        
        TPCFBase(DeviceData<T, I> *a);

    public:

        enum Party { PARTY_A, PARTY_B };
        static const int numParties = 2;

        void set(DeviceData<T, I> *a);
        size_t size() const;
        void zero();
        void fill(T val);
        void setPublic(std::vector<double> &v);
        DeviceData<T, I> *getShare(int i);
        const DeviceData<T, I> *getShare(int i) const;
        static int numShares();
        static int otherParty(int party);
        typedef T share_type;
        typedef I iterator_type;

        TPCFBase<T, I> &operator+=(const T rhs);
        TPCFBase<T, I> &operator-=(const T rhs);
        TPCFBase<T, I> &operator*=(const T rhs);
        TPCFBase<T, I> &operator>>=(const T rhs);

        template<typename I2>
        TPCFBase<T, I> &operator+=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        TPCFBase<T, I> &operator-=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        TPCFBase<T, I> &operator*=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        TPCFBase<T, I> &operator^=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        TPCFBase<T, I> &operator&=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        TPCFBase<T, I> &operator>>=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        TPCFBase<T, I> &operator<<=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        TPCFBase<T, I> &operator+=(const TPCFBase<T, I2> &rhs);
        template<typename I2>
        TPCFBase<T, I> &operator-=(const TPCFBase<T, I2> &rhs);
        template<typename I2>
        TPCFBase<T, I> &operator*=(const TPCFBase<T, I2> &rhs);
        template<typename I2>
        TPCFBase<T, I> &operator^=(const TPCFBase<T, I2> &rhs);
        template<typename I2>
        TPCFBase<T, I> &operator&=(const TPCFBase<T, I2> &rhs);

    protected:
        
        DeviceData<T, I> *share_;
};

template<typename T, typename I = BufferIterator<T> >
class TPCF : public TPCFBase<T, I> {

    public:

        TPCF(DeviceData<T, I> *a);
};

template<typename T>
class TPCF<T, BufferIterator<T> > : public TPCFBase<T, BufferIterator<T> > {

    public:

        TPCF(DeviceData<T> *a);
        TPCF(size_t n);
        TPCF(std::initializer_list<double> il, bool convertToFixedPoint = true);

        void resize(size_t n);

    private:

        DeviceData<T> share_buffer_;
};



#include "TPCF.inl"

