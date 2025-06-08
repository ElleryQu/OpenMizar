#pragma once

#include <cstddef>
#include <initializer_list>
#include <cutlass/conv/convolution.h>

#include "../gpu/DeviceData.h"
#include "../globals.h"
#include "MizarTPC.h"


template <typename T, typename I>
class MizarBase {

    protected:
        
        MizarBase(DeviceData<T, I> *a, DeviceData<T, I> *b);

    public:

        enum Party { PARTY_A, PARTY_B, PARTY_K };
        static const int numParties = 3;

        void set(DeviceData<T, I> *m, DeviceData<T, I> *r);
        size_t size() const;
        void zero();
        void fill(T val);
        void setPublic(std::vector<double> &v);
        DeviceData<T, I> *getShare(int i);
        const DeviceData<T, I> *getShare(int i) const;
        static int numShares();
        static int nextParty(int party);
        static int prevParty(int party);
        typedef T share_type;
        //using share_type = T;
        typedef I iterator_type;


        MizarBase<T, I> &operator+=(const T rhs);
        MizarBase<T, I> &operator-=(const T rhs);
        MizarBase<T, I> &operator*=(const T rhs);
        MizarBase<T, I> &operator>>=(const T rhs);

        template<typename I2>
        MizarBase<T, I> &operator+=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        MizarBase<T, I> &operator-=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        MizarBase<T, I> &operator*=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        MizarBase<T, I> &operator^=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        MizarBase<T, I> &operator&=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        MizarBase<T, I> &operator>>=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        MizarBase<T, I> &operator<<=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        MizarBase<T, I> &operator+=(const MizarBase<T, I2> &rhs);
        template<typename I2>
        MizarBase<T, I> &operator-=(const MizarBase<T, I2> &rhs);
        template<typename I2>
        MizarBase<T, I> &operator*=(const MizarBase<T, I2> &rhs);
        template<typename I2>
        MizarBase<T, I> &operator^=(const MizarBase<T, I2> &rhs);
        template<typename I2>
        MizarBase<T, I> &operator&=(const MizarBase<T, I2> &rhs);

    protected:
        
        DeviceData<T, I> *share_m_;
        DeviceData<T, I> *share_r_;
};

template<typename T, typename I = BufferIterator<T> >
class Mizar : public MizarBase<T, I> {

    public:

        Mizar(DeviceData<T, I> *a, DeviceData<T, I> *b);
};

template<typename T>
class Mizar<T, BufferIterator<T> > : public MizarBase<T, BufferIterator<T> > {

    public:

        Mizar(DeviceData<T> *a, DeviceData<T> *b);
        Mizar(size_t n);
        Mizar(std::initializer_list<double> il, bool convertToFixedPoint = true);

        void resize(size_t n);

    private:

        DeviceData<T> share_m_buffer_;
        DeviceData<T> share_r_buffer_;
};

template<typename T, typename I, typename I2>
void reconstruct(Mizar<T, I> &in, DeviceData<T, I2> &out);

template<typename T>
void matmul(const Mizar<T> &a, const Mizar<T> &b, Mizar<T> &c,
        int M, int N, int K,
        bool transpose_a, bool transpose_b, bool transpose_c, T truncation);

template<typename T, typename I, typename I2, typename I3, typename I4>
void selectShare(const Mizar<T, I> &x, const Mizar<T, I2> &y, const Mizar<T, I3> &b, Mizar<T, I4> &z);

template<typename T, typename U, typename I, typename I2, typename I3, typename I4>
void selectShare(const Mizar<T, I> &x, const Mizar<T, I2> &y, const Mizar<U, I3> &b, Mizar<T, I4> &z);

template<typename T, typename I, typename I2>
void sqrt(const Mizar<T, I> &in, Mizar<T, I2> &out);

template<typename T, typename I, typename I2>
void inverse(const Mizar<T, I> &in, Mizar<T, I2> &out);

template<typename T, typename I, typename I2>
void sigmoid(const Mizar<T, I> &in, Mizar<T, I2> &out);

template<typename T, typename I, typename I2>
void GeLU(const Mizar<T, I> &in, Mizar<T, I2> &out);

template<typename T>
void convolution(const Mizar<T> &A, const Mizar<T> &B, Mizar<T> &C,
        cutlass::conv::Operator op,
        int batchSize, int imageHeight, int imageWidth, int filterSize,
        int Din, int Dout, int stride, int padding, int truncation);

template<typename T, typename I, typename I2>
void dReLU(const Mizar<T, I> &input, Mizar<T, I2> &result);
 
template<typename T, typename I, typename I2, typename I3>
void ReLU(const Mizar<T, I> &input, Mizar<T, I2> &result, Mizar<T, I3> &dresult);

template<typename T, typename I, typename I2, typename I3>
void maxpool(const Mizar<T, I> &input, Mizar<T, I2> &result, Mizar<T, I3> &dresult, int k);


template<typename T, typename I, typename I2>
void shareFromKing(const DeviceData<T, I> &input, MizarTPC<T, I2> &result);

template<typename T, typename I, typename I2>
void OpenToKing(const MizarTPC<T, I> &input, DeviceData<T, I2> &result);

#include "Mizar.inl"