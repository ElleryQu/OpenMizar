#pragma once

#include <thrust/iterator/discard_iterator.h>

#include "../gpu/DeviceData.h"
#include "../gpu/functors.cuh"
#include "../util/functors.h"

// make thrust::transform_iterator from DeviceData DD and returns somewhat iterator VNAME.
#define MAKE_TITER_FROM_DD(VNAME, DD, FUNC)                                             \
    auto VNAME##_citer_ = thrust::counting_iterator<int>(0);                            \
    auto VNAME##_titer_ = thrust::make_transform_iterator(VNAME##_citer_, FUNC);        \
    auto VNAME = thrust::make_permutation_iterator((DD).begin(), VNAME##_titer_);

// apply transform iterator to a DeviceData DD and returns devicedata VNAME.
#define MAKE_TRANSFORM_TO_DD(VNAME, DD, SIZE, ...)                                      \
    auto VNAME##_transform_func_ = __VA_ARGS__;                                         \
    MAKE_TITER_FROM_DD(VNAME##_dd_iter_, DD, VNAME##_transform_func_);                  \
    DeviceData<                                                                         \
        typename std::remove_reference_t<decltype(DD)>::BaseT,                          \
        decltype(VNAME##_dd_iter_)>                                                     \
        VNAME(VNAME##_dd_iter_, VNAME##_dd_iter_ + SIZE);

// apply transform iterator to a STYPE share SNAME (only supports RSS, Mizar) and returns share VNAME.
#define MAKE_TRANSFORM_TO_SHARE(VNAME, SNAME, SIZE, ...)                                \
    auto VNAME##_transform_func_ = __VA_ARGS__;                                         \    
    MAKE_TITER_FROM_DD(VNAME##_dd_iter_0_, *SNAME.getShare(0), VNAME##_transform_func_);\
    MAKE_TITER_FROM_DD(VNAME##_dd_iter_1_, *SNAME.getShare(1), VNAME##_transform_func_);\
    DeviceData<                                                                         \
        typename std::remove_reference_t<decltype(*SNAME.getShare(0))>::BaseT,          \
        decltype(VNAME##_dd_iter_0_)>                                                   \
        VNAME##_dd_0_(VNAME##_dd_iter_0_, VNAME##_dd_iter_0_ + SIZE);                   \
    DeviceData<                                                                         \
        typename std::remove_reference_t<decltype(*SNAME.getShare(1))>::BaseT,          \
        decltype(VNAME##_dd_iter_0_)>                                                   \
        VNAME##_dd_1_(VNAME##_dd_iter_1_, VNAME##_dd_iter_1_ + SIZE);                   \
    Mizar<                                                                              \
        typename std::remove_reference_t<decltype(*SNAME.getShare(1))>::BaseT,          \
        decltype(VNAME##_dd_iter_0_)>                                                   \
        VNAME(&VNAME##_dd_0_, &VNAME##_dd_1_);


template<typename T>
__host__ __device__ T idiv_mod_127(T n) {
    const T m = 127;
    
    return n % m;
}

template<typename T>
__host__ __device__ T barrett_mod_127(T n) {
    const T k = 258;
    const T s = 15;

    T q = (n * k) >> s;
    T r = n - q * 127;

    r -= ((r >= 127) >> 31);

    return r & 0x7F;
}

// only supports up to 1 mult
template<typename T>
__host__ __device__ T mersenne_mod_127(T n) {
    const T p = 7;
    const T m = 127;

    n = (n >> p) + (n & m);

    return n - ((n >= m) * m); 
}

inline int get_num_steps(int n) {
    int steps = 0;
    while (n >>= 1) steps++;
    return steps;
}

// Hillis-steele algorithm
template<typename Func>
__global__ void hs_prefix_scan(uint32_t *input, uint32_t *output, uint32_t *part, size_t n, int num_steps, const Func& func) {
    extern __shared__ uint32_t s_data_u32[];  

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int offset = bid * blockDim.x;

    if (offset + tid < n) {
        s_data_u32[tid] = input[offset + tid];
    } else {
        s_data_u32[tid] = 0; 
    }
    
    uint32_t *src = s_data_u32;
    uint32_t *dst = s_data_u32 + blockDim.x;  
    
    __syncthreads();

    for (int d = 0; d < num_steps; ++d) {
        int stride = 1 << d; 
        if (tid >= stride) dst[tid] = func(src[tid], src[tid - stride]);
        else dst[tid] = src[tid];
        __syncthreads();
        
        uint32_t *temp = src;
        src = dst;
        dst = temp;
    }

    output[tid + offset] = src[tid];
    if (tid == blockDim.x - 1) part[bid] = src[tid];
    __syncthreads();
}

template<typename Func>
__global__ void hs_prefix_scan(uint64_t *input, uint64_t *output, uint64_t *part, size_t n, int num_steps, const Func& func) {
    extern __shared__ uint64_t s_data_u64[];  

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int offset = bid * blockDim.x;

    if (offset + tid < n) {
        s_data_u64[tid] = input[offset + tid];
    } else {
        s_data_u64[tid] = 0; 
    }
    
    uint64_t *src = s_data_u64;
    uint64_t *dst = s_data_u64 + blockDim.x;  
    
    __syncthreads();

    for (int d = 0; d < num_steps; ++d) {
        int stride = 1 << d; 
        if (tid >= stride) dst[tid] = func(src[tid], src[tid - stride]);
        else dst[tid] = src[tid];
        __syncthreads();
        
        uint64_t *temp = src;
        src = dst;
        dst = temp;
    }

    output[tid + offset] = src[tid];
    if (tid == blockDim.x - 1) part[bid] = src[tid];
    __syncthreads();
}

template<typename T, typename Func>
__global__ void adjust_results_kernel(T *output, T *part, size_t n, const Func& func) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int offset = bid * blockDim.x;

    T add_value = (bid > 0) ? part[bid - 1] : 0;

    if (offset + tid < n) {
        output[offset + tid] = func(output[offset + tid], add_value);
    }
}

template<typename T>
struct select_range_functor {

    const T size;

    select_range_functor(T size_): size(size_) {}

    __host__ __device__ T operator()(const T &x) const {
        return x / size;
    }
};

// knot size:   num_polys
// coeff size:  num_polys X (degree + 1)
template<typename T, typename I1, typename I2>
void evaluate_spline(
        const Mizar<T, I1> &in, Mizar<T, I2> &out,
        const DeviceData<T> &knot, const DeviceData<T> &coeff, T truncation) {
    
    assert(in.size() == out.size() && "in and out should have the same size.");
    assert(coeff.size() % (knot.size() + 1) == 0 && "coeff should be divisible by knot + 1.");

    size_t num_ele = in.size();
    size_t num_polys = knot.size() + 1;
    size_t degree = coeff.size() / num_polys - 1;

    Mizar<T> compare(num_ele * num_polys); compare.fill(1); // num_ele X num_polys
    Mizar<T> powers(num_ele * (degree + 1)); powers.fill(1); // num_ele X (degree + 1)

    // evaluate the compare results of the knot and the input.
    {
        // std::cout << "evaluate the compare results of the knot and the input." << std::endl;
        // compare the knot and the input: x > knot_i.
        {
            // std::cout << "compare the knot and the input: x > knot_i." << std::endl;
            MAKE_TRANSFORM_TO_SHARE(compare_slice, compare, num_ele * (num_polys - 1), 
                [num_polys] __host__ __device__ (int i) {
                    return i + i / (num_polys - 1) + 1;
                });
            MAKE_TRANSFORM_TO_SHARE(in_repeat, in, num_ele * (num_polys - 1), 
                [num_polys] __host__ __device__ (int i) {
                    return i / (num_polys - 1);
                });
            MAKE_TRANSFORM_TO_DD(knot_repeat, knot, num_ele * (num_polys - 1), 
                [num_polys] __host__ __device__ (int i) {
                    return i % (num_polys - 1);
                });
            Mizar<T> temp(num_ele * (num_polys - 1));
            thrust::copy(in_repeat.getShare(0)->begin(), in_repeat.getShare(0)->end(), temp.getShare(0)->begin());
            thrust::copy(in_repeat.getShare(1)->begin(), in_repeat.getShare(1)->end(), temp.getShare(1)->begin());
            temp -= knot_repeat;
            dReLU(temp, compare_slice);
        }

        // x^0 = 1, x^1 = x.
        {
            // std::cout << "x^0 = 1, x^1 = x." << std::endl;
            MAKE_TRANSFORM_TO_SHARE(powers_slice, powers, num_ele, 
                [degree] __host__ __device__ (int i) {
                    return i * (degree + 1) + 1;
                });
            thrust::copy(in.getShare(0)->begin(), in.getShare(0)->end(), powers_slice.getShare(0)->begin());
            thrust::copy(in.getShare(1)->begin(), in.getShare(1)->end(), powers_slice.getShare(1)->begin());
        }

        // given x, ..., x^{2^ld}, compute x^{2^ld+1}, ..., x^{2^{ld+1}}.
        size_t log2_degree = round_up_log_2(degree);
        // std::cout << "given x, ..., x^{2^ld}, compute x^{2^ld+1}, ..., x^{2^{ld+1}}." << std::endl;
        for (int ld = 0; ld < log2_degree; ld++) {
            // std::cout << "ld: " << ld << std::endl;
            // slice [1, 2^ld] and [2^ld+1, 2^(ld+1)].
            // logic shape:     [num_ele    , tile_size ]
            // logic stride:    [degree + 1 , 1         ]
            // logic offset:    1   (skip the first element, i.e. x^0)
            size_t tile_size = ld != log2_degree - 1 ? 1 << ld : degree - (1 << ld);
            MAKE_TRANSFORM_TO_SHARE(from_powers_slice, powers, num_ele * tile_size, 
                [tile_size, degree] __host__ __device__ (int i) {
                    size_t idx0 = i / tile_size, idx1 = i % tile_size;
                    return idx0 * (degree + 1) + idx1 + 1;
                });
            // logic shape:     [num_ele    , tile_size ]
            // logic stride:    [degree + 1 , 1         ]
            // logic offset:    tile_size + 1
            MAKE_TRANSFORM_TO_SHARE(target_powers_slice, powers, num_ele * tile_size, 
                [tile_size, degree] __host__ __device__ (int i) {
                    size_t idx0 = i / tile_size, idx1 = i % tile_size;
                    return idx0 * (degree + 1) + idx1 + tile_size + 1;
                });
            thrust::copy(from_powers_slice.getShare(0)->begin(), from_powers_slice.getShare(0)->end(), target_powers_slice.getShare(0)->begin());
            thrust::copy(from_powers_slice.getShare(1)->begin(), from_powers_slice.getShare(1)->end(), target_powers_slice.getShare(1)->begin());

            // std::cout << "select the 2^ld-th element." << std::endl;
            // select the 2^ld-th element.
            // logic shape:     [num_ele    , tile_size ]
            MAKE_TRANSFORM_TO_SHARE(selected_powers_slice, powers, num_ele * tile_size, 
                [tile_size, degree] __host__ __device__ (int i) {
                    size_t j = i / tile_size;
                    return j * (degree + 1) + tile_size;
                });

            // compute x^{2^ld+1}, ..., x^{2^{ld+1}}.
            target_powers_slice *= selected_powers_slice;
            dividePublic(target_powers_slice, (T)1 << truncation);
        }
    }

    Mizar<T> poly_result(num_ele * num_polys); 

    // evaluate each polynomial.
    {
        // std::cout << "evaluate each polynomial." << std::endl;
        // [num_ele, degree + 1] -> [num_ele, num_polys, degree + 1]
        // logic shape:     [num_ele, num_polys, degree + 1]
        // logic stride:    [num_polys * (degree + 1), degree + 1, 1]
        // logic offset:    0
        MAKE_TRANSFORM_TO_SHARE(powers_slice, powers, num_ele * num_polys * (degree + 1), 
            [num_polys, degree] __host__ __device__ (int i) {
                size_t idx0 = i / (num_polys * (degree + 1));
                size_t idx1 = (i / (degree + 1)) % num_polys;
                size_t idx2 = i % (degree + 1);
                return idx0 * (degree + 1) + idx2;
            });
        // [num_polys, degree + 1] -> [num_ele, num_polys, degree + 1]
        // logic shape:     [num_ele, num_polys, degree + 1]
        // logic stride:    [num_polys * (degree + 1), degree + 1, 1]
        // logic offset:    0
        MAKE_TRANSFORM_TO_DD(coeff_slice, coeff, num_polys * (degree + 1), 
            [num_polys, degree] __host__ __device__ (int i) {
                return i % ((degree + 1) * num_polys);
            });

        // std::cout << "compute coeff_i[0], coeff_i[1] * x, ..., coeff_i[degree] * x^degree." << std::endl;
        // compute coeff_i[0], coeff_i[1] * x, ..., coeff_i[degree] * x^degree.
        Mizar<T> expand_powers(num_ele * num_polys * (degree + 1));
        thrust::copy(powers_slice.getShare(0)->begin(), powers_slice.getShare(0)->end(), expand_powers.getShare(0)->begin());
        thrust::copy(powers_slice.getShare(1)->begin(), powers_slice.getShare(1)->end(), expand_powers.getShare(1)->begin());
        expand_powers *= coeff_slice;

        // std::cout << "compute f(x) = coeff_i[0] + coeff_i[1] * x + ... + coeff_i[degree] * x^degree." << std::endl;
        // compute f(x) = coeff_i[0] + coeff_i[1] * x + ... + coeff_i[degree] * x^degree.
        // [num_ele, num_polys, degree + 1] -> [num_ele, num_polys]
        auto counting_iter = thrust::counting_iterator<int>(0);
        auto reduce_flag_iter = thrust::make_transform_iterator(
            counting_iter, [num_polys, num_ele] __host__ __device__ (int i) {
                return i / (num_ele * num_polys);
            });
        thrust::reduce_by_key(
            reduce_flag_iter, reduce_flag_iter + num_ele * num_polys * (degree + 1), 
            expand_powers.getShare(0)->begin(), thrust::make_discard_iterator(), poly_result.getShare(0)->begin(), 
            thrust::equal_to<T>(), thrust::plus<T>());
        thrust::reduce_by_key(
            reduce_flag_iter, reduce_flag_iter + num_ele * num_polys * (degree + 1), 
            expand_powers.getShare(1)->begin(), thrust::make_discard_iterator(), poly_result.getShare(1)->begin(), 
            thrust::equal_to<T>(), thrust::plus<T>());
        dividePublic(poly_result, (T)1 << truncation);
    }

    // evaluate the comparison result with each knot and compute the final result.
    {
        // std::cout << "evaluate the comparison result with each knot and compute the final result." << std::endl;
        // compute the indice vector.
        // stride range.
        MAKE_TRANSFORM_TO_SHARE(compare_slice_skip_back, compare, num_ele * (num_polys - 1), 
            [num_polys] __host__ __device__ (int i) {
                return i + i / (num_polys - 1);
            });
        MAKE_TRANSFORM_TO_SHARE(compare_slice_skip_front, compare, num_ele * (num_polys - 1), 
            [num_polys] __host__ __device__ (int i) {
                return i + i / (num_polys - 1) + 1;
            });
        compare_slice_skip_back ^= compare_slice_skip_front;

        // compute b_0 * f_0(x), b_1 * f_1(x), ..., b_{num_polys - 1} * f_{num_polys - 1}(x).
        Mizar<T> zeros(num_ele * num_polys); zeros.zero();
        poly_result *= compare;

        // compute b_0 * f_0(x) + b_1 * f_1(x) + ... + b_{num_polys - 1} * f_{num_polys - 1}(x).
        auto counting_iter = thrust::counting_iterator<int>(0);
        auto transform_func = [num_polys] __host__ __device__ (int i) {
            return i / num_polys;
        };
        auto reduce_flag_iter = thrust::make_transform_iterator(counting_iter, transform_func);
        thrust::reduce_by_key(
            reduce_flag_iter, reduce_flag_iter + num_ele * num_polys, 
            poly_result.getShare(0)->begin(), thrust::make_discard_iterator(), out.getShare(0)->begin(), 
            thrust::equal_to<T>(), thrust::plus<T>());
        thrust::reduce_by_key(
            reduce_flag_iter, reduce_flag_iter + num_ele * num_polys, 
            poly_result.getShare(1)->begin(), thrust::make_discard_iterator(), out.getShare(1)->begin(), 
            thrust::equal_to<T>(), thrust::plus<T>());
    }
}


// dReLU support.
template<typename U, int BLOCK_DIM>
__global__ void compute_v_of_drelu(U *msbm, U *mhat_bv, U* mp, U* w, U* res, U Delta, int n) {

    int tid = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int stride = BLOCK_DIM * gridDim.x;

    if (tid < n) res[tid] = (((msbm[tid] ^ mhat_bv[tid] ^ Delta) + mp[tid]) * w) % TPCFPRIME;
    if (tid + stride < n) res[tid + stride] = (((msbm[tid + stride] ^ mhat_bv[tid + stride] ^ Delta) + mp[tid + stride]) * w) % TPCFPRIME;
}

// Gamma = \Delta + rp - 2 * \Delta * rp + rz
template<typename T>
struct compute_Gamma_server {
    T Delta;
    compute_Gamma_server(T Delta_) : Delta(Delta_) {}
     __device__ T operator() (T rp, T rz) {
        T res = (Delta + rp + (TPCFPRIME - 2) * Delta * rp + rz) % TPCFPRIME;
        return res;
    }
};

template<typename T>
struct compute_mhat_server {
    __device__ T operator() (T input) {
       T res = input & ((1ull << (sizeof(T) * 8 - 1)) - 1);    // b0 b1 b2 ... -> 0 b1 b2 ...
       res ^= 1ull << (sizeof(T) * 8 - 1);                     // 0 b1 b2 ... -> 1 b1 b2 ...
       return res;
   }
};

template<typename U>
struct compute_m_server {
    __device__ U operator() (U rhat_bv, U mhat_bv) {
       U res = (mhat_bv + rhat_bv + (TPCFPRIME - 2) * mhat_bv * rhat_bv) % TPCFPRIME;
       return res;
   }
};

template<typename U>
struct compute_mp_server {
    __device__ U operator() (U m, U prefix_sum_m) {
       U res = ((TPCFPRIME - 2) * m + 1 + prefix_sum_m) % TPCFPRIME;
       return res;
   }
};

template<typename T>
struct compute_mp2_server {
    T Delta;

    compute_mp2_server(T Delta_) : Delta(Delta_) {}

     __device__ T operator() (T mp, T Gamma) {
        T res = (mp * Delta + Gamma) % TPCFPRIME;
        return res;
    }
};

template<typename T, typename U>
struct compute_temp_server {
    __device__ T operator() (T input) {
       U res = (input & (1ull << (sizeof(T) * 8 - 1))) >> static_cast<T>(sizeof(T) * 8 - 1);    // b0 b1 b2 ... -> b0 0 0 ...
       return res;
   }
};

template<typename U>
struct compute_msbm_server {
    U Delta;

    compute_msbm_server(U Delta_) : Delta(Delta_) {}

     __device__ U operator() (U msbm, U mhat_bv) {
        U res = msbm ^ Delta ^ mhat_bv;
        return res;
    }
};

// rhat = 2^{l-1} + sign(-rx) * 2^{l-1} + rx
template<typename T>
struct compute_rhat_dealer {
    __device__ T operator() (T input0, T input1) {
       T input = input0 + input1;
       T res = input & (1ull << (sizeof(T) * 8 - 1));  // b0 b1 b2 ... -> b0 0 0 ...
       res ^= 1ull << (sizeof(T) * 8 - 1);             // b0 b1 b2 ... -> not(b0) 0 0 ...
       res += input + 1ull << (sizeof(T) * 8 - 1);
       return res;
   }
};

template<typename T>
struct compute_sign_neg_r_dealer {
    __device__ T operator() (T input0, T input1) {
       T input = input0 + input1;
       T res = input & (1ull << (sizeof(T) * 8 - 1));  // b0 b1 b2 ... -> b0 0 0 ...
       res ^= 1ull << (sizeof(T) * 8 - 1);             // b0 b1 b2 ... -> not(b0) 0 0 ...
       res >>= (sizeof(T) * 8 - 1);
       return res;
   }
};

template<typename T>
struct compute_mp0_server {
    __device__ T operator() (T sign_neg_r, T rp) {
       T res = sign_neg_r - rp;
       return res;
   }
};

template<typename T>
struct compute_mp1_server {
    __device__ T operator() (T sign_neg_r, T rp) {
       T res = (sign_neg_r ^ 1ull) - rp;
       return res;
   }
};

template<typename U>
struct compute_flags_server {
    __device__ U operator() (U input0, U input1) {
       return (input0 * input1) != 0;
   }
};
