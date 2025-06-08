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
    RSS<                                                                                \
        typename std::remove_reference_t<decltype(*SNAME.getShare(1))>::BaseT,          \
        decltype(VNAME##_dd_iter_0_)>                                                   \
        VNAME(&VNAME##_dd_0_, &VNAME##_dd_1_);

template<typename T, typename I1, typename I2>
void evaluate_spline(
        const RSS<T, I1> &in, RSS<T, I2> &out,
        const DeviceData<T> &knot, const DeviceData<T> &coeff, T truncation) {
    
    assert(in.size() == out.size() && "in and out should have the same size.");
    assert(coeff.size() % (knot.size() + 1) == 0 && "coeff should be divisible by knot + 1.");

    size_t num_ele = in.size();
    size_t num_polys = knot.size() + 1;
    size_t degree = coeff.size() / num_polys - 1;

    RSS<uint8_t> compare(num_ele * num_polys); compare.fill(1); // num_ele X num_polys
    // RSS<T> compare(num_ele * num_polys); compare.fill(1); // num_ele X num_polys
    RSS<T> powers(num_ele * (degree + 1)); powers.fill(1); // num_ele X (degree + 1)

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
            RSS<T> temp(num_ele * (num_polys - 1));
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

        // test.
        // auto test_transform_func_ = [degree] __host__ __device__ (int i) {
        //     return i * (degree + 1) + 1;
        // };
        // auto test_counting_iter_ = thrust::counting_iterator<int>(0);
        // auto test_transform_iter_ = thrust::make_transform_iterator(test_counting_iter_, test_transform_func_);
        // auto test_dd_iter_ = thrust::make_permutation_iterator((*powers.getShare(0)).begin(), test_transform_iter_);
        // DeviceData<T, decltype(test_dd_iter_)> test_dd(test_dd_iter_, test_dd_iter_ + num_ele);
        // thrust::copy(in.getShare(0)->begin(), in.getShare(0)->end(), test_dd.begin());
        // thrust::copy(test_dd.begin(), test_dd.end(), (*powers.getShare(0)).begin());

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

    RSS<T> poly_result(num_ele * num_polys); 

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
        RSS<T> expand_powers(num_ele * num_polys * (degree + 1));
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
        RSS<T> zeros(num_ele * num_polys); zeros.zero();
        selectShare(poly_result, zeros, compare, poly_result);

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
