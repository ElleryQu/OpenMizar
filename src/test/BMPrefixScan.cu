
// #include "unitTests.h"
// #include "../mpc/MizarKernels.cuh"

// #include <cutlass/conv/convolution.h>
// #include <cub/cub.cuh>

// template<typename T>
// struct BMPrefixScan : public testing::Test {
//     using ParamType = T;
// };

// using Types = testing::Types<
//                 DeviceData<uint32_t>,
//                 DeviceData<uint64_t>>;
// TYPED_TEST_CASE(BMPrefixScan, Types);

// int TEST_PRIME = 127;

// template<typename T>
// struct add_mod_idiv_functor {
//     __host__ __device__ T operator()(const T &x, const T &y) const {
//         return idiv_mod_127(x + y);
//     }
// };

// template<typename T>
// struct add_mod_barrett_functor {
//     __host__ __device__ T operator()(const T &x, const T &y) const {
//         return barrett_mod_127(x + y);
//     }
// };

// template<typename T>
// struct add_mod_mersenne_functor {
//     __host__ __device__ T operator()(const T &x, const T &y) const {
//         return mersenne_mod_127(x + y);
//     }
// };


// TYPED_TEST(BMPrefixScan, IDIV) {

//     using DD = typename TestFixture::ParamType;
//     using T = typename DD::BaseT;

//     func_profiler.clear();

//     if (partyNum >= 1) return;

//     std::array<size_t, 2> exp_size{1ull << 10, 1ull << 20};

//     for (auto size: exp_size) {
//         func_profiler.clear();

//         DD inputs(size); inputs.fill(1);
//         DD outputs(size);
//         DD part(size / 1024);
//         DD part_temp(size / 1024);
//         add_mod_idiv_functor<T> amf{};


//         for (int i = 0; i <= EXP_TIMES + 10; i++) {
//             if (i >= 10) func_profiler.start();
//             assert(size % 1024 == 0);
//             hs_prefix_scan<<<size / 1024 + ((size % 1024) != 0), 1024, 2*1024*sizeof(T)>>>(
//                 thrust::raw_pointer_cast(&inputs.raw()[0]), 
//                 thrust::raw_pointer_cast(&outputs.raw()[0]), 
//                 thrust::raw_pointer_cast(&part.raw()[0]),
//                 size, get_num_steps(1024), amf);
//             cudaDeviceSynchronize();
//             hs_prefix_scan<<<1, size / 1024, size/512*sizeof(T)>>>(
//                 thrust::raw_pointer_cast(&part.raw()[0]), 
//                 thrust::raw_pointer_cast(&part.raw()[0]), 
//                 thrust::raw_pointer_cast(&part_temp.raw()[0]),
//                 size, get_num_steps(size/1024), amf);
//             cudaDeviceSynchronize();
//             adjust_results_kernel<<<size / 1024, 1024>>>(
//                 thrust::raw_pointer_cast(&outputs.raw()[0]), 
//                 thrust::raw_pointer_cast(&part.raw()[0]), 
//                 size, amf);
//             cudaDeviceSynchronize();

//             if (i >= 10) func_profiler.accumulate("prefix scan naive idiv");

//             CUDA_CHECK( cudaPeekAtLastError() );
//             CUDA_CHECK( cudaDeviceSynchronize() );
//         }

//         printf("prefix scan naive idiv (N=%d) - %f sec.\n", size, func_profiler.get_elapsed("prefix scan naive idiv") / 1000.0 / EXP_TIMES);
//         EXPECT_EQ(*(outputs.end() - 1), size % TEST_PRIME) << "Naive prefix scan with idiv get wrong result!";

//         outputs.fill(0);

//         for (int i = 0; i < EXP_TIMES + 10; i++) {
//             if (i >= 10) func_profiler.start();
//             thrust::inclusive_scan(inputs.raw().begin(), inputs.raw().end(), outputs.raw().begin(), amf);
//             cudaDeviceSynchronize();
//             if (i >= 10) func_profiler.accumulate("prefix scan ours idiv");
//         }
        
//         printf("prefix scan ours idiv (N=%d) - %f sec.\n", size, func_profiler.get_elapsed("prefix scan ours idiv") / 1000.0 / EXP_TIMES);
//         EXPECT_EQ(*(outputs.end() - 1), size % TEST_PRIME) << "Our prefix scan with idiv get wrong result!";
//     }   
// }

// TYPED_TEST(BMPrefixScan, BARRETT) {

//     using DD = typename TestFixture::ParamType;
//     using T = typename DD::BaseT;

//     func_profiler.clear();

//     if (partyNum >= 1) return;

//     std::array<size_t, 2> exp_size{1ull << 10, 1ull << 20};

//     for (auto size: exp_size) {
//         func_profiler.clear();

//         DD inputs(size); inputs.fill(1);
//         DD outputs(size);
//         DD part(size / 1024);
//         DD part_temp(size / 1024);
//         add_mod_barrett_functor<T> amf{};

//         for (int i = 0; i <= EXP_TIMES + 10; i++) {
//             if (i >= 10) func_profiler.start();

//             hs_prefix_scan<<<size / 1024 + ((size % 1024) != 0), 1024, 2*1024*sizeof(T)>>>(
//                 thrust::raw_pointer_cast(&inputs.raw()[0]), 
//                 thrust::raw_pointer_cast(&outputs.raw()[0]), 
//                 thrust::raw_pointer_cast(&part.raw()[0]),
//                 size, get_num_steps(1024), amf);
//             cudaDeviceSynchronize();
//             hs_prefix_scan<<<1, size / 1024, size/512*sizeof(T)>>>(
//                 thrust::raw_pointer_cast(&part.raw()[0]), 
//                 thrust::raw_pointer_cast(&part.raw()[0]), 
//                 thrust::raw_pointer_cast(&part_temp.raw()[0]),
//                 size, get_num_steps(size / 1024), amf);
//             cudaDeviceSynchronize();
//             adjust_results_kernel<<<size / 1024, 1024>>>(
//                 thrust::raw_pointer_cast(&outputs.raw()[0]), 
//                 thrust::raw_pointer_cast(&part.raw()[0]), 
//                 size, amf);
//             cudaDeviceSynchronize();

//             if (i >= 10) func_profiler.accumulate("prefix scan naive barrett");

//             CUDA_CHECK( cudaPeekAtLastError() );
//             CUDA_CHECK( cudaDeviceSynchronize() );
//         }

//         printf("prefix scan naive barrett (N=%d) - %f sec.\n", size, func_profiler.get_elapsed("prefix scan naive barrett") / 1000.0 / EXP_TIMES);
//         EXPECT_EQ(*(outputs.end() - 1), size % TEST_PRIME) << "Naive prefix scan with barrett get wrong result!";

//         outputs.fill(0);

//         for (int i = 0; i < EXP_TIMES + 10; i++) {
//             if (i >= 10) func_profiler.start();
//             thrust::inclusive_scan(inputs.raw().begin(), inputs.raw().end(), outputs.raw().begin(), amf);
//             cudaDeviceSynchronize();
//             if (i >= 10) func_profiler.accumulate("prefix scan ours barrett");
//         }
        
//         printf("prefix scan ours barrett (N=%d) - %f sec.\n", size, func_profiler.get_elapsed("prefix scan ours barrett") / 1000.0 / EXP_TIMES);
//         EXPECT_EQ(*(outputs.end() - 1), size % TEST_PRIME) << "Our prefix scan with barrett get wrong result!";
//     }
// }

// TYPED_TEST(BMPrefixScan, MERSENNE) {

//     using DD = typename TestFixture::ParamType;
//     using T = typename DD::BaseT;

//     func_profiler.clear();

//     if (partyNum >= 1) return;

//     std::array<size_t, 2> exp_size{1ull << 10, 1ull << 20};

//     for (auto size: exp_size) {
//         func_profiler.clear();

//         DD inputs(size); inputs.fill(1);
//         DD outputs(size);
//         DD part(size / 1024);
//         DD part_temp(size / 1024);
//         add_mod_mersenne_functor<T> amf{};

//         for (int i = 0; i <= EXP_TIMES + 10; i++) {
//             if (i >= 10) func_profiler.start();

//             hs_prefix_scan<<<size / 1024 + ((size % 1024) != 0), 1024, 2*1024*sizeof(T)>>>(
//                 thrust::raw_pointer_cast(&inputs.raw()[0]), 
//                 thrust::raw_pointer_cast(&outputs.raw()[0]), 
//                 thrust::raw_pointer_cast(&part.raw()[0]),
//                 size, get_num_steps(1024), amf);
//             cudaDeviceSynchronize();
//             hs_prefix_scan<<<1, size / 1024, size/512*sizeof(T)>>>(
//                 thrust::raw_pointer_cast(&part.raw()[0]), 
//                 thrust::raw_pointer_cast(&part.raw()[0]), 
//                 thrust::raw_pointer_cast(&part_temp.raw()[0]),
//                 size, get_num_steps(size/1024), amf);
//             cudaDeviceSynchronize();
//             adjust_results_kernel<<<size / 1024, 1024>>>(
//                 thrust::raw_pointer_cast(&outputs.raw()[0]), 
//                 thrust::raw_pointer_cast(&part.raw()[0]), 
//                 size, amf);
//             cudaDeviceSynchronize();

//             if (i >= 10) func_profiler.accumulate("prefix scan naive mersenne");

//             CUDA_CHECK( cudaPeekAtLastError() );
//             CUDA_CHECK( cudaDeviceSynchronize() );
//         }

//         printf("prefix scan naive mersenne (N=%d) - %f sec.\n", size, func_profiler.get_elapsed("prefix scan naive mersenne") / 1000.0 / EXP_TIMES);
//         EXPECT_EQ(*(outputs.end() - 1), size % TEST_PRIME) << "Naive prefix scan with mersenne get wrong result!";

//         outputs.fill(0);

//         size_t temp_storage_bytes = 0;
//         for (int i = 0; i < EXP_TIMES + 10; i++) {
//             if (i >= 10) func_profiler.start();
//             thrust::inclusive_scan(inputs.raw().begin(), inputs.raw().end(), outputs.raw().begin(), amf);
//             cudaDeviceSynchronize();
//             if (i >= 10) func_profiler.accumulate("prefix scan ours mersenne");
//         }
        
//         printf("prefix scan ours mersenne (N=%d) - %f sec.\n", size, func_profiler.get_elapsed("prefix scan ours mersenne") / 1000.0 / EXP_TIMES);
//         EXPECT_EQ(*(outputs.end() - 1), size % TEST_PRIME) << "Our prefix scan with mersenne get wrong result!";
//     }
// }