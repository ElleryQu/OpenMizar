
#include "unitTests.h"

#include "../gpu/DeviceData.h"
#include <cub/cub.cuh>

struct BMModule: public testing::Test {};

TEST(BMModule, Module8) {

    using DD = DeviceData<uint8_t>;
    using T = uint8_t;

    func_profiler.clear();

    if (partyNum >= 1) return;

    std::array<size_t, 1> exp_size{1ull << 20};
    T prime = 127;

    for (auto size: exp_size) {
        func_profiler.clear();

        DD inputs(size); inputs.fill(126);
        DD primes(size); primes.fill(prime);

        for (int i = 0; i <= EXP_TIMES + 10; i++) {
            DD input_(size); input_.zero();
            input_ += inputs;

            if (i >= 10) func_profiler.start();

            input_ *= inputs;
            input_ %= primes;
            
            cudaDeviceSynchronize();

            if (i >= 10) func_profiler.accumulate("module p time");

            CUDA_CHECK( cudaPeekAtLastError() );
            CUDA_CHECK( cudaDeviceSynchronize() );
        }

        printf("square-mod p (p=%d, N=%d) - %f sec.\n", prime, size, func_profiler.get_elapsed("module p time") / 1000.0 / EXP_TIMES);
        func_profiler.clear();

        for (int i = 0; i <= EXP_TIMES + 10; i++) {
            DD input_(size); input_.zero();
            input_ += inputs;

            if (i >= 10) func_profiler.start();

            input_ *= inputs;
            
            cudaDeviceSynchronize();

            if (i >= 10) func_profiler.accumulate("module 2^l time");

            CUDA_CHECK( cudaPeekAtLastError() );
            CUDA_CHECK( cudaDeviceSynchronize() );
        }

        printf("square-mod 2^l (l=%d, N=%d) - %f sec.\n", sizeof(T), size, func_profiler.get_elapsed("module 2^l time") / 1000.0 / EXP_TIMES);
    }   
}

TEST(BMModule, Module64) {

    using DD = DeviceData<uint64_t>;
    using T = uint64_t;

    func_profiler.clear();

    if (partyNum >= 1) return;

    std::array<size_t, 1> exp_size{1ull << 20};
    T prime = 18446744073709551557ull;

    for (auto size: exp_size) {
        func_profiler.clear();

        DD inputs(size); inputs.fill((1ull << 61) - 1);
        DD primes(size); primes.fill(prime);

        for (int i = 0; i <= EXP_TIMES + 10; i++) {
            DD input_(size); input_.zero();
            input_ += inputs;

            if (i >= 10) func_profiler.start();

            input_ *= inputs;
            input_ %= primes;
            
            cudaDeviceSynchronize();

            if (i >= 10) func_profiler.accumulate("module p time");

            CUDA_CHECK( cudaPeekAtLastError() );
            CUDA_CHECK( cudaDeviceSynchronize() );
        }

        printf("square-mod p (p=%d, N=%d) - %f sec.\n", prime, size, func_profiler.get_elapsed("module p time") / 1000.0 / EXP_TIMES);
        func_profiler.clear();

        for (int i = 0; i <= EXP_TIMES + 10; i++) {
            DD input_(size); input_.zero();
            input_ += inputs;

            if (i >= 10) func_profiler.start();

            input_ *= inputs;
            
            cudaDeviceSynchronize();

            if (i >= 10) func_profiler.accumulate("module 2^l time");

            CUDA_CHECK( cudaPeekAtLastError() );
            CUDA_CHECK( cudaDeviceSynchronize() );
        }

        printf("square-mod 2^l (l=%d, N=%d) - %f sec.\n", sizeof(T), size, func_profiler.get_elapsed("module 2^l time") / 1000.0 / EXP_TIMES);
    }   
}