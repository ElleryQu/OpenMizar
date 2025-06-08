
#include "unitTests.h"

#include <cutlass/conv/convolution.h>

#if defined(MIZAR) || defined(AEGIS)
template<typename T>
struct BMMizarFunc : public testing::Test {
    using ParamType = T;
    using DataType = typename T::share_type;

    static void SetUpTestSuite() {
        #if defined(MIZAR)
        std::cout << "Benchamrk for Mizar, including dReLU and activation functions." << std::endl;
        #elif defined(AEGIS)
        std::cout << "Benchamrk for Mizar, including dReLU and activation functions." << std::endl;
        #endif
        std::cout << "Ring bitwidth: " << sizeof(DataType) << std::endl;
    }
};

using Types = testing::Types<
                    Mizar<uint32_t>,
                    Mizar<uint64_t>>;
TYPED_TEST_CASE(BMMizarFunc, Types);

TYPED_TEST(BMMizarFunc, DReLU) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    std::array<size_t, 2> exp_size{1ull << 10, 1ull << 20};
    // std::array<size_t, 1> exp_size{1ull << 10};

    for (auto size: exp_size) {
        func_profiler.clear(), comm_profiler.clear(), memory_profiler.clear();

        for (int i = 0; i < EXP_TIMES + 5; i++) {
            if (i >= 5) func_profiler.start(), comm_profiler.start(), memory_profiler.start();
            Share inputs(size);
            Share outputs(size);
            // dReLUFromMizar(inputs, outputs);
            dReLU(inputs, outputs);
            cudaDeviceSynchronize();
            if (i >= 5) func_profiler.accumulate("drelu for mizar"), comm_profiler.accumulate("drelu for mizar");
        }

        std::cout << "n: " << size << std::endl;
        printf("drelu for mizar (N=%d) - %f sec.\n", size, func_profiler.get_elapsed("drelu for mizar") / 1000.0 / EXP_TIMES);
        printf("tx comm: %f, rx comm: %f \n", 
            comm_profiler.get_comm_tx_bytes() / 1024. / 1024. / EXP_TIMES,
            comm_profiler.get_comm_rx_bytes() / 1024. / 1024. / EXP_TIMES);
        printf("memory: %f MB\n", memory_profiler.get_max_mem_mb() / EXP_TIMES);
    }
}

TYPED_TEST(BMMizarFunc, ReLU) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    std::array<size_t, 2> exp_size{1ull << 10, 1ull << 20};
    // std::array<size_t, 1> exp_size{1ull << 10};

    for (auto size: exp_size) {
        func_profiler.clear(), comm_profiler.clear(), memory_profiler.clear();

        for (int i = 0; i < EXP_TIMES + 5; i++) {
            if (i >= 5) func_profiler.start(), comm_profiler.start(), memory_profiler.start();
            Share inputs(size);
            Share outputs(size);
            Share doutputs(size);
            ReLU(inputs, outputs, doutputs);
            cudaDeviceSynchronize();
            if (i >= 5) func_profiler.accumulate("relu for mizar"), comm_profiler.accumulate("relu for mizar");
        }

        std::cout << "n: " << size << std::endl;
        printf("relu for mizar (N=%d) - %f sec.\n", size, func_profiler.get_elapsed("relu for mizar") / 1000.0 / EXP_TIMES);
        printf("tx comm: %f, rx comm: %f \n", 
            comm_profiler.get_comm_tx_bytes() / 1024. / 1024. / EXP_TIMES,
            comm_profiler.get_comm_rx_bytes() / 1024. / 1024. / EXP_TIMES);
        printf("memory: %f MB\n", memory_profiler.get_max_mem_mb() / EXP_TIMES);
    }
}

TYPED_TEST(BMMizarFunc, GeLU) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    std::array<size_t, 2> exp_size{1ull << 10, 1ull << 20};
    // std::array<size_t, 1> exp_size{1ull << 10};

    for (auto size: exp_size) {
        func_profiler.clear(), comm_profiler.clear(), memory_profiler.clear();

        for (int i = 0; i < EXP_TIMES + 5; i++) {
            if (i >= 5) func_profiler.start(), comm_profiler.start(), memory_profiler.start();
            Share inputs(size);
            Share outputs(size);
            GeLU(inputs, outputs);
            cudaDeviceSynchronize();
            if (i >= 5) func_profiler.accumulate("gelu for mizar"), comm_profiler.accumulate("gelu for mizar");
        }

        std::cout << "n: " << size << std::endl;
        printf("gelu for mizar (N=%d) - %f sec.\n", size, func_profiler.get_elapsed("gelu for mizar") / 1000.0 / EXP_TIMES);
        printf("tx comm: %f, rx comm: %f \n", 
            comm_profiler.get_comm_tx_bytes() / 1024. / 1024. / EXP_TIMES,
            comm_profiler.get_comm_rx_bytes() / 1024. / 1024. / EXP_TIMES);
        printf("memory: %f MB\n", memory_profiler.get_max_mem_mb() / EXP_TIMES);
    }
}

TYPED_TEST(BMMizarFunc, Sigmoid) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    std::array<size_t, 2> exp_size{1ull << 10, 1ull << 20};
    // std::array<size_t, 1> exp_size{1ull << 10};

    for (auto size: exp_size) {
        func_profiler.clear(), comm_profiler.clear(), memory_profiler.clear();

        for (int i = 0; i < EXP_TIMES + 5; i++) {
            if (i >= 5) func_profiler.start(), comm_profiler.start(), memory_profiler.start();
            Share inputs(size);
            Share outputs(size);
            sigmoid(inputs, outputs);
            cudaDeviceSynchronize();
            if (i >= 5) func_profiler.accumulate("sigmoid for mizar"), comm_profiler.accumulate("sigmoid for mizar");
        }

        std::cout << "n: " << size << std::endl;
        printf("sigmoid for mizar (N=%d) - %f sec.\n", size, func_profiler.get_elapsed("sigmoid for mizar") / 1000.0 / EXP_TIMES);
        printf("tx comm: %f, rx comm: %f \n", 
            comm_profiler.get_comm_tx_bytes() / 1024. / 1024. / EXP_TIMES,
            comm_profiler.get_comm_rx_bytes() / 1024. / 1024. / EXP_TIMES);
        printf("memory: %f MB\n", memory_profiler.get_max_mem_mb() / EXP_TIMES);
    }
}

TYPED_TEST(BMMizarFunc, Maxpool) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    std::array<size_t, 2> exp_size{1ull << 10, 1ull << 20};
    // std::array<size_t, 1> exp_size{1ull << 10};
    int maxpool_window_size = 4;

    for (auto size: exp_size) {
        func_profiler.clear(), comm_profiler.clear(), memory_profiler.clear();

        for (int i = 0; i < EXP_TIMES + 5; i++) {
            if (i >= 5) func_profiler.start(), comm_profiler.start(), memory_profiler.start();
            Share inputs(size);
            Share outputs(size / maxpool_window_size);
            Share doutputs(size);
            maxpool(inputs, outputs, doutputs, maxpool_window_size);
            CUDA_CHECK(cudaDeviceSynchronize());
            if (i >= 5) func_profiler.accumulate("mp for mizar"), comm_profiler.accumulate("mp for mizar");
        }

        std::cout << "n: " << size << ", maxpool window size: " << maxpool_window_size << std::endl;
        // piranha have use func_profiler to measure the time of components of maxpool,
        // which is in conflict with ours.
        printf("mp for mizar (N=%d) - %f sec.\n", size, comm_profiler.get_elapsed("mp for mizar") / 1000.0 / EXP_TIMES);
        printf("tx comm: %f, rx comm: %f \n", 
            comm_profiler.get_comm_tx_bytes() / 1024. / 1024. / EXP_TIMES,
            comm_profiler.get_comm_rx_bytes() / 1024. / 1024. / EXP_TIMES);
        printf("memory: %f MB\n", memory_profiler.get_max_mem_mb() / EXP_TIMES);
    }
}
#endif