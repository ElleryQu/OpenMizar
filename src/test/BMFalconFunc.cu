
#include "unitTests.h"

#include <cutlass/conv/convolution.h>

#if defined(PFALCON)
template<typename T>
struct BMFalconFunc : public testing::Test {
    using ParamType = T;
    using DataType = typename T::share_type;

    static void SetUpTestSuite() {
        std::cout << "Benchamrk for P-Falcon, including dReLU and activation functions." << std::endl;
        std::cout << "Ring bitwidth: " << sizeof(DataType) << std::endl;
    }
};

using Types = testing::Types<
                    RSS<uint32_t>,
                    RSS<uint64_t>>;
TYPED_TEST_CASE(BMFalconFunc, Types);

TYPED_TEST(BMFalconFunc, DReLU) {

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
            RSS<uint8_t> outputs(size);
            dReLU(inputs, outputs);
            cudaDeviceSynchronize();
            if (i >= 5) func_profiler.accumulate("drelu for falcon"), comm_profiler.accumulate("drelu for falcon");
        }

        auto drelu_time = func_profiler.get_elapsed("drelu for falcon");    // ms
        auto tx_bytes = comm_profiler.get_comm_tx_bytes();                  // B
        auto rx_bytes = comm_profiler.get_comm_rx_bytes();                  // B
        std::cout << "n: " << size << std::endl;
        printf("drelu for falcon (N=%d) - %f sec.\n", size, drelu_time / 1000.0 / EXP_TIMES);
        printf("tx comm: %f, rx comm: %f\n", 
            comm_profiler.get_comm_tx_bytes() / 1024. / 1024. / EXP_TIMES,
            comm_profiler.get_comm_rx_bytes() / 1024. / 1024. / EXP_TIMES);
        printf("memory: %f MB\n", memory_profiler.get_max_mem_mb() / EXP_TIMES);
    }
}

TYPED_TEST(BMFalconFunc, ReLU) {

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
            RSS<uint8_t> doutputs(size);
            ReLU(inputs, outputs, doutputs);
            cudaDeviceSynchronize();
            if (i >= 5) func_profiler.accumulate("relu for falcon"), comm_profiler.accumulate("relu for falcon");
        }

        std::cout << "n: " << size << std::endl;
        printf("relu for falcon (N=%d) - %f sec.\n", size, func_profiler.get_elapsed("relu for falcon") / 1000.0 / EXP_TIMES);
        printf("tx comm: %f, rx comm: %f \n", 
            comm_profiler.get_comm_tx_bytes() / 1024. / 1024. / EXP_TIMES,
            comm_profiler.get_comm_rx_bytes() / 1024. / 1024. / EXP_TIMES);
        printf("memory: %f MB\n", memory_profiler.get_max_mem_mb() / EXP_TIMES);
    }
}

TYPED_TEST(BMFalconFunc, GeLU) {

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
            if (i >= 5) func_profiler.accumulate("gelu for falcon"), comm_profiler.accumulate("gelu for falcon");
        }

        std::cout << "n: " << size << std::endl;
        printf("gelu for falcon (N=%d) - %f sec.\n", size, func_profiler.get_elapsed("gelu for falcon") / 1000.0 / EXP_TIMES);
        printf("tx comm: %f, rx comm: %f \n", 
            comm_profiler.get_comm_tx_bytes() / 1024. / 1024. / EXP_TIMES,
            comm_profiler.get_comm_rx_bytes() / 1024. / 1024. / EXP_TIMES);
        printf("memory: %f MB\n", memory_profiler.get_max_mem_mb() / EXP_TIMES);
    }
}

TYPED_TEST(BMFalconFunc, Sigmoid) {

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
            if (i >= 5) func_profiler.accumulate("sigmoid for falcon"), comm_profiler.accumulate("sigmoid for falcon");
        }

        std::cout << "n: " << size << std::endl;
        printf("sigmoid for falcon (N=%d) - %f sec.\n", size, func_profiler.get_elapsed("sigmoid for falcon") / 1000.0 / EXP_TIMES);
        printf("tx comm: %f, rx comm: %f \n", 
            comm_profiler.get_comm_tx_bytes() / 1024. / 1024. / EXP_TIMES,
            comm_profiler.get_comm_rx_bytes() / 1024. / 1024. / EXP_TIMES);
        printf("memory: %f MB\n", memory_profiler.get_max_mem_mb() / EXP_TIMES);
    }
}

TYPED_TEST(BMFalconFunc, Maxpool) {

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
            RSS<uint8_t> doutputs(size);
            maxpool(inputs, outputs, doutputs, maxpool_window_size);
            CUDA_CHECK(cudaDeviceSynchronize());
            if (i >= 5) func_profiler.accumulate("mp for falcon"), memory_profiler.accumulate("mp for falcon");
        }

        std::cout << "n: " << size << ", maxpool window size: " << maxpool_window_size << std::endl;
        printf("mp for falcon (N=%d) - %f sec.\n", size, memory_profiler.get_elapsed("mp for falcon") / 1000.0 / EXP_TIMES);
        printf("tx comm: %f, rx comm: %f \n", 
            comm_profiler.get_comm_tx_bytes() / 1024. / 1024. / EXP_TIMES,
            comm_profiler.get_comm_rx_bytes() / 1024. / 1024. / EXP_TIMES);
        printf("memory: %f MB\n", memory_profiler.get_max_mem_mb() / EXP_TIMES);
    }
}
#endif
