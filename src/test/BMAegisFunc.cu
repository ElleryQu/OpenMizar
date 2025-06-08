
// #include "unitTests.h"

// #include <cutlass/conv/convolution.h>

// #if defined(AEGIS)
// template<typename T>
// struct BMMizarFunc : public testing::Test {
//     using ParamType = T;
//     using DataType = typename T::share_type;

//     static void SetUpTestSuite() {
//         std::cout << "Benchamrk for Mizar, including dReLU and activation functions." << std::endl;
//         std::cout << "Ring bitwidth: " << sizeof(DataType) << std::endl;
//     }
// };

// using Types = testing::Types<
//                     Mizar<uint32_t>,
//                     Mizar<uint64_t>>;
// TYPED_TEST_CASE(BMMizarFunc, Types);

// TYPED_TEST(BMMizarFunc, DRELU) {

//     using Share = typename TestFixture::ParamType;
//     using T = typename Share::share_type;

//     if (partyNum >= Share::numParties) return;

//     std::array<size_t, 2> exp_size{1ull << 10, 1ull << 20};
//     // std::array<size_t, 1> exp_size{1ull << 10};

//     for (auto size: exp_size) {
//         func_profiler.clear(), comm_profiler.clear();

//         for (int i = 0; i < EXP_TIMES + 5; i++) {
//             if (i >= 5) func_profiler.start(), comm_profiler.start();
//             Share inputs(size);
//             Share outputs(size);
//             // dReLUFromMizar(inputs, outputs);
//             dReLUFromMizarOpt(inputs, outputs);
//             cudaDeviceSynchronize();
//             if (i >= 5) func_profiler.accumulate("drelu for aegis"), comm_profiler.accumulate("drelu for aegis");
//         }

//         std::cout << "n: " << size << std::endl;
//         printf("drelu from aegis (N=%d) - %f sec.\n", size, func_profiler.get_elapsed("drelu for aegis") / 1000.0 / EXP_TIMES);
//         printf("tx comm: %f, rx comm: %f \n", 
//             comm_profiler.get_comm_tx_bytes() / 1024. / 1024. / EXP_TIMES,
//             comm_profiler.get_comm_rx_bytes() / 1024. / 1024. / EXP_TIMES);

//         // {    
//         //     auto drelu_time = func_profiler.get_elapsed("drelu for aegis");    // ms
//         //     auto tx_bytes = comm_profiler.get_comm_tx_bytes();                  // B
//         //     auto rx_bytes = comm_profiler.get_comm_rx_bytes();                  // B
//         //     ::testing::Test::RecordProperty("n", exp_size);
//         //     ::testing::Test::RecordProperty("Time_ms", drelu_time);
//         //     ::testing::Test::RecordProperty("CommTX_bytes", tx_bytes);
//         //     ::testing::Test::RecordProperty("CommRX_bytes", rx_bytes);
//         // }
//     }
// }

// TYPED_TEST(BMMizarFunc, ReLU) {

//     using Share = typename TestFixture::ParamType;
//     using T = typename Share::share_type;

//     if (partyNum >= Share::numParties) return;

//     std::array<size_t, 2> exp_size{1ull << 10, 1ull << 20};
//     // std::array<size_t, 1> exp_size{1ull << 10};

//     for (auto size: exp_size) {
//         func_profiler.clear(), comm_profiler.clear();

//         for (int i = 0; i < EXP_TIMES + 5; i++) {
//             if (i >= 5) func_profiler.start(), comm_profiler.start();
//             Share inputs(size);
//             Share outputs(size);
//             // dReLUFromMizar(inputs, outputs);
//             ReLU(inputs, outputs);
//             cudaDeviceSynchronize();
//             if (i >= 5) func_profiler.accumulate("drelu for aegis"), comm_profiler.accumulate("drelu for aegis");
//         }

//         std::cout << "n: " << size << std::endl;
//         printf("drelu from aegis (N=%d) - %f sec.\n", size, func_profiler.get_elapsed("drelu for aegis") / 1000.0 / EXP_TIMES);
//         printf("tx comm: %f, rx comm: %f \n", 
//             comm_profiler.get_comm_tx_bytes() / 1024. / 1024. / EXP_TIMES,
//             comm_profiler.get_comm_rx_bytes() / 1024. / 1024. / EXP_TIMES);

//         // {    
//         //     auto drelu_time = func_profiler.get_elapsed("drelu for aegis");    // ms
//         //     auto tx_bytes = comm_profiler.get_comm_tx_bytes();                  // B
//         //     auto rx_bytes = comm_profiler.get_comm_rx_bytes();                  // B
//         //     ::testing::Test::RecordProperty("n", exp_size);
//         //     ::testing::Test::RecordProperty("Time_ms", drelu_time);
//         //     ::testing::Test::RecordProperty("CommTX_bytes", tx_bytes);
//         //     ::testing::Test::RecordProperty("CommRX_bytes", rx_bytes);
//         // }
//     }
// }

// TYPED_TEST(BMMizarFunc, GeLU) {

//     using Share = typename TestFixture::ParamType;
//     using T = typename Share::share_type;

//     if (partyNum >= Share::numParties) return;

//     std::array<size_t, 2> exp_size{1ull << 10, 1ull << 20};
//     // std::array<size_t, 1> exp_size{1ull << 10};

//     for (auto size: exp_size) {
//         func_profiler.clear(), comm_profiler.clear();

//         for (int i = 0; i < EXP_TIMES + 5; i++) {
//             if (i >= 5) func_profiler.start(), comm_profiler.start();
//             Share inputs(size);
//             Share outputs(size);
//             // dReLUFromMizar(inputs, outputs);
//             GeLU(inputs, outputs);
//             cudaDeviceSynchronize();
//             if (i >= 5) func_profiler.accumulate("drelu for aegis"), comm_profiler.accumulate("drelu for aegis");
//         }

//         std::cout << "n: " << size << std::endl;
//         printf("drelu from aegis (N=%d) - %f sec.\n", size, func_profiler.get_elapsed("drelu for aegis") / 1000.0 / EXP_TIMES);
//         printf("tx comm: %f, rx comm: %f \n", 
//             comm_profiler.get_comm_tx_bytes() / 1024. / 1024. / EXP_TIMES,
//             comm_profiler.get_comm_rx_bytes() / 1024. / 1024. / EXP_TIMES);

//         // {    
//         //     auto drelu_time = func_profiler.get_elapsed("drelu for aegis");    // ms
//         //     auto tx_bytes = comm_profiler.get_comm_tx_bytes();                  // B
//         //     auto rx_bytes = comm_profiler.get_comm_rx_bytes();                  // B
//         //     ::testing::Test::RecordProperty("n", exp_size);
//         //     ::testing::Test::RecordProperty("Time_ms", drelu_time);
//         //     ::testing::Test::RecordProperty("CommTX_bytes", tx_bytes);
//         //     ::testing::Test::RecordProperty("CommRX_bytes", rx_bytes);
//         // }
//     }
// }

// TYPED_TEST(BMMizarFunc, Maxpool) {

//     using Share = typename TestFixture::ParamType;
//     using T = typename Share::share_type;

//     if (partyNum >= Share::numParties) return;

//     std::array<size_t, 2> exp_size{1ull << 10, 1ull << 20};
//     // std::array<size_t, 1> exp_size{1ull << 10};

//     for (auto size: exp_size) {
//         func_profiler.clear(), comm_profiler.clear();

//         for (int i = 0; i < EXP_TIMES + 5; i++) {
//             if (i >= 5) func_profiler.start(), comm_profiler.start();
//             Share inputs(size);
//             Share outputs(size);
//             Share doutputs(size);
//             // dReLUFromMizar(inputs, outputs);
//             maxpool(inputs, outputs, doutputs, round_up_log_2(size));
//             cudaDeviceSynchronize();
//             if (i >= 5) func_profiler.accumulate("drelu for aegis"), comm_profiler.accumulate("drelu for aegis");
//         }

//         std::cout << "n: " << size << std::endl;
//         printf("drelu from aegis (N=%d) - %f sec.\n", size, func_profiler.get_elapsed("drelu for aegis") / 1000.0 / EXP_TIMES);
//         printf("tx comm: %f, rx comm: %f \n", 
//             comm_profiler.get_comm_tx_bytes() / 1024. / 1024. / EXP_TIMES,
//             comm_profiler.get_comm_rx_bytes() / 1024. / 1024. / EXP_TIMES);

//         // {    
//         //     auto drelu_time = func_profiler.get_elapsed("drelu for aegis");    // ms
//         //     auto tx_bytes = comm_profiler.get_comm_tx_bytes();                  // B
//         //     auto rx_bytes = comm_profiler.get_comm_rx_bytes();                  // B
//         //     ::testing::Test::RecordProperty("n", exp_size);
//         //     ::testing::Test::RecordProperty("Time_ms", drelu_time);
//         //     ::testing::Test::RecordProperty("CommTX_bytes", tx_bytes);
//         //     ::testing::Test::RecordProperty("CommRX_bytes", rx_bytes);
//         // }
//     }
// }
// #endif