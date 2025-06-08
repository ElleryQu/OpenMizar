// // Funcnaility test.

// #include "unitTests.h"

// #include "../mpc/Mizar.h"

// #define TEST_PROT(PROT)                                     \
//     dReLUFrom##PROT(input, result);                         \
//     reconstruct(result, result_dd);                         \
//     printDeviceData(result_dd, "actual for " #PROT, false); \
//     assertDeviceData(result_dd, expected, false);

// template<typename T>
// struct FuncTest : public testing::Test {
//     using ParamType = T;

//     void SetUp() override {
//         std::cout << "Function tesst for Mizar, Mizar and MizarOpt." << std::endl;
//     }
// };

// using Types = testing::Types<
//                     Mizar<uint32_t>,
//                     Mizar<uint64_t>>;

// TYPED_TEST_CASE(FuncTest, Types);

// TYPED_TEST(FuncTest, Reconstruct) {

//     using Share = typename TestFixture::ParamType;
//     using T = typename Share::share_type;

//     if (partyNum >= Share::numParties) return;

//     Share a = {1, 8, 0.01, 12.2, -1, -0.01, -12.2}; 
//     DeviceData<T> result(7);

//     func_profiler.clear();
//     func_profiler.start();
//     reconstruct(a, result);

//     std::vector<double> expected = {1, 8, 0.01, 12.2, -1, -0.01, -12.2};
//     assertDeviceData(result, expected);
// }

// // TYPED_TEST(FuncTest, Mult) {

// //     using Share = typename TestFixture::ParamType;
// //     using T = typename Share::share_type;

// //     if (partyNum >= Share::numParties) return;

// //     Share a ({12, 24, 3, 5, -2, -3}, false); 
// //     Share b ({1, 0, 11, 3, -1, 11}, false);

// //     DeviceData<T> result(a.size());

// //     func_profiler.clear();
// //     func_profiler.start();
// //     b *= a;
// //     reconstruct(b, result);

// //     std::vector<double> expected = {12, 0, 33, 15, 2, -33};
// //     assertDeviceData(result, expected, false);
// // }

// TYPED_TEST(FuncTest, DRELU_T1) {

//     using Share = typename TestFixture::ParamType;
//     using T = typename Share::share_type;

//     if (partyNum >= Share::numParties) return;

//     // Share input = {
//     //     -1, -1, -1, -1, -1, -1, -1, -1,
//     //     -1, -1, -1, -1, -1, -1, -1, -1,
//     //      1,  1,  1,  1,  1,  1,  1,  1,
//     //      1,  1,  1,  1,  1,  1,  1,  1,
//     // };
//     Share input = {
//         -1, 1,
//     };


//     // std::vector<double> expected = {
//     //     0, 0, 0, 0, 0, 0, 0, 0,
//     //     0, 0, 0, 0, 0, 0, 0, 0,
//     //     1, 1, 1, 1, 1, 1, 1, 1,
//     //     1, 1, 1, 1, 1, 1, 1, 1,
//     // };
//     std::vector<double> expected = {
//         0, 1,
//     };

//     Share result(input.size());
//     DeviceData<T> result_dd(result.size());

//     TEST_PROT(Mizar);
//     TEST_PROT(Mizar);
//     TEST_PROT(MizarOpt);
// }

// TYPED_TEST(FuncTest, DRELU_T2) {

//     using Share = typename TestFixture::ParamType;
//     using T = typename Share::share_type;

//     if (partyNum >= Share::numParties) return;

//     Share input = {
//         -0.001922, 0.001187, 0.461417, -0.221444, 39.846086, -16.100100, 0.000000, 0.000000,
//         -0.001922, 0.001187, 0.461417, -0.221444, 39.846086, -16.100100, 0.000000, 0.000000,
//         -0.001922, 0.001187, 0.461417, -0.221444, 39.846086, -16.100100, 0.000000, 0.000000
//     };
//     std::vector<double> expected = {
//         0, 1, 1, 0, 1, 0, 1, 1,
//         0, 1, 1, 0, 1, 0, 1, 1,
//         0, 1, 1, 0, 1, 0, 1, 1
//     };
//     Share result(input.size());
//     DeviceData<T> result_dd(result.size());

//     TEST_PROT(Mizar);
//     TEST_PROT(Mizar);
//     TEST_PROT(MizarOpt);
// }

// #undef TEST_PROT