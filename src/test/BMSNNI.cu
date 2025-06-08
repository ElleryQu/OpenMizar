// #include "unitTests.h"

// template<typename T>
// struct BMSNNI : public testing::Test {
//     using ParamType = T;
//     using DataType = typename T::share_type;

//     static void SetUpTestSuite() {
//         std::cout << "Benchamrk for SNNI." << std::endl;
//         std::cout << "Ring bitwidth: " << sizeof(T) << std::endl;
//     }
// };

// using Types = testing::Types<
//                     Mizar<uint32_t>,
//                     Mizar<uint64_t>>;