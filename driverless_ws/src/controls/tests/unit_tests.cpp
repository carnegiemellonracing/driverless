#include <gtest/gtest.h>

TEST(controls_lib, stub_test) {
    ASSERT_EQ(1, 1);
}

int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}