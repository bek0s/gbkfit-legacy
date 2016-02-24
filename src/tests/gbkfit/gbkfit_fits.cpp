
#include <gtest/gtest.h>

#include "gbkfit/fits.hpp"
#include "gbkfit/ndarray_host.hpp"

TEST(FitsTest, FooTestFits)
{
    const std::string filename = "foo.fits";
    const std::string key = "key1";

    //gbkfit::fits::set_key<float>(filename, key, 1 , "foo");

    gbkfit::fits::set_key_value<std::string>(filename, key, "hello");

    std::string out;
    gbkfit::fits::get_key_value<std::string>(filename, key, out);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
