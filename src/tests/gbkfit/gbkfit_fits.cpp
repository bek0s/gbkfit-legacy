
#include <gtest/gtest.h>

#include "gbkfit/fits.hpp"
#include "gbkfit/ndarray_host.hpp"

TEST(FitsTest, FooTestFits)
{
    const std::string filename = "foo.fits";
    const std::string key = "key1";

    //gbkfit::fits::set_key<float>(filename, key, 1 , "foo");

    /*
    gbkfit::fits::set_key_value<std::string>(filename, key, "hello");

    std::string out;
    gbkfit::fits::get_key_value<std::string>(filename, key, out);
    */

    std::string keyword1_name = "key1";
    std::string keyword1_value = "value1";
    std::string keyword1_comment = "comment1";

    gbkfit::fits::Header header;

    EXPECT_NO_THROW(header.add_keyword(keyword1_name));
    EXPECT_TRUE(header.has_keyword(keyword1_name));
    EXPECT_NO_THROW(header.get_keyword(keyword1_name));
    EXPECT_THROW(header.add_keyword(keyword1_name), std::runtime_error);

    EXPECT_NO_THROW(header.del_keyword(keyword1_name));
    EXPECT_FALSE(header.has_keyword(keyword1_name));
    EXPECT_THROW(header.get_keyword(keyword1_name), std::runtime_error);
    EXPECT_THROW(header.del_keyword(keyword1_name), std::runtime_error);

    gbkfit::fits::Header::Keyword& keyword = header.add_keyword(keyword1_name);

    std::string value;
    std::string comment;
    EXPECT_NO_THROW(keyword.get<std::string>(value, comment));
    EXPECT_TRUE(value == "");
    EXPECT_TRUE(comment == "");

    EXPECT_NO_THROW(keyword.set<std::string>(keyword1_value, keyword1_comment));


    //EXPECT_NO_THROW


    /*
    header.has_keyword(keyword_name);

    header.del_keyword(keyword_name);

    header.has_keyword(keyword_name);

    header.add_keyword(keyword_name);
    */



}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
