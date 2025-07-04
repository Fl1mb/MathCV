#include "unittest-cpp/UnitTest++/UnitTest++.h"

// Пример теста
TEST(SampleTest) {
    CHECK_EQUAL(true, true);
}

// Обязательная функция main() для UnitTest++
int main(int argc, char** argv) {
    return UnitTest::RunAllTests();
}