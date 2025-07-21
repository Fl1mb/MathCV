#include "unittest-cpp/UnitTest++/UnitTest++.h"
#include "unittest-cpp/UnitTest++/TestReporterStdout.h"

#include "mathcv.h"

using namespace MathCV;

namespace{
    SUITE(MathCVTests){
        // Пример теста
        TEST(SampleTest) {
            CHECK_EQUAL(true, true);
        }

    }
}
