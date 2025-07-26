#include "unittest-cpp/UnitTest++/UnitTest++.h"
#include "unittest-cpp/UnitTest++/TestReporterStdout.h"

#include "mathcv.h"

using namespace MathCV;

namespace {
    SUITE(MatrixTests) {
        TEST(ConstructorWithDimensions) {
            Matrix<> m(3, 4);
            CHECK_EQUAL(3, m.rows());
            CHECK_EQUAL(4, m.columns());
            CHECK_EQUAL(12, m.total());
        }

        TEST(CopyConstructor) {
            Matrix<> m1(2, 2);
            m1(0, 0) = 1; m1(0, 1) = 2;
            m1(1, 0) = 3; m1(1, 1) = 4;

            CHECK_EQUAL(1, m1(0,0));
            CHECK_EQUAL(2, m1(0,1));
            CHECK_EQUAL(3, m1(1,0));
            CHECK_EQUAL(4, m1(1, 1));
            
            Matrix<> m2(m1);
            CHECK_EQUAL(m1.rows(), m2.rows());
            CHECK_EQUAL(m1.columns(), m2.columns());
            CHECK_EQUAL(m1(0, 0), m2(0, 0));
            CHECK_EQUAL(m1(1, 1), m2(1, 1));
        }

        TEST(MoveConstructor) {
            Matrix<> m1(2, 2);
            m1(0, 0) = 5;
            
            Matrix<> m2(std::move(m1));
            CHECK_EQUAL(5, m2(0, 0));
            CHECK_EQUAL(0, m1.rows()); // Проверка, что исходная матрица очищена
        }

        TEST(AssignmentOperator) {
            Matrix<> m1(2, 2);
            m1(0, 0) = 1; m1(0, 1) = 2;
            m1(1, 0) = 3; m1(1, 1) = 4;
            
            Matrix<> m2(2, 2);
            m2 = m1;
            CHECK_EQUAL(m1.rows(), m2.rows());
            CHECK_EQUAL(m1(1, 0), m2(1, 0));
        }

        TEST(MoveAssignmentOperator) {
            Matrix<> m1(2, 2);
            m1(0, 1) = 7;
            
            Matrix<> m2(1, 2);
            m2 = std::move(m1);
            CHECK_EQUAL(7, m2(0, 1));
            CHECK_EQUAL(0, m1.columns());
        }

        TEST(ElementAccess) {
            Matrix<> m(2, 3);
            m(1, 2) = 3.14;
            CHECK_EQUAL(3.14, m(1, 2));
            
            const Matrix<> cm(2, 2);
            CHECK_EQUAL(0.0, cm(0, 0)); // Проверка const-версии
        }

        TEST(MatrixAddition) {
            Matrix<> m1(2, 2), m2(2, 2);
            m1(0, 0) = 1; m1(0, 1) = 2;
            m1(1, 0) = 3; m1(1, 1) = 4;
            
            m2(0, 0) = 4; m2(0, 1) = 3;
            m2(1, 0) = 2; m2(1, 1) = 1;
            
            auto result = m1 + m2;
            CHECK_EQUAL(5, result(0, 0));
            CHECK_EQUAL(5, result(1, 1));
        }

        TEST(MatrixSubtraction) {
            Matrix<> m1(2, 2), m2(2, 2);
            m1(0, 0) = 5; m1(0, 1) = 5;
            m1(1, 0) = 5; m1(1, 1) = 5;
            
            m2(0, 0) = 1; m2(0, 1) = 2;
            m2(1, 0) = 3; m2(1, 1) = 4;
            
            auto result = m1 - m2;
            CHECK_EQUAL(4, result(0, 0));
            CHECK_EQUAL(1, result(1, 1));
        }
        TEST(MatrixMultiplication) {
            Matrix<> m1(2, 3), m2(3, 2);
            m1(0, 0) = 1; m1(0, 1) = 2; m1(0, 2) = 3;
            m1(1, 0) = 4; m1(1, 1) = 5; m1(1, 2) = 6;
            
            m2(0, 0) = 7; m2(0, 1) = 8;
            m2(1, 0) = 9; m2(1, 1) = 10;
            m2(2, 0) = 11; m2(2, 1) = 12;
            
            auto result = m1 * m2;
            CHECK_EQUAL(58, result(0, 0));
            CHECK_EQUAL(64, result(0, 1));
            CHECK_EQUAL(139, result(1, 0));
            CHECK_EQUAL(154, result(1, 1));
        }
        TEST(ScalarMultiplication) {
            Matrix<> m(2, 2);
            m(0, 0) = 1; m(0, 1) = 2;
            m(1, 0) = 3; m(1, 1) = 4;
            
            auto result = m * 2.5;
            CHECK_CLOSE(2.5, result(0, 0), 1e-5);
            CHECK_CLOSE(10.0, result(1, 1), 1e-5);
        }
        TEST(ScalarDivision) {
            Matrix<> m(2, 2);
            m(0, 0) = 2; m(0, 1) = 4;
            m(1, 0) = 6; m(1, 1) = 8;
            
            auto result = m / 2.0;
            CHECK_EQUAL(1, result(0, 0));
            CHECK_EQUAL(4, result(1, 1));
        }
        TEST(Determinant) {
            Matrix<> m(3, 3);
            m(0, 0) = 1; m(0, 1) = 2; m(0, 2) = 3;
            m(1, 0) = 4; m(1, 1) = 5; m(1, 2) = 6;
            m(2, 0) = 7; m(2, 1) = 8; m(2, 2) = 9;
            
            CHECK_CLOSE(0, m.det(), 1e-15); // Вырожденная матрица
            
            Matrix<> m2(2, 2);
            m2(0, 0) = 1; m2(0, 1) = 2;
            m2(1, 0) = 3; m2(1, 1) = 4;
            CHECK_EQUAL(-2, m2.det());
        }
        TEST(Trace) {
            Matrix<> m(3, 3);
            m(0, 0) = 1; m(0, 1) = 2; m(0, 2) = 3;
            m(1, 0) = 4; m(1, 1) = 5; m(1, 2) = 6;
            m(2, 0) = 7; m(2, 1) = 8; m(2, 2) = 9;
            
            CHECK_EQUAL(15, m.tr());
        }

        TEST(EqualityOperators) {
            Matrix<> m1(2, 2), m2(2, 2), m3(2, 2);
            m1(0, 0) = 1; m1(0, 1) = 2; m1(1, 0) = 3; m1(1, 1) = 4;
            m2(0, 0) = 1; m2(0, 1) = 2; m2(1, 0) = 3; m2(1, 1) = 4;
            m3(0, 0) = 4; m3(0, 1) = 3; m3(1, 0) = 2; m3(1, 1) = 1;

            CHECK(m1 == m2);
            CHECK(m1 != m3);
        }

        TEST(CompoundAssignmentOperators) {
            Matrix<> m(2, 2);
            m(0, 0) = 1; m(0, 1) = 2;
            m(1, 0) = 3; m(1, 1) = 4;
            
            m *= 2;
            CHECK_EQUAL(2, m(0, 0));
            CHECK_EQUAL(8, m(1, 1));
            
            m /= 2;
            CHECK_EQUAL(1, m(0, 0));
            CHECK_EQUAL(4, m(1, 1));
        }

        TEST(StaticMethods) {
            Matrix<> m1(2, 2), m2(2, 2), result(2, 2);
            m1(0, 0) = 1; m1(0, 1) = 1;
            m1(1, 0) = 1; m1(1, 1) = 1;
            
            m2(0, 0) = 2; m2(0, 1) = 2;
            m2(1, 0) = 2; m2(1, 1) = 2;
            
            // Test static addition
            CHECK_EQUAL(true, Matrix<>::summarize(m1, m2, result) == Result::SUCCESS);
            CHECK_EQUAL(3, result(0, 0));
            CHECK_EQUAL(3, result(1, 1));
            
            // Test static subtraction
            CHECK_EQUAL(true, Matrix<>::substraction(m2, m1, result) == Result::SUCCESS);
            CHECK_EQUAL(1, result(0, 0));
            CHECK_EQUAL(1, result(1, 1));
        }
        TEST(TransposeOperation) {
            Matrix<> m(2, 3);
            m(0, 0) = 1; m(0, 1) = 2; m(0, 2) = 3;
            m(1, 0) = 4; m(1, 1) = 5; m(1, 2) = 6;
            
            Matrix<> result(3, 2);
            CHECK_EQUAL(true, Matrix<>::transpose(m, result) == Result::SUCCESS);
            CHECK_EQUAL(3, result.rows());
            CHECK_EQUAL(2, result.columns());
            CHECK_EQUAL(1, result(0, 0));
            CHECK_EQUAL(4, result(0, 1));
            CHECK_EQUAL(3, result(2, 0));
        }

        TEST(InverseMatrix) {
            Matrix<> m(2, 2);
            m(0, 0) = 4; m(0, 1) = 7;
            m(1, 0) = 2; m(1, 1) = 6;
            
            Matrix<> inv(2, 2);
            CHECK_EQUAL(true, Matrix<>::inverse(m, inv) == Result::SUCCESS);
            
            // Verify A*A⁻¹ = I
            Matrix<> identity(2, 2);
            CHECK_EQUAL(true, Matrix<>::multiply(m, inv, identity) == Result::SUCCESS);
            CHECK_CLOSE(1.0, identity(0, 0), 1e-9);
            CHECK_CLOSE(0.0, identity(0, 1), 1e-9);
            CHECK_CLOSE(0.0, identity(1, 0), 1e-9);
            CHECK_CLOSE(1.0, identity(1, 1), 1e-9);
        }
    }
}
