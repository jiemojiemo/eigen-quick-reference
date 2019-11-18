/*
 * Eigen 100
 *
 * This is quick reference of Eigen.
 *
 * We use Eigen to implement 100 Numpy (https://github.com/rougier/numpy-100)
 *
 * It is highly recommended to read Eigen document before starting if you never read it
 *
 */

#include <Eigen/Core>
#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace std;
using namespace Eigen;

void exercise_2()
{
    // 2. Print the eigen version
    cout << "Eigen version " << EIGEN_MAJOR_VERSION << "."
                             << EIGEN_MINOR_VERSION << endl;
}

void exercise_3()
{
    // 3 Create a null vector of size 10 (★☆☆)
    VectorXf Z = VectorXf::Zero(10);
    cout << Z << endl;
}

void exercise_4()
{
    // 4. How to find the memory size of any array (★☆☆)
    MatrixXf Z = MatrixXf::Zero(10, 10);
    cout << Z.size() * sizeof(MatrixXf::Scalar) << endl;
}

void exercise_6()
{
    // 6. Create a null vector of size 10 but the fifth value which is 1 (★☆☆)
    VectorXf Z = VectorXf::Zero(10);
    Z(4) = 1;

    cout << Z << endl;
}

void exercise_7()
{
    // 7. Create a vector with values ranging from 10 to 49 (★☆☆)
    const int start = 10;
    const int end = 49;
    const int size = end - start + 1;

    VectorXf Z = VectorXf::LinSpaced(size, start, end);

    cout << Z << endl;
}

void exercise_8()
{
    // 8. Reverse a vector (first element becomes last) (★☆☆)
    VectorXf Z = VectorXf::LinSpaced(10, 1, 10);
    Z = Z.reverse().eval();
    cout << Z << endl;

    // Z = Z.reverse() is aliasing
    // you can use .eval() or inplace to solve this:
    //           Z = Z.reverse().eval()
    //           Z.reverseInPlace()
}

void exercise_9()
{
    //9. Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆)

    // Eigen does not expose convenient methods to take slices or to reshape a matrix yet.
    // Nonetheless, such features can easily be emulated using the Map class.
    VectorXf Z1 = VectorXf::LinSpaced(9, 1, 9);
    Map<MatrixXf> Z2(Z1.data(), 3, 3);
    cout << Z2 << endl;
}

void exercise_10()
{
    // 10. Find indices of non-zero elements from [1,2,0,0,4,0] (★☆☆)
    VectorXf Z(6);
    Z << 1,2,0,0,4,0;

    std::vector<Index> nz;
    for(Index i = 0; i < Z.size(); ++i)
    {
        if(Z(i))
        {
            nz.push_back(i);
        }
    }

    Map<Matrix<Index , 1, Dynamic>> nzz(nz.data(), nz.size());

    cout << nzz;
}

void exercise_11()
{
    // 11. Create a 3x3 identity matrix (★☆☆)
    MatrixXf Z = MatrixXf::Identity(3,3);
    cout << Z << endl;
}

void exercise_12()
{
    // 12. Create a 3x3x3 array with random values (★☆☆)

    // NOTE: Tensor is unsupported module in 3.3.7
    Tensor<float,3> T(3,3,3);
    T.setRandom();

    cout << T << endl;

    // 12.1 Create a 3x3 array with random values (★☆☆)
    MatrixXf Z = MatrixXf::Random(3,3);
    cout << Z << endl;
}

void exercise_13()
{
    // 13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆)
    MatrixXf Z = MatrixXf::Random(10, 10);

    cout << Z.maxCoeff() << "," << Z.minCoeff() << endl;
}

void exercise_14()
{
    // 14. Create a random vector of size 30 and find the mean value (★☆☆)

    VectorXf Z = VectorXf::Random(30);

    cout << Z.mean() << endl;
}

void exercise_15()
{
    // 15. Create a 2d array with 1 on the border and 0 inside (★☆☆)
    MatrixXf Z = MatrixXf::Zero(5, 5);
    VectorXf padding = VectorXf::Constant(5, -1);

    Z.topRows<1>() = padding;
    Z.bottomRows<1>() = padding;
    Z.leftCols<1>() = padding;
    Z.rightCols<1>() = padding;

    cout << Z << endl;
}

void exercise_16()
{
    // 16. How to add a border (filled with 0's) around an existing array? (★☆☆)
    MatrixXf Z = MatrixXf::Ones(5, 5);

    Z.conservativeResize(6,6);
    VectorXf padding = VectorXf::Zero(6);

    Z.topRows<1>() = padding;
    Z.bottomRows<1>() = padding;
    Z.leftCols<1>() = padding;
    Z.rightCols<1>() = padding;

    cout << Z << endl;
}

void exercise_18()
{
    // 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆)

    // Is difficult to implement in Eigen, but create a diagonal is easy
    VectorXf V(4);
    V << 1, 2, 3, 4;

    MatrixXf Z = V.asDiagonal();

    cout << Z << endl;
}

void exercise_21()
{
    // 21. Create a checkerboard 8x8 matrix using the tile function (★☆☆)
    MatrixXf Z(2,2);
    Z << 0,1,
         1,0;

    cout <<  Z.replicate(4, 4) << endl;
}

void exercise_22()
{
    // 22. Normalize a 5x5 random matrix (★☆☆)
    MatrixXf Z = MatrixXf::Random(5,5);
    float mean = Z.mean();
    float std = std::sqrt( (Z.array() - mean).square().sum() / (Z.size() - 1) );
    Z = (Z.array() - mean) / (std);

    cout << Z << endl;
}


void exercise_24()
{
    // 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆)
    MatrixXf A = MatrixXf::Ones(5,3);
    MatrixXf B = MatrixXf::Ones(3,2);

    MatrixXf Z = A * B;

    cout << Z << endl;
}
void exercise_25()
{
    // 25. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆)

    VectorXf Z = VectorXf::LinSpaced(11, 0, 10);
    Matrix<float, Dynamic, 1> B = (3 < Z.array() && Z.array() <= 8).cast<float>() * -1.0;
    Matrix<float, Dynamic, 1> C = B.array() + 1.0;

    cout << Z.array() * B.array() + Z.array() * C.array() << endl;
}

void exercise_30()
{
    // 30. How to find common values between two arrays? (★☆☆)
    std::mt19937 gen(0);
    std::uniform_int_distribution<int> dis(0, 10);

    // generate random int numbers in range [0, 10]
    auto func = [&](int x){return dis(gen);};
    VectorXi A = VectorXd::Zero(10).unaryExpr(func);
    VectorXi B = VectorXd::Zero(10).unaryExpr(func);

    std::set<int> commom_values_set;
    auto find_common_values = [&](int x){
        if( (B.array() == x).any() )
        {
            commom_values_set.insert(x);
        }
        return x;
    };

    A = A.unaryExpr(find_common_values);

    for(const auto& v : commom_values_set)
    {
        cout << v << " ";
    }
}

void exercise_39()
{
    // 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆)¶
    VectorXf Z = VectorXf::LinSpaced(12, 0, 1);
    Z = Z.segment(1, 10);
    cout << Z << endl;
}

void exercise_40()
{
    // 40. Create a random vector of size 10 and sort it (★★☆)
   VectorXf Z = VectorXf::Random(10);

   sort(Z.data(), Z.data()+Z.size(), [](float x, float y){return x < y;});

   cout << Z << endl;
}

void exercise_40_1()
{
    // 40_1. Create a random matrix of size 10x10 and sort it row by row (★★☆)
    MatrixXf Z = MatrixXf::Random(10, 10);

    auto sort_func = [](float x, float y){return x<y;};

    std::vector<MatrixXf::Scalar> data(Z.cols());
    for(int i = 0; i < Z.rows(); ++i)
    {
        // copy row to data array
        for(int j = 0; j < Z.cols(); ++j)
        {
            data[j] = Z(i, j);
        }

        // sort data array
        sort(data.begin(), data.end(), sort_func);

        // copy back to row
        for(int j = 0; j < Z.cols(); ++j)
        {
            Z(i, j) = data[j];
        }
    }

    cout << Z << endl;
}

void exercise_40_2()
{
    // 40_2. Create a random matrix of size 10x10 and sort it col by col (★★☆)
    MatrixXf Z = MatrixXf::Random(10, 10);

    auto sort_func = [](float x, float y){return x<y;};

    std::vector<MatrixXf::Scalar> data(Z.rows());
    for(int i = 0; i < Z.cols(); ++i)
    {
        // copy row to data array
        for(int j = 0; j < Z.rows(); ++j)
        {
            data[j] = Z(j, i);
        }

        // sort data array
        sort(data.begin(), data.end(), sort_func);

        // copy back to row
        for(int j = 0; j < Z.rows(); ++j)
        {
            Z(j, i) = data[j];
        }
    }

    cout << Z << endl;

}


void exercise_42()
{
    // 42. Consider two random array A and B, check if they are equal (★★☆)
    MatrixXf A = MatrixXf::Random(5,5);
    MatrixXf B = MatrixXf::Random(5,5);

    bool equal = (A.array() == B.array()).all();

    cout << equal << endl;
}

void exercise_44()
{
    // 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆)
    MatrixXf Z = MatrixXf::Random(10, 2);
    VectorXf X = Z.col(0);
    VectorXf Y = Z.col(1);

    VectorXf R = (X.array().square() + Y.array().square()).sqrt();
    VectorXf T = (Y.array()/X.array()).atan();

    cout << "R:\n" << R << endl;
    cout << "T:\n" << T << endl;
}

void exercise_45()
{
    // 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆)
    VectorXf Z = VectorXf::Random(10);
    VectorXf::Index max_index;

    Z.maxCoeff(&max_index);

    Z(max_index) = 0.0;

    cout << Z << endl;
}

void exercise_47()
{
    // 47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj))
    VectorXf X = VectorXf::LinSpaced(8, 0, 7);
    VectorXf Y = X.array() + 0.5;

    MatrixXf C(X.size(), Y.size());

    for(int i = 0; i < C.cols(); ++i)
    {
        C.col(i) = 1.0/(X.array() - Y(i));
    }

    cout << C << endl;
}

void exercise_50()
{
    // 50. How to find the closest value (to a given scalar) in a vector? (★★☆)
    VectorXf Z = VectorXf::LinSpaced(100, 0, 99);
    float v = 10;
    VectorXf::Index index;

    (Z.array() - v).abs().minCoeff(&index);

    cout << index << endl;
}

void exercise_51()
{
    //    52. Consider a random vector with shape (10,2) representing coordinates, find point by point distances (★★☆)
    MatrixXf Z = MatrixXf::Random(10, 2);
    Matrix<float, Dynamic, 1> X = Z.col(0);
    Matrix<float, 1, Dynamic> Y = Z.col(1);

    MatrixXf XX = X.rowwise().replicate(10);
    MatrixXf YY = Y.colwise().replicate(10);

    MatrixXf D = (XX - XX.transpose()).array().square() + (YY - YY.transpose()).array().square();

    cout << D.cwiseSqrt() << endl; // D.cwiseSqrt() = D.array().sqrt()
}


void exercises()
{
//    exercise_2();
//    exercise_3();
//    exercise_4();
//    exercise_6();
//    exercise_7();
//    exercise_8();
//    exercise_9();
//    exercise_10();
//    exercise_11();
//    exercise_12();
//    exercise_13();
//    exercise_14();
//    exercise_15();
//    exercise_16();
//    exercise_18();
//    exercise_21();
//    exercise_22();
//    exercise_24();
//    exercise_25();
//    exercise_30();
//    exercise_39();
//    exercise_40();
//    exercise_40_1()
//    exercise_40_2();
//    exercise_42();
//    exercise_44();
//    exercise_45();
//    exercise_47();
//    exercise_50();
    exercise_51();
}

int main()
{
    exercises();
    return 0;
}