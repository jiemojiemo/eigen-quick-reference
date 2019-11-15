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

void exercise_22()
{
    // 22. Normalize a 5x5 random matrix (★☆☆)
    MatrixXf Z = MatrixXf::Random(5,5);
    float mean = Z.mean();
    float std = std::sqrt( (Z.array() - mean).square().sum() / (Z.size() - 1) );
    Z = (Z.array() - mean) / (std);

    cout << Z << endl;
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
    exercise_22();
}

int main()
{
    exercises();
    return 0;
}