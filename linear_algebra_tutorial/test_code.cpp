// on templates https://stackoverflow.com/questions/44669412/use-of-template-class-t-in-c-when-declaring-classes-and-functions
#include <iostream>
#include <string>
#include <math.h>
#include <chrono>
#include "qb_matrix2.h"

using namespace std;

// A simple function to print a matrix to stdout.
template <class T>
void print_matrix_2(qb_matrix2<T> matrix)
{

	int n_rows = matrix.get_num_rows();
	int n_cols = matrix.get_num_cols();

	for (int row = 0; row < n_rows; row++) {
		
		for (int col = 0; col < n_cols; col++) {

			cout << matrix.get_element(row, col) << "  ";

		}
		cout << endl << endl;
	}
	

}

int main()
{
	double simple_data[12] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11, 12};

	qb_matrix2<double> my_first_matrix(3, 4, simple_data);
	print_matrix_2(my_first_matrix);

	// Testing my implementation of determinant calculation
	// -------------------------------------------------------------

	double det_2x2_test_data[4] = { 1.0, 2.0, 3.0, 4.0 };
	qb_matrix2<double> det_2x2_test_matrix(2, 2, det_2x2_test_data);

	double det;
	det = my_first_matrix.det_2x2();
	cout << "my_first_matrix det: " << det << endl;

	det = det_2x2_test_matrix.det_2x2();
	cout << "det_2x2_test_matrix det: " << det << endl;

	cout << " ----------------- testing 3x3 det -----------------\n";
	det = det_2x2_test_matrix.det_3x3();
	cout << "det_2x2_test_matrix det " << det << endl;

	double det_3x3_test_data[9] = {1.0, 2.0, -3.0, 4.0, 5.0, -6.0, 7.0, -8.0, 9.0};
	qb_matrix2<double> det_3x3_test_matrix(3, 3, det_3x3_test_data);

	det = det_3x3_test_matrix.det_3x3();
	cout << "det_3x3_test_matrix det " << det << endl << endl;

    print_matrix_2(det_3x3_test_matrix);

	qb_matrix2<double> test_inverse = det_3x3_test_matrix.calculate_inverse();

	cout << "Testing ! ! ! ! \n";
    print_matrix_2(det_3x3_test_matrix);
	print_matrix_2(test_inverse);


	cout << "tutorial inverse implementation\n";
	det_3x3_test_matrix.tutorial_inverse();
	print_matrix_2(det_3x3_test_matrix);

	// Extra section to learn how to optimise in c++
	cout << "speed testing a 1000 x 1000 matrix\n";

	// Time my implementation
	int N = 500;
	int ELE = N * N;

	std::vector<double> large_test_data;
	large_test_data.resize(ELE);
	for (int i = 0; i < ELE; ++i)
		large_test_data[i] = std::rand();

	qb_matrix2<double> my_large_matrix(N, N, large_test_data);

	auto start = std::chrono::high_resolution_clock::now();
	my_large_matrix.calculate_inverse();
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);

	cout << "time taken my implementation: " << duration.count() << std::endl;

	// Re-init a matrix (to combat caching) and try with their implementation TODO: major DRY
	std::vector<double> large_test_data2;
	large_test_data2.resize(ELE);
	for (int i = 0; i < ELE; ++i)
		large_test_data2[i] = std::rand();

	qb_matrix2<double> my_large_matrix2(N, N, large_test_data2);
	start = std::chrono::high_resolution_clock::now();
	my_large_matrix2.tutorial_inverse();
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
	
	cout << "time taken tutorial implementation: " << duration.count() << std::endl;

	// print_matrix_2(my_large_matrix);

	// my implementation - 10 s
	// their implementatino - 868 s (?)
	// MATLAB implementation - 0.04 s   :D those BLAS / LAPACK libraries *drool* (+ smarter approach)



	/*
	// Test element retrieval.
	cout << endl << "*****************************************" << endl;
	cout << "Element (0, 0) = " << my_first_matrix.get_element(0, 0) << endl;
	cout << "Element (1, 0) = " << my_first_matrix.get_element(1, 0) << endl;
	cout << "Element (2, 0) = " << my_first_matrix.get_element(2, 0) << endl;
	cout << "Element (0, 1) = " << my_first_matrix.get_element(0, 1) << endl;
	cout << "Element (1, 1) = " << my_first_matrix.get_element(1, 1) << endl;
	cout << "Element (2, 1) = " << my_first_matrix.get_element(2, 1) << endl;
	cout << "Element (5, 5) = " << my_first_matrix.get_element(5, 5) << endl;

	// Test Matrix Multiplication
	cout << endl << "****************************************" << endl;
	cout << "Test matrix multiplication." << endl;
	double simple_data_2[12] = { 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0};

	qb_matrix2<double> test_matrix(4, 3, simple_data_2);
	cout << "4 x 3 matrix (test_matrix) " << endl;
	print_matrix_2(test_matrix);

	cout << "Multiplication (my_first_matrix * test_matrix result): " << endl;
	qb_matrix2<double> mult_test_1 = my_first_matrix * test_matrix;
	print_matrix_2(mult_test_1);

	cout << "Multiplication (test_matrix_result * my_first_matrix): " << endl;
	print_matrix_2(test_matrix * my_first_matrix);

	cout << endl << "*******************************************" << endl;
	cout << "Test multiplication of column vector by matrix. " << endl;
	double column_data[3] = { 1.5, 2.5, 3.5 };
	double square_data[9] = { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 };
	qb_matrix2<double> test_column(3, 1, column_data);
	qb_matrix2<double> test_square(3, 3, square_data);

	cout << "Column Vector: " << endl;
	print_matrix_2(test_column);

	cout << "Square Matrix: " << endl;
	print_matrix_2(test_square);

	cout << "Coloumn Vector x Square Matrix: " << endl;
	print_matrix_2(test_column * test_square);

	cout << "Square Matrix x Column Vecetor" << endl;
	print_matrix_2(test_square * test_column);

	cout << "Square matrix + 1.0" << endl;
	qb_matrix2<double> square_matrix2 = test_square + 1.0;
	print_matrix_2(square_matrix2);

	cout << "(Square matrix + 1.0) * Column Vector = " << endl;
	print_matrix_2(square_matrix2 * test_column);

	// Test Equality Operator
	 cout << endl << "**********************************************" << endl;
	 cout << "Test equality operator. " << endl;
	 cout << "test_matrix == test_matrix_2 " << (my_first_matrix == test_matrix) << endl;
	 cout << "test_matrix_2 == test_matrix " << (test_matrix == my_first_matrix) << endl;
	 cout << "(Let test_matrix_3 = my_first_matrix) " << endl;
	 qb_matrix2<double> test_matrix_3(my_first_matrix);  // this should be equivilent to test_matrix_3 = my_first_matrix;
	 cout << "my_first_matrix == test_matrix_3 " << (my_first_matrix == test_matrix_3) << endl;
	 cout << "test_matrix_3 == my_first_matrix " << (test_matrix_3 == my_first_matrix) << endl;


	// Test matrix addition
	cout << endl << "***********************************************" << endl;
	cout << "my_first_matrix + 2.0 = " << endl;
	print_matrix_2(my_first_matrix + 2.0);
	cout << endl;
	cout << "2.0 + my_first_matrix" << endl;
	print_matrix_2(2.0 + my_first_matrix);

	// Test matrix subtraction by scaler
	cout << endl;
	cout << "test matrix subtraction by scalar" << endl;
	cout << "my_first_matrix - 2.0 = " << endl;
	print_matrix_2(my_first_matrix - 2.0);
	cout << endl;

	cout << "2.0 - my_first_matrix = " << endl;
	print_matrix_2(2.0 - my_first_matrix);

	// Test matrix multiplication by scalar
	cout << endl << "**************************************" << endl;
	cout << "Test multiplciation by scalar" << endl;
	cout << "my_first_matrix - 2.0 = " << endl;
	print_matrix_2(my_first_matrix * 2.0);
	cout << endl;

	cout << "2.0 * my_first_matrix = " << endl;
	print_matrix_2(2.0 * my_first_matrix);

	*/



	return 0;

}




