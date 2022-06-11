#pragma once
#ifndef QBMATRIX2_H
#define QBMATRIX2_H
#include <iostream>
// https://www.youtube.com/watch?v=jmo_HN_-PxI

// lesson 2
#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>

// & is used to reference a lot, see https://stackoverflow.com/questions/1943276/what-does-do-in-a-c-declaration for explaination

template <class T>  // Because we want this class to take any input data type, we need to define it all in the header file. 
class qb_matrix2    // T is the input type. 
{
public:
	
	// Constructers
	qb_matrix2();												// default constructer that accepts no input parameters
	qb_matrix2(int n_rows, int n_cols);                         // a constructer than takes only 2 input arguments, the number of rows and columns
	qb_matrix2(int n_rows, int n_cols, const T *input_data);    // take num row and column input argument, but also a pointer to a linear array (type T) we will fill the matrix with/
																// const keyword makes input data a constant (won't be changed)
	qb_matrix2(const qb_matrix2<T>& input_matrix);              // take a pre-made matrix of qb_matrix2 class (type T) as input 
	qb_matrix2(int n_rows, int n_cols, const std::vector<T> input_data);

	// Destructor (free up any memory we allocate)
	~qb_matrix2();

	// Configuration methods.
	bool resize(int n_rows, int n_cols);

	// Element access methods
	T get_element(int row, int col);
	bool set_element(int row, int col, T element_value);
	int get_num_rows();
	int get_num_cols();

	// calculator methods
	double det_2x2();
	double calculate_det_2x2(double a, double d, double b, double c);
	double det_3x3();

	// Custom
	void print_matrix(int n_cols, int n_rows, const qb_matrix2<T>& qb_matrix2);
	void print_matrix_2(qb_matrix2<T> matrix);

	// Now we overload the standard operators so they can also handle matrix operations

	// Lesson 2 - Gauss Jordan Elimination
	qb_matrix2<T> calculate_inverse();
	bool tutorial_inverse();
	bool join_colwise(const qb_matrix2<T>& matrix2);
	bool separate(qb_matrix2<T>* matrix1, qb_matrix2<T>* matrix2, int col_num);
	void set_to_identity();
	bool row_has_only_nonzero_from_another_col(int row_of_interest, int col_of_interest, int left_cols, int left_rows);

	// Overload == operator
	bool operator== (const qb_matrix2<T>& rhs);  // takes as input a qb_matrix2 of type T and a reference to it (which we call right-hand-side (of the operator))

	// Overload +, 0 and * operators (as 'friends'). So this is classes we define here, with 'template' keyword. The template variable is called 'U' here and it is also a type U!? ...?
	// They are friends because they need to have access to private functions.  'Necessary to create a new template variable'
	// They don't actually belong to the class!
	
	template <class U> friend qb_matrix2<U> operator+ (const qb_matrix2<U>& lhs, const qb_matrix2<U>& rhs);  // add matrix to a matrix 
	template <class U> friend qb_matrix2<U> operator+ (const U& lhs, const qb_matrix2<U>& rhs);              // add scalar (of type U) to a matrix
	template <class U> friend qb_matrix2<U> operator+ (const qb_matrix2<U>& lhs, const U& rhs);				 // add matrix to a scalar (of type U)

	template <class U> friend qb_matrix2<U> operator- (const qb_matrix2<U>& lhs, const qb_matrix2<U>& rhs);
	template <class U> friend qb_matrix2<U> operator- (const U& lhs, const qb_matrix2<U>& rhs);
	template <class U> friend qb_matrix2<U> operator- (const qb_matrix2<U>& lhs, const U& rhs);

	template <class U> friend qb_matrix2<U> operator* (const qb_matrix2<U>& lhs, const qb_matrix2<U>& rhs);
	template <class U> friend qb_matrix2<U> operator* (const U& lhs, const qb_matrix2<U>& rhs);
	template <class U> friend qb_matrix2<U> operator* (const qb_matrix2<U>& lhs, const U& rhs);    // note these are all copies of eachother, the onlything changed is the operator

	
	int sub_2_ind(int row, int col, int n_cols) const; // https://stackoverflow.com/questions/2157458/using-const-in-classs-functions

// private:
	int sub_2_ind(int row, int col);

// Private! Lesson 2
	bool is_square();
	bool compare(const qb_matrix2<T>& matrix1, double tolerance);
	bool close_enough(T f1, T f2);
	void swap_row(int i, int j);
	void mult_row_add(int row_added, int row_to_add, T mult_factor);
	void mult_row(int row, T mult_factor);
	int find_row_with_max_element(int col_number, int starting_row);
	

private:
	T *m_matrix_data;                      // pointer to array for matrix data
	int m_n_rows, m_n_cols, m_nelements;   // define private class member variables

};

// ====================================================================================================================================================================================
// Constructer / Destructor Functions 
// ====================================================================================================================================================================================
// Note: because of the typing explained above, all functions are defined in the header file.

// The Default Constructer
template <class T>
qb_matrix2<T>::qb_matrix2()
{
	m_n_rows = 1;
	m_n_cols = 1;
	m_nelements = 1;
	m_matrix_data = new T[m_nelements];
	m_matrix_data[0] = 0.0;

}


// Construct empty matrix
template <class T>
qb_matrix2<T>::qb_matrix2(int n_rows, int n_cols)  // take int number of cols and rows, generate an empty matrix of zeros
{
	m_n_rows = n_rows;
	m_n_cols = n_cols;
	m_nelements = n_rows * n_cols;
	m_matrix_data = new T[m_nelements];

	for (int i = 0; i < m_nelements; ++i) {
		m_matrix_data[i] - 0.0;
	}
}

// Construct from const linear array
template <class T>
qb_matrix2<T>::qb_matrix2(int n_rows, int n_cols, const T *input_data)  // same as above but now fill with the passed input_data (assuming inputdata is the right size, linear array
{
	m_n_rows = n_rows;
	m_n_cols = n_cols;
	m_nelements = n_rows * n_cols;
	m_matrix_data = new T[m_nelements];

	for (int i = 0; i < m_nelements; ++i) {
		m_matrix_data[i] = input_data[i];
	}
}

// The copy constructer (copy the data from input matrix)
template <class T>
qb_matrix2<T>::qb_matrix2(const qb_matrix2<T>& input_matrix)
{
	m_n_rows = input_matrix.m_n_rows;
	m_n_cols = input_matrix.m_n_cols;
	m_nelements = input_matrix.m_nelements;
	m_matrix_data = new T[m_nelements];

	for (int i = 0; i < m_nelements; ++i)
	{
		m_matrix_data[i] = input_matrix.m_matrix_data[i];
	};

}

template <typename T>
qb_matrix2<T>::qb_matrix2(int n_rows, int n_cols, const std::vector<T> input_data)
{
	m_n_rows = n_rows;
	m_n_cols = n_cols;
	m_nelements = n_rows * n_cols;

	m_matrix_data = new T[m_nelements];
	for (int i = 0; i < m_nelements; ++i) {
		m_matrix_data[i] = input_data[i];
	}
}

// Destructor!! Required because we use the new keyword! 
template <class T>
qb_matrix2<T>::~qb_matrix2()
{
	if (m_matrix_data != nullptr)
	{
		delete[] m_matrix_data;
	}
};


// ====================================================================================================================================================================================
// Configuration Functions
// ====================================================================================================================================================================================

template <class T>
bool qb_matrix2<T>::resize(int n_rows, int n_cols)  // dont copy elements (gets complicated e.g. what elements to keep? just init again as zero).
{
	m_n_rows = n_rows;
	m_n_cols = n_cols;
	m_nelements = n_rows * n_cols;
	
	delete[] m_matrix_data;  

	m_matrix_data = new T[m_nelements];
	
	if (m_matrix_data != nullptr)  // TODO: what is this check!? 
	{
		for (int i = 0; i < m_nelements; ++i) {
			m_matrix_data[i] = 0.0;
		};

		return true;
		
	}
	else
	{
		return false;
	}


}

// ====================================================================================================================================================================================
// Element Functions
// ====================================================================================================================================================================================

template <class T>
T qb_matrix2<T>::get_element(int row, int col)
{
	int linear_index = sub_2_ind(row, col);
//	std::cout << row << "  " << col << "  " << linear_index << std::endl;
	if (linear_index >= 0)
	{
		return m_matrix_data[linear_index];
	}
	else 
	{
		return 0;
	}
}

template <class T>
bool qb_matrix2<T>::set_element(int row, int col, T element_value)
{

	int linear_index = sub_2_ind(row, col);
	if (linear_index >= 0)
	{
		m_matrix_data[linear_index] = element_value;
		return true;
	}
	else
	{
		return false;
	}
}

template <class T>
int qb_matrix2<T>::get_num_rows()
{
	return m_n_rows;
}

template <class T>
int qb_matrix2<T>::get_num_cols()
{
	return m_n_cols;
}

// ====================================================================================================================================================================================
// Lesson 2 - Matrix Inverse
// ====================================================================================================================================================================================


// My go at calculating the determinant of 2x2 and 3x3 matrix (used for calcualting the inverse). 
// Pretty simple but to scale to 4x4+ would be a pain.
// There must be a better way! 
// ---------------------------------------------------------------------------------------------------------------

template <class T>
double qb_matrix2<T>::det_2x2()
{
	if ((m_n_rows != 2) || (m_n_cols != 2)) {  // TODO: can create helper function
		std::cout << "det_2x2: this matrix is not a 2x2 matrix \n";
		return std::nan("");
		}

	double det;
	det = calculate_det_2x2(m_matrix_data[0], m_matrix_data[3], m_matrix_data[1], m_matrix_data[2]);

	return det;
}

template <class T>
double qb_matrix2<T>::calculate_det_2x2(double a, double d, double b, double c)
{
	double det;

	det = (a * d) - (b * c);

	return det;
}

template <typename T>  // typename is same as class!! 
double qb_matrix2<T>::det_3x3()
{
	if ((m_n_rows != 3) || (m_n_cols != 3)) {
		std::cout << "det_3x3: this matrix is not a 3x3 matrix \n";  // TODO: can create helper function
		return std::nan("");
	}

	double a, b, c;
	double a_minor_det, b_minor_det, c_minor_det;

	a = m_matrix_data[0];
	b = m_matrix_data[1];
	c = m_matrix_data[2];

	a_minor_det = calculate_det_2x2(m_matrix_data[4], m_matrix_data[8], m_matrix_data[5], m_matrix_data[7]);
	b_minor_det = calculate_det_2x2(m_matrix_data[3], m_matrix_data[8], m_matrix_data[5], m_matrix_data[6]);
	c_minor_det = calculate_det_2x2(m_matrix_data[3], m_matrix_data[7], m_matrix_data[4], m_matrix_data[6]);

	return a * a_minor_det - b * b_minor_det + c * c_minor_det;

}

// ====================================================================================================================================================================================
// Lession 2 implementation
// ====================================================================================================================================================================================
/*
// Notes: https://www.youtube.com/watch?v=wOlG_fnd3v8
   Gaussian elimination - row echelon form - can find Z directly and solve the linear system by substitution
   Gauss-Jordan elimination - reduced row echelon form - can read the coefficients directly as x,y,z coefficients are indentity
   So a very clever method for finding the inverse!

	So here we add elemental functions for gauss-jordan elimination, the first three the classic operations
	for Gauss-Jordan and the thers helper functions:
		- swap two rows in the matrix
		- multiply a row by a given number
		- muliply a row by a given number and add the result to another row

		- find the row with the maximum element at the given column
		- join two matricies together columnwise
		- separate a matrix in two around the specified column


*/

// Have implemented all functions as described in the video. Now to have a go at implementing Gauss-Jordian elimination myself!!
// So we first need to concatenate both matricies column-wise, then make the left-hand matrix the indentity matrix.
// Then the right hand matrix will be the inverse. Beautiful!
template <typename T>
bool qb_matrix2<T>::row_has_only_nonzero_from_another_col(int row_of_interest, int col_of_interest, int left_cols, int left_rows)
//
// For use with an augmented matrix during Gauss-Jordan elimination. Check the row (candidate for swapping with another row such
// that all rows have a non-zero value in the correct (i.e. diagonal) position) does not also have the only non-zero
// value for another column. In this case, swapping it would just create another problem later.
{

	int col_check = 0;

	for (int col = 0; col < left_cols; ++col)
	{
		if ((col == col_of_interest) || (m_matrix_data[sub_2_ind(row_of_interest, col)] == 0))  // TODO check equality Ignore col we are looking to swap for, and cols where our row is zero
			continue;

		for (int row = 0; row < left_rows; ++row)
		{
			if (row == row_of_interest)  // we are looking at rows in the column that are not our row of interest
				continue;

			if (m_matrix_data[sub_2_ind(row, col)] != 0)  // if we find one non-zero instance in the column from another row, it is safe to swap this row.
				break;									  // we want this to be the case for all columns. If this is not, we will return false underneath. 

			return false;

		}
	}
	return true;
}


template <typename T>
qb_matrix2<T> qb_matrix2<T>::calculate_inverse()
// First my own implementation, before seeing the tutorials. Works well when compared to
// the tutorial, implementation is slightly different in that it returns the inverse matrix
// rather than set it as the class matrix.
// 
// TODO: this way is soem waste, because we loop through the matrix twice in total
// ----------------------------------------------------------------------------------------
{
	if (m_n_rows != m_n_cols)
		throw std::invalid_argument("Matrix is not square\n");

	int left_rows = m_n_rows;
	int left_cols = m_n_cols;
	int left_n_elements = m_nelements;

	// concatenate identitify matrix columnwise to right edge. After putting 
	// origional matrix to RREF, this matrix will be the inverse

	T* orig_matrix = new T[m_nelements];                                            // store the origional m_matrix_data on the heap at a new pointer 
	std::memcpy(orig_matrix, m_matrix_data, sizeof(m_matrix_data) * m_nelements);   

	qb_matrix2<T> identity_matrix = qb_matrix2(m_n_rows, m_n_cols, m_matrix_data);
	identity_matrix.set_to_identity();

	join_colwise(identity_matrix);
	
	// swap rows so that correct cols have a digit in
	// iter 100 times trying to swap around the rows 
	int num_tries = 100;
	int other_row_idx;
	T other_row_value;
	int mult_sign;
	T mult_factor;

	for (int iter = 0; iter < num_tries; ++iter) {

		for (int col = 0; col < left_cols; ++col)
		{
			for (int row = 0; row < left_rows; ++row)
			{  // for every column, if the appropriate row is not non-zero swap with a row that
			   // has a non-zero digit then move to end of iter to try again. 
				if ((row == col) && (m_matrix_data[sub_2_ind(row, col)] == 0)) {  // TODO: use difference method

					other_row_idx = find_row_with_max_element(col, 0);
					swap_row(other_row_idx, row);
					goto end;
				}
			}
		}
	end:
		;
	}

	// Now for each column, first put the diagonal to one, then work down the column top-to-bottom
	// zeroing the off-diagonals
	for (int col = 0; col < left_cols; ++col)
	{
		if (m_matrix_data[sub_2_ind(col, col)] != 1)
			mult_row(col, 1 / m_matrix_data[sub_2_ind(col, col)]);

		for (int row = 0; row < left_rows; ++row)
		{
			if ((row != col) && (m_matrix_data[sub_2_ind(row, col)] != 0)) {

				T value = m_matrix_data[sub_2_ind(row, col)];

				other_row_value = m_matrix_data[sub_2_ind(col, col)];  

				if (signbit(value) == signbit(other_row_value))  // see tutorial implementation for better handling of this mult factor
					mult_sign = -1;
				else
					mult_sign = 1;

				mult_factor = fabs(value) * mult_sign; // because cell we are multing will always be 1 from previous step

				mult_row_add(row, col, mult_factor);
			}
		}
	}

	// now peel off the inverse, return as a qb_matrix2 and put the origional matrix back! 
	
	// T* inverse_data = new T[left_rows * left_cols];
	std::vector<T> inverse_data;
	inverse_data.resize(left_n_elements);

	for (int row = 0; row < m_n_rows; ++row)
	{
		for (int col = 0; col < m_n_cols; ++col)
		{
			if (col >= left_cols)
					inverse_data[sub_2_ind(row, col - left_cols, left_cols)] = m_matrix_data[sub_2_ind(row, col)];


		}
	}
	qb_matrix2<T> matrix_inverse(left_rows, left_cols, inverse_data);
	
	delete[] m_matrix_data;  // release the joined matrix data, below point back to the origional data 
	
	m_matrix_data = orig_matrix;  // Don't need to delete orig_matrix as now pointed to by m_matrix_data 
	m_n_rows = left_rows;
	m_n_cols = left_cols;
	m_nelements = left_n_elements;
	
	return matrix_inverse;

}

	
template <typename T>
bool qb_matrix2<T>::tutorial_inverse()
// The implementation from the tutorial
{
	if (!is_square())
		throw std::invalid_argument("Cannot compute the invers of a matrix that is not sqare.");

	// join with indentity matrix on the right 

	qb_matrix2<T> identity_matrix(m_n_rows, m_n_cols);
	identity_matrix.set_to_identity();

	int original_num_cols = m_n_cols;
	join_colwise(identity_matrix);

	int c_row, c_col;
	int max_count = 100;
	int count = 0;
	bool complete_flag = false;

	while ((!complete_flag) && (count < max_count))
	{
		// Go through the diagonals, sorting the col then row for that diagonal.
		// first, set the diagonal to one
		for (int diag_index = 0; diag_index < m_n_rows; ++diag_index)
		{
			c_row = diag_index;
			c_col = diag_index;

			int max_index = find_row_with_max_element(c_col, c_row);

			if (max_index != c_row)
			{
				swap_row(c_row, max_index);
			}
			if (m_matrix_data[sub_2_ind(c_row, c_col)] != 1.0)
			{
				T mult_factor = 1.0 / m_matrix_data[sub_2_ind(c_row, c_col)];
				mult_row(c_row, mult_factor);
			}


			// consider the row, and zero all off-diagonals
			for (int row_index = c_row + 1; row_index < m_n_rows; ++row_index)
			{
				if (!close_enough(m_matrix_data[sub_2_ind(row_index, c_col)], 0.0))
				{
					int row_one_index = c_col;

					T current_element_value = m_matrix_data[sub_2_ind(row_index, c_col)];

					T row_one_value = m_matrix_data[sub_2_ind(row_one_index, c_col)];

					if (!close_enough(row_one_value, 0.0))
					{

						T correction_factor = -(current_element_value / row_one_value);  // nice!

						mult_row_add(row_index, row_one_index, correction_factor);

					}
				}
			}


			// consider the col, and zero all off-diagonals
			for (int col_index = c_col + 1; col_index < original_num_cols; ++col_index)
			{
				if (!close_enough(m_matrix_data[sub_2_ind(c_row, col_index)], 0.0))
				{
					int row_one_index = col_index;

					T current_element_value = m_matrix_data[sub_2_ind(c_row, col_index)];

					T row_one_value = m_matrix_data[sub_2_ind(row_one_index, col_index)];

					if (!close_enough(row_one_value, 0.0))
					{
						T correction_factor = -(current_element_value / row_one_value);

						mult_row_add(c_row, row_one_index, correction_factor);

					}
				}
			}
		}


		// split the matrix in half, check left is identity and
		// set matrix to its inverse
		qb_matrix2<T> left_half;
		qb_matrix2<T> right_half;

		this->separate(&left_half, &right_half, original_num_cols);

		if (left_half == identity_matrix)
		{
			complete_flag = true;

			m_n_cols = original_num_cols;
			m_nelements = m_n_rows * m_n_cols;

			delete[] m_matrix_data;
			m_matrix_data = new T[m_nelements];

			for (int i = 0; i < m_nelements; ++i)
				m_matrix_data[i] = right_half.m_matrix_data[i];
		}
		
	count++;
	}
	return complete_flag;
}


template <typename T>
void qb_matrix2<T>::set_to_identity()
{
	if (!is_square())
		throw std::invalid_argument("Matrix is not square \n");

	for (int row = 0; row < m_n_rows; row++) {
		for (int col = 0; col < m_n_cols; col++) {

			if (col == row)
				m_matrix_data[sub_2_ind(row, col)] = 1.0;
			else
				m_matrix_data[sub_2_ind(row, col)] = 0.0;
			
		}
	}
}

template <typename T>
bool qb_matrix2<T>::separate(qb_matrix2<T>* matrix1, qb_matrix2<T>* matrix2, int sep_col)  // the output is returned into the two qb_matrix2<T> pointers in the input arguments
{
	int n_rows = m_n_rows;
	int n_cols1 = sep_col;
	int n_cols2 = m_n_cols - sep_col;

	matrix1->resize(n_rows, n_cols1);
	matrix2->resize(n_rows, n_cols2);

	for (int row = 0; row < m_n_rows; ++row) {
		for (int col = 0; col < m_n_cols; ++col) {

			if (col < sep_col)
				matrix1->set_element(row, col, this->get_element(row, col));
			else
				matrix2->set_element(row, col - sep_col, this->get_element(row, col));

		}
	}
	return true;
}


template <typename T>
bool qb_matrix2<T>::join_colwise(const qb_matrix2<T>& matrix2) {

	// Extract the information we need from both matricies
	int n_rows1 = m_n_rows;
	int n_rows2 = matrix2.m_n_rows;
	int n_cols1 = m_n_cols;
	int n_cols2 = matrix2.m_n_cols;
	int new_num_cols = n_cols1 + n_cols2;
	// Cannot separate unless rows are even
	if (n_rows1 != n_rows2)
		throw std::invalid_argument("Attempt to join matricies with different numbers of rows is invalid.");

	// Allocate memory for the result
	// Note that only the number of columns increases.
	T* new_matrix_data = new T[n_rows1 * new_num_cols];

	// Copy the two matricies into the new one
	int linear_index, result_linear_index;

	for (int i = 0; i < n_rows1; ++i) {
		for (int j = 0; j < new_num_cols; j++) {

			result_linear_index = (i * new_num_cols) + j;  // TODO: this should be in own package

			// if j is in the left hand matrix we get it from there
			if (j < n_cols1) {
				linear_index = (i * n_cols1) + j;
				new_matrix_data[result_linear_index] = m_matrix_data[linear_index];
			}

			// otherwise, we get it from the right-hand matrix
			else {
				linear_index = (i * n_cols2) + (j - n_cols1);
				new_matrix_data[result_linear_index] = matrix2.m_matrix_data[linear_index];
			}
		}
	}

	// Update the stored data - note I dont think we can just re-assign the matrix
	// without deleting because the old matrix will loose reference and be memory leak. 
	m_n_cols = new_num_cols;
	m_nelements = m_n_rows * new_num_cols;

	delete[] m_matrix_data;

	m_matrix_data = new T[m_nelements];
	for (int i = 0; i < m_nelements; ++i)
		m_matrix_data[i] = new_matrix_data[i];

	delete[] new_matrix_data;

	return true;

}

template <typename T>
bool qb_matrix2<T>::is_square()
{
	if (m_n_rows == m_n_cols)
		return true;
	else
		return false;
}


template <typename T>
void qb_matrix2<T>::swap_row(int row_1, int row_2)
{
	// store a temporary copy of row 1 
	T* temp_row = new T[m_n_cols];
	for (int col = 0; col < m_n_cols; col++)
		temp_row[col] = m_matrix_data[sub_2_ind(row_1, col)];

	// Replace row 1 with row 2
	for (int col = 0; col < m_n_cols; col++)
		m_matrix_data[sub_2_ind(row_1, col)] = m_matrix_data[sub_2_ind(row_2, col)];

	// replace row 2 with temp row 1
	for (int col = 0; col < m_n_cols; col++)
		m_matrix_data[sub_2_ind(row_2, col)] = temp_row[col];

	delete[] temp_row;

}


template <typename T>
void qb_matrix2<T>::mult_row_add(int row_added, int row_to_add, T mult_factor) 
// Multiply a row then add to another
{
	for (int col = 0; col < m_n_cols; col++)
		m_matrix_data[sub_2_ind(row_added, col)] += (m_matrix_data[sub_2_ind(row_to_add, col)] * mult_factor);
}


template <typename T>
void qb_matrix2<T>::mult_row(int row, T mult_factor)
{
	for (int col = 0; col < m_n_cols; ++col)
		m_matrix_data[sub_2_ind(row, col)] *= mult_factor;
}



template <typename T>
int qb_matrix2<T>::find_row_with_max_element(int col_number, int starting_row)
{
	T temp_value = fabs(m_matrix_data[sub_2_ind(starting_row, col_number)]);  // we must read this from the matrix as do not know type initially
	T row_value;
	int row_with_max = starting_row;

	for (int row = starting_row + 1; row < m_n_rows; ++row) {

		row_value = std::fabs(m_matrix_data[sub_2_ind(row, col_number)]);

		if (row_value > temp_value) {
			temp_value = row_value;
			row_with_max = row;
		}

	}
	return row_with_max;
}






template <typename T>
bool qb_matrix2<T>::compare(const qb_matrix2<T>& matrix1, double tolerance)
//
// Compare the matricies by checking their element-wise sum of squared difference is within some tolerance
// (this method due to difficulties in element-wise equality checks).
// -------------------------------------------------------------------------
{
	// first check the matricies are not the same size
	int num_rows_1 = matrix1.m_n_rows;
	int num_cols_1 = matrix1.m_n_cols;

	if ((m_n_rows != num_rows_1) || (m_n_cols != num_cols_1))
		return false;

	double cumulative_sum = 0.0;
	for (int i = 0; i < m_nelements; ++i) {

		cumulative_sum += pow(m_matrix_data[i] - matrix1.m_matrix_data[i], 2);

	}

	double difference;
	difference = sqrt(cumulative_sum / (m_nelements - 1));

	if (difference < tolerance)
		return true;
	else;
		return false;


}


// ====================================================================================================================================================================================
// Operators 
// ====================================================================================================================================================================================


// Addition 

template <class T>
qb_matrix2<T> operator+ (const qb_matrix2<T>& lhs, const qb_matrix2<T>& rhs)  // matrix + matrix 
{
	int n_rows = lhs.m_n_rows;
	int n_cols = lhs.m_n_cols;
	int n_elements = lhs.m_nelements;

	T *temp_result = new T[n_elements];

	for (int i = 0; i < n_elements; ++i) {

		temp_result[i] = lhs.m_matrix_data[i] + rhs.m_matrix_data[i];
	}
	qb_matrix2<T> result(n_rows, n_cols, temp_result);
	delete[] temp_result;

	return result;

}

template <class T>
qb_matrix2<T> operator+ (const T& lhs, const qb_matrix2<T>& rhs)  // scalar + matrix 
{
	int n_rows = rhs.m_n_rows;
	int n_cols = rhs.m_n_cols;
	int n_elements = rhs.m_nelements;

	T *temp_result = new T[n_elements];

	for (int i = 0; i < n_elements; ++i) {
		temp_result[i] = lhs + rhs.m_matrix_data[i];
	}
	qb_matrix2<T> result(n_rows, n_cols, temp_result);
	
	delete[] temp_result;

	return result;

}


template <class T>
qb_matrix2<T> operator+ (const qb_matrix2<T>& lhs, const T& rhs)  // TODO: could check matricies are the same size (for all operators)
{
	int n_rows = lhs.m_n_rows;
	int n_cols = lhs.m_n_cols;
	int n_elements = lhs.m_nelements;

	T *temp_result = new T[n_elements];

	for (int i = 0; i < n_elements; ++i) {
		temp_result[i] = lhs.m_matrix_data[i] + rhs;
	}
	qb_matrix2<T> result(n_rows, n_cols, temp_result);

	delete[] temp_result;

	return result;

}

// Subtraction

template <class T>
qb_matrix2<T> operator- (const qb_matrix2<T>& lhs, const qb_matrix2<T>& rhs)  // matrix + matrix 
{
	int n_rows = lhs.m_n_rows;
	int n_cols = lhs.m_n_cols;
	int n_elements = lhs.m_nelements;

	T *temp_result = new T[n_elements];

	for (int i = 0; i < n_elements; ++i) {

		temp_result[i] = lhs.m_matrix_data[i] - rhs.m_matrix_data[i];
	}
	qb_matrix2<T> result(n_rows, n_cols, temp_result);
	delete[] temp_result;

	return result;

}

template <class T>
qb_matrix2<T> operator- (const T& lhs, const qb_matrix2<T>& rhs)  // scalar + matrix 
{
	int n_rows = rhs.m_n_rows;
	int n_cols = rhs.m_n_cols;
	int n_elements = rhs.m_nelements;

	T *temp_result = new T[n_elements];

	for (int i = 0; i < n_elements; ++i) {
		temp_result[i] = lhs - rhs.m_matrix_data[i];
	}
	qb_matrix2<T> result(n_rows, n_cols, temp_result);

	delete[] temp_result;

	return result;

}


template <class T>
qb_matrix2<T> operator- (const qb_matrix2<T>& lhs, const T& rhs)  // TODO: could check matricies are the same size (for all operators)
{
	int n_rows = lhs.m_n_rows;
	int n_cols = lhs.m_n_cols;
	int n_elements = lhs.m_nelements;

	T *temp_result = new T[n_elements];

	for (int i = 0; i < n_elements; ++i) {
		temp_result[i] = lhs.m_matrix_data[i] - rhs;
	}
	qb_matrix2<T> result(n_rows, n_cols, temp_result);

	delete[] temp_result;

	return result;

}

// Multiplication

template <class T>
qb_matrix2<T> operator* (const qb_matrix2<T>& lhs, const qb_matrix2<T>& rhs)  // matrix + matrix 
//
// Matrix multiplication for 2 x 2 matrix. Wrote as test so uses std::vector rather than double pointer.
//
// ---------------------------------------------------------------------------
{
	int l_nelements = lhs.m_n_rows * lhs.m_n_cols;
	int r_nelements = rhs.m_n_rows * rhs.m_n_cols;

	int new_n_rows = lhs.m_n_rows;
	int new_n_cols = rhs.m_n_cols;

	double element_sum = 0;
	int l_idx = 0;
	int r_idx = 0;
	int row_sum = 0;
	int new_idx = 0;
	int new_nelements = new_n_rows * new_n_cols;

	if (lhs.m_n_cols != rhs.m_n_rows) {
		std::cout << std::endl << "Matrix Multiplication not allowed" << std::endl;
		qb_matrix2<double> null_matrix(1, 1);
		return null_matrix;  // return an empty matrix on fail
	}

    T *new_matrix = new T[new_nelements];

	for (int n = 0; n < lhs.m_n_rows; n++) {  // cycle through the row

		row_sum = 0;
		for (int m = 0; m < rhs.m_n_cols; m++) {  // for each row, cycle through the columns for that row


			element_sum = 0;
			for (int e = 0; e < lhs.m_n_cols; e++) {

				l_idx = lhs.sub_2_ind(n, e, lhs.m_n_cols);
				r_idx = lhs.sub_2_ind(e, m, rhs.m_n_cols);

				element_sum += lhs.m_matrix_data[l_idx] * rhs.m_matrix_data[r_idx];

			}

			new_idx = lhs.sub_2_ind(n, m, new_n_cols);
			new_matrix[new_idx] = element_sum;
		}
	}

	qb_matrix2<T> result(new_n_rows, new_n_cols, new_matrix);
	
	delete[] new_matrix;

	return result;
}


template <class T>
qb_matrix2<T> operator* (const T& lhs, const qb_matrix2<T>& rhs)  // scalar + matrix 
{
	int n_rows = rhs.m_n_rows;
	int n_cols = rhs.m_n_cols;
	int n_elements = rhs.m_nelements;

	T *temp_result = new T[n_elements];

	for (int i = 0; i < n_elements; ++i) {
		temp_result[i] = lhs * rhs.m_matrix_data[i];
	}
	qb_matrix2<T> result(n_rows, n_cols, temp_result);

	delete[] temp_result;

	return result;

}


template <class T>
qb_matrix2<T> operator* (const qb_matrix2<T>& lhs, const T& rhs)  
{
	int n_rows = lhs.m_n_rows;
	int n_cols = lhs.m_n_cols;
	int n_elements = lhs.m_nelements;

	T *temp_result = new T[n_elements];

	for (int i = 0; i < n_elements; ++i) {
		temp_result[i] = lhs.m_matrix_data[i] * rhs;
	}
	qb_matrix2<T> result(n_rows, n_cols, temp_result);

	delete[] temp_result;

	return result;

}

// ====================================================================================================================================================================================
// The == Operator
// ====================================================================================================================================================================================


template <class T>
bool qb_matrix2<T>::operator== (const qb_matrix2<T>& rhs)
{
	// check if the matricies are the same size
	if ((this->m_n_rows != rhs.m_n_rows) || (this->m_n_cols != rhs.m_n_cols)) {  // he uses this->m_n_rows  ... 
		return false;
	}

	for (int i = 0; i < this->m_nelements; ++i) {

		if (!close_enough(this->m_matrix_data[i], rhs.m_matrix_data[i])) {
			return false;
		}
	}

	return true;

}


template <class T>
bool qb_matrix2<T>::close_enough(T q1, T q2)
{
	return fabs(q1 - q2) < 1e-9;
}



// ====================================================================================================================================================================================
// Private Functions
// ====================================================================================================================================================================================

template <class T>
int qb_matrix2<T>::sub_2_ind(int row, int col, int n_cols) const
{
	return (row * n_cols) + col;
}

template <class T>
int qb_matrix2<T>::sub_2_ind(int row, int col)
{
	if ((row < m_n_rows) && (row >= 0) && (col < m_n_cols) && (col >= 0)) {

		return sub_2_ind(row, col, m_n_cols);
	}
	else {
		return -1;
	}
}

// ====================================================================================================================================================================================
// Custom Functions
// ====================================================================================================================================================================================


template <class T>
void qb_matrix2<T>::print_matrix(int n_rows, int n_cols, const qb_matrix2<T>& qb_matrix) {  // TODO: try using const 
	//
	// Print a 2D matrix of any size.
	// ----------------------------------------------

	for (int n = 0; n < n_rows; n++) {

		std::cout << std::endl << std::endl;

		for (int m = 0; m < n_cols; m++) {

			int idx = sub_2_ind(n, m, n_cols);
			std::cout << " " << qb_matrix.m_matrix_data[idx] << " ";
		}
	}
}


#endif


