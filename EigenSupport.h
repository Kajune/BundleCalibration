#pragma once
//#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS 
//#define _SILENCE_CXX17_NEGATORS_DEPRECATION_WARNING

#include <Eigen/Dense>

namespace Eigen {

#define EIGEN_MAKE_TYPEDEFS(TypeSuffix, Size, SizeSuffix)		  \
	template <typename Type>										  \
	using Matrix##SizeSuffix##TypeSuffix = Matrix<Type, Size, Size>;  \
	template <typename Type>										  \
	using Vector##SizeSuffix##TypeSuffix = Matrix<Type, Size, 1>;     \
	template <typename Type>										  \
	using RowVector##SizeSuffix##TypeSuffix = Matrix<Type, 1, Size>;

#define EIGEN_MAKE_FIXED_TYPEDEFS(TypeSuffix, Size)				  \
	template <typename Type>										  \
	using Matrix##Size##X##TypeSuffix = Matrix<Type, Size, Dynamic>;  \
	template <typename Type>										  \
	using Matrix##X##Size##TypeSuffix = Matrix<Type, Dynamic, Size>;

#define EIGEN_MAKE_TYPEDEFS_ALL_SIZES(TypeSuffix) \
	EIGEN_MAKE_TYPEDEFS(TypeSuffix, 2, 2) \
	EIGEN_MAKE_TYPEDEFS(TypeSuffix, 3, 3) \
	EIGEN_MAKE_TYPEDEFS(TypeSuffix, 4, 4) \
	EIGEN_MAKE_TYPEDEFS(TypeSuffix, Dynamic, X) \
	EIGEN_MAKE_FIXED_TYPEDEFS(TypeSuffix, 2) \
	EIGEN_MAKE_FIXED_TYPEDEFS(TypeSuffix, 3) \
	EIGEN_MAKE_FIXED_TYPEDEFS(TypeSuffix, 4)

	EIGEN_MAKE_TYPEDEFS_ALL_SIZES(t)

}
