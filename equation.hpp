#pragma once

#include "EigenSupport.h"
#include "complex.hpp"

template <typename T>
using complex = std::complex<T>;

//1次方程式を解く関数
//coefs: 係数(a1〜a2)
//x: 解(出力)
template <typename T>
void Solve1st(T coefs[], complex<T> x[]) {
	T a = coefs[0];
	T b = coefs[1];

	if (a == T(0.0)) {
		throw std::invalid_argument("This equation has infinite solutions.");
	}
	x[0] = -b / a;
}

//2次方程式を解く関数
//coefs: 係数(a1〜a3)
//x: 解(出力)
template <typename T>
void Solve2nd(T coefs[], complex<T> x[]) {
	T a = coefs[0];
	T b = coefs[1];
	T c = coefs[2];

	if (a == T(0.0)) {
		Solve1st(&coefs[1], x);
		return;
	}

	x[0] = (-b + std::sqrtc(complex<T>(b * b - T(4.0) * a * c))) / (T(2.0) * a);
	x[1] = (-b - std::sqrtc(complex<T>(b * b - T(4.0) * a * c))) / (T(2.0) * a);
}

//3次方程式を解く関数
//coefs: 係数(a1〜a4)
//x: 解(出力)
template <typename T>
void Solve3rd(T coefs[], complex<T> x[]) {
	T a = coefs[0];
	T b = coefs[1];
	T c = coefs[2];
	T d = coefs[3];

	if (a == T(0.0)) {
		Solve2nd(&coefs[1], x);
		return;
	}

	T A0 = d / a;
	T A1 = c / a;
	T A2 = b / a;
	T p = A1 - A2 * A2 / T(3.0);
	T q = A0 - A1 * A2 / T(3.0) + A2 * A2 * A2 * T(2.0 / 27.0);

	//還元不能の場合
	//還元可能でもrをcomplexで扱えるが精度が気になるのでこうしている
	if (q * q / T(4.0) + pow(p / T(3.0), T(3.0)) < T(0)) {
		complex<T> r = std::sqrtc(complex<T>(q * q / T(4.0) + pow(p / T(3.0), T(3.0))));
		complex<T> L = std::powc(-q / T(2.0) + r, T(1.0 / 3.0));
		complex<T> R = std::powc(-q / T(2.0) - r, T(1.0 / 3.0));
		complex<T> o(T(-0.5), T(sqrt(3.0) / 2.0));

		x[0] = L + R - A2 / T(3.0);
		x[1] = o * o * L + o * R - A2 / T(3.0);
		x[2] = o * L + o * o * R - A2 / T(3.0);
		return;
	}

	T r = sqrt(pow(q / T(2.0), T(2.0)) + pow(p / T(3.0), T(3.0)));
	complex<T> L = cbrt(-q / T(2.0) + r);
	complex<T> R = cbrt(-q / T(2.0) - r);
	complex<T> o(T(-0.5), T(sqrt(3.0)) / T(2.0));

	x[0] = L + R - A2 / T(3.0);
	x[1] = o * o * L + o * R - A2 / T(3.0);
	x[2] = o * L + o * o * R - A2 / T(3.0);
}

//4次方程式を解く関数
//coefs: 係数(a1〜a5)
//x: 解(出力)
template <typename T>
void Solve4th(T coefs[], complex<T> x[]) {
	T a = coefs[0];
	T b = coefs[1];
	T c = coefs[2];
	T d = coefs[3];
	T e = coefs[4];

	if (a == T(0.0)) {
		Solve3rd(&coefs[1], x);
		return;
	}

	T A0 = e / a;
	T A1 = d / a;
	T A2 = c / a;
	T A3 = b / a;
	T p = A2 - T(6.0) * A3 * A3 / T(16);
	T q = A1 - A2 * A3 / T(2.0) + T(8.0) * pow(A3 / T(4.0), T(3.0));
	T r = A0 - A1 * A3 / T(4.0) + A2 * A3 * A3 / T(16) - T(3.0) * pow(A3 / T(4.0), T(4.0));

	if (q == T(0.0)) {
		T coef2[] = {
			T(1.0), p, r,
		};
		complex<T> t[2];
		Solve2nd(coef2, t);
		x[0] = std::sqrtc(t[0]) - A3 / T(4.0);
		x[1] = std::sqrtc(t[1]) - A3 / T(4.0);
		x[2] = -std::sqrtc(t[0]) - A3 / T(4.0);
		x[3] = -std::sqrtc(t[1]) - A3 / T(4.0);
		return;
	}

	T coef2[] = {
		T(1.0), T(2.0) * p, p * p - T(4.0) * r, -q * q,
	};
	complex<T> u[3];
	Solve3rd(coef2, u);

	T t(0.0);
	T minImag(DBL_MAX);
	for (int i = 0; i < 3; i++) {
		if (abs(u[i].imag()) < minImag && u[i].real() > T(0.0)) {
			t = u[i].real();
			minImag = abs(u[i].imag());
		}
	}

	T coef3_1[] = {
		T(1.0), sqrt(t), (p + t) / T(2.0) - sqrt(t) * q / (T(2.0) * t),
	};
	T coef3_2[] = {
		T(1.0), -sqrt(t), (p + t) / T(2.0) + sqrt(t) * q / (T(2.0) * t),
	};
	complex<T> y[4];
	Solve2nd(coef3_1, y);
	Solve2nd(coef3_2, &y[2]);

	for (int i = 0; i < 4; i++) {
		x[i] = y[i] - A3 / T(4.0);
	}
}

//n次方程式を同伴行列を用いて解く関数(遅い)
//coefs: 係数
//x: 解(出力)
template <typename T>
void Solventh(T coefs[], complex<T> x[], int n) {
	Eigen::MatrixXt<T> comp(n, n);
	comp *= T(0);
	for (int i = 0; i < n; i++) {
		comp(0, i) = -coefs[i + 1] / coefs[0];
		if (i > 0) {
			comp(i, i - 1) = T(1);
		}
	}
	Eigen::EigenSolver<Eigen::MatrixXt<T>> eig(comp);
	for (int i = 0; i < n; i++) {
		x[i] = eig.eigenvalues()[i];
	}
}
