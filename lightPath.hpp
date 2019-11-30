#pragma once

#include "ray.hpp"
#include "equation.hpp"
#include "EigenSupport.h"
#include "unsupported/Eigen/NonLinearOptimization"
#include "unsupported/Eigen/NumericalDiff"

template<typename T>
const T r_w = T(1.33);

template <typename T>
ray<T> backward(const Eigen::Vector2t<T>& pos2d, const plane<T>& pl, const camera& cam) {
	auto ray = cam.castRay(pos2d);
	ray = intersectPlane(ray, pl);
	ray.dir = refract(ray.dir, pl.getNormal(ray), r_w<T>);
	return ray;
}

template <typename T, typename U>
Eigen::Vector3t<U> depthTo3D(const Eigen::Vector2t<T>& pos2d, U d_inv, const plane<T>& pl, const camera& cam) {
	auto ray = backward(pos2d, pl, cam);
	return ray.origin + ray.dir / (ray.dir.z() * d_inv);
}

template <typename T>
Eigen::Vector3t<T> forward(const Eigen::Vector3t<T>& pos3d, const plane<T>& pl) {
	T d = pl.origin.norm();
	Eigen::Vector3t<T> n = -pl.normal.normalized();
	Eigen::Vector3t<T> POR = n.cross(pos3d).normalized();
	Eigen::Vector3t<T> Z1 = -n;
	Eigen::Vector3t<T> Z2 = POR.cross(Z1).normalized();

	T v = pos3d.transpose() * Z1;
	T u = pos3d.transpose() * Z2;

	std::vector<T> coefs;
	coefs.push_back((r_w<T> -T(1))*(r_w<T> +T(1)));
	coefs.push_back(T(-2)*u*(r_w<T> -T(1))*(r_w<T> +T(1)));
	coefs.push_back(d * d * r_w<T> * r_w<T> -d * d + T(2) * d*v + r_w<T> * r_w<T> * u * u - u * u - v * v);
	coefs.push_back(T(-2)*d * d * r_w<T> * r_w<T> * u);
	coefs.push_back(d * d * r_w<T> * r_w<T> * u *u);

	complex<T> x[4] = { T(0) };
	Solve4th(coefs.data(), x);
//	Solventh(coefs.data(), x, 4);

	for (int i = 0; i < 4 - 1; i++) {
		// 下から上に順番に比較します
		for (int j = 4 - 1; j > i; j--) {
			// 上の方が大きいときは互いに入れ替えます
			if (std::absc(x[j]) > std::absc(x[j - 1])) {
				std::swap(x[j], x[j - 1]);
			}
		}
	}

	Eigen::Vector2t<T> Normal = Eigen::Vector2t<T>(0, -1).normalized();
	T minError = T(DBL_MAX);
	T best{ 0 };
	for (int i = 0; i < 4; i++) {
		T xx = x[i].real();
		Eigen::Vector2t<T> vi(xx, d);
		T a = T(1) / r_w<T>;
		T bb = vi.transpose() * Normal;
		T bbb = vi.transpose() * vi;

		T b = T(-1) * bb - sqrt(bb * bb - (T(1) - r_w<T> * r_w<T>)*bbb);
		b = b / r_w<T>;

		Eigen::Vector2t<T> vr = a * vi + b * Normal;
		Eigen::Vector2t<T> vrd(u, v);
		vrd -= vi;

		T error = abs(vrd[0] * vr[1] - vrd[1] * vr[0]);
		if (error < T(0.0001)) {
			return xx * Z2 + d * Z1;
		}
		if (error < minError) {
			minError = error;
			best = xx;
		}
	}
	return best * Z2 + d * Z1;
}

template <typename T>
Eigen::Vector3t<T> iterativeForward(const Eigen::Vector3t<T>& pos3d, const plane<T>& pl) {
	if (abs(pl.normal.dot(pos3d)) < T(1e-10)) {
		return pos3d;
	}
	Eigen::Vector3t<T> dir = pl.normal.normalized().cross(pos3d).cross(pl.normal).normalized() * T(0.1);
	Eigen::Vector3t<T> pos = intersectPlane(ray<T>{pos3d, -pos3d.normalized()}, pl).origin;
	T error(0), oldError(0);
	error = sin(acos(pos.normalized().dot(pl.normal))) - sin(acos((pos3d - pos).normalized().dot(pl.normal))) * r_w<T>;
	do {
		if (oldError * error < T(0)) {
			dir *= T(-0.1);
		}
		oldError = error;
		pos += dir;
		error = sin(acos(pos.normalized().dot(pl.normal))) - sin(acos((pos3d - pos).normalized().dot(pl.normal))) * r_w<T>;
	} while (abs(error) > T(1e-13));
	return pos;
}

namespace {
	template<typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
	struct Functor {
		typedef _Scalar Scalar;
		enum {
			InputsAtCompileTime = NX,
			ValuesAtCompileTime = NY
		};
		typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
		typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
		typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;
	};

	template<typename T>
	struct IterativeForwardFunctor : Functor<double> {
		IterativeForwardFunctor(int inputs, int values, const Eigen::Vector3t<T>& pos3d, const plane<T>& pl, const camera& cam)
			: inputs_(inputs), values_(values), pos3d(pos3d), pl(pl), cam(cam) {
		}

		const plane<T>& pl;
		const Eigen::Vector3t<T>& pos3d;
		const camera& cam;

		int operator()(const Eigen::VectorXt<T>& b, Eigen::VectorXt<T>& fvec) const {
			Eigen::Vector2t<T> pos2d(b[0], b[1]);
			auto ray = backward(pos2d, pl, cam);
			Eigen::Vector3t<T> pos = ray.origin + ray.dir * (pos3d.z() - ray.origin.z()) / ray.dir.z();
			fvec[0] = pos.x() - pos3d.x();
			fvec[1] = pos.y() - pos3d.y();
			return 0;
		}

		const int inputs_;
		const int values_;
		int inputs() const { return inputs_; }
		int values() const { return values_; }
	};
}

template <typename T>
Eigen::Vector2t<T> iterativeForward(const Eigen::Vector3t<T>& pos3d, const plane<T>& pl, const camera& cam) {
	IterativeForwardFunctor<T> functor(2, 2, pos3d, pl, cam);

	Eigen::NumericalDiff<IterativeForwardFunctor<T>> numDiff(functor);
	Eigen::LevenbergMarquardt<Eigen::NumericalDiff<IterativeForwardFunctor<T>>> lm(numDiff);

	Eigen::VectorXt<T> p = cam.project(pos3d);
	int info = lm.minimize(p);

	return { p[0], p[1] };
}

