#pragma once
#include "EigenSupport.h"
#define _USE_MATH_DEFINES
#include <cmath>
#include <list>
#include <random>
#include <memory>
#include <iostream>
#include <opencv2\core.hpp>
#include <opencv2\core\eigen.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\calib3d.hpp>

template <typename T>
struct ray {
	Eigen::Vector3t<T> origin, dir;
	Eigen::Vector2t<T> pos2D;
	Eigen::Vector3t<T> at(const T& t) const {
		return origin + t * dir;
	}
};

template <typename T>
struct plane {
	Eigen::Vector3t<T> normal, origin;
	plane(const Eigen::Vector3t<T> n, const T& d) : normal(n.normalized()), origin(n.normalized() * d) {}
	plane() : normal(T(0), T(0), T(1)), origin(T(0), T(0), T(0)) {}
	plane(const Eigen::Quaternion<T>& q, const T& d)
		: normal(q * Eigen::Vector3t<T>(T(0), T(0), T(1))), origin(q * Eigen::Vector3t<T>(T(0), T(0), T(1)) * d) {
	}

	virtual Eigen::Vector3t<T> getNormal(const ray<T>& in) const {
		return normal;
	}
};

struct camera {
	Eigen::Vector2d resolution, CCDSize, center;
	double focus;

	virtual Eigen::Matrix3d getCameraMatrix() const {
		Eigen::Matrix3d m;
		Eigen::Vector2d focus_ = Eigen::Vector2d(focus, focus).cwiseProduct(resolution).cwiseQuotient(CCDSize);
		m << focus_.x(), 0, center.x(), 0, focus_.y(), center.y(), 0, 0, 1;
		return m;
	}

	template <typename T>
	ray<T> castRay(const Eigen::Vector2t<T>& pos2D) const {
		Eigen::Vector3t<T> pos3d = (Eigen::Matrix3t<T>(getCameraMatrix().inverse()) * Eigen::Vector3t<T>(pos2D.x(), pos2D.y(), T(1)));
		return ray<T>{ { T(0), T(0), T(0) }, pos3d.normalized(), pos2D };
	}

	template <typename T>
	Eigen::Vector2t<T> project(const Eigen::Vector3t<T>& pos3D) const {
		Eigen::Vector3t<T> proj = getCameraMatrix() * pos3D;
		proj /= proj.z();
		return Eigen::Vector2t<T>{ proj.x(), proj.y() };
	}
};

struct realCamera : public camera {
	cv::Mat camMat, camDist;

	realCamera(const cv::Mat& camMat_, const cv::Mat& camDist_) 
		: camMat(camMat_), camDist(camDist_) {}

	void calcCamParams(const Eigen::Vector2d& resolution_, const Eigen::Vector2d& CCDSize_) {
		resolution = resolution_;
		CCDSize = CCDSize_;
		double fovx, fovy, aspect;
		cv::Point2d pp;
		cv::calibrationMatrixValues(camMat, cv::Size(resolution.x(), resolution.y()), 0.0, 0.0, fovx, fovy, focus, pp, aspect);
		center = Eigen::Vector2d(pp.x, pp.y);
		cv::calibrationMatrixValues(camMat, cv::Size(resolution.x(), resolution.y()), CCDSize.x(), CCDSize.y(), fovx, fovy, focus, pp, aspect);
	}

	virtual Eigen::Matrix3d getCameraMatrix() const {
		Eigen::Matrix3d ret;
		cv::cv2eigen(camMat, ret);
		return ret;
	}
};

template <typename T>
ray<T> intersectPlane(const ray<T>& in, const plane<T>& target) {
	return ray<T>{ in.at(-(target.normal.dot(in.origin - target.origin) / target.normal.dot(in.dir))), in.dir, in.pos2D };
}

template <typename T>
Eigen::Vector3t<T> reflect(const Eigen::Vector3t<T> in, const Eigen::Vector3t<T> normal) {
	return (in - 2.0 * in.dot(normal) * normal).normalized();
}

template <typename T>
Eigen::Vector3t<T> refract(const Eigen::Vector3t<T> in, const Eigen::Vector3t<T> normal, const T& n) {
	auto dot = in.dot(-normal);
	return (in / n - (dot + sqrt(pow(n, 2.0) + pow(dot, 2.0) - 1.0)) / n * (-normal)).normalized();
}

template <typename T>
bool lineToLineCenter(const ray<T>& ray1, const ray<T>& ray2, Eigen::Vector3t<T>& center) {
	T D1 = (ray2.origin - ray1.origin).dot(ray1.dir);
	T D2 = (ray2.origin - ray1.origin).dot(ray2.dir);
	T Dv = ray1.dir.dot(ray2.dir);

	if (Dv < T(1e-10)) {
		return false;
	}

	T t1 = (D1 - D2 * Dv) / (T(1.0) - Dv * Dv);
	T t2 = (D2 - D1 * Dv) / (Dv * Dv - T(1.0));

	center = (ray1.at(t1) + ray2.at(t2)) / T(2);
	return true;
}

template <typename T>
T lineToPointDistance(const ray<T>& ray, const Eigen::Vector3t<T>& pos3d) {
	return ((ray.origin - pos3d) - (ray.origin - pos3d).dot(ray.dir) * ray.dir).norm();
}

