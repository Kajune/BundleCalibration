#define _USE_MATH_DEFINES
#include <cmath>
#include <fstream>
#include <memory>

#include "..\EigenSupport.h"
#include "..\equation.hpp"
#include "..\lightPath.hpp"
#include "ceres\ceres.h"
#include "glog\logging.h"

using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

struct CostFunctor {
	template <typename T> bool operator()(const T* const boardR, const T* const boardT, const T* const planeN, const T* const planeD, T* residual) const {
		plane<T> pl(Eigen::Vector3t<T>(planeN[0], planeN[1], T(1)).normalized(), planeD[0]);

		Eigen::Vector3t<T> rvec(boardR[0], boardR[1], boardR[2]);
		Eigen::Matrix3t<T> rmat;
		T theta = rvec.norm();
		if (theta < DBL_EPSILON) {
			rmat.setIdentity();
		} else {
			T c = cos(theta);
			T s = sin(theta);

			Eigen::Matrix3t<T> rrt = rvec * rvec.transpose();
			Eigen::Matrix3t<T> r_x;
			r_x << T(0), -rvec.z(), rvec.y(), rvec.z(), T(0), -rvec.x(), -rvec.y(), rvec.x(), T(0);

			rvec /= theta;
			rmat = Eigen::Matrix3t<T>::Identity() * c + (T(1.0) - c) * rrt + s * r_x;
		}

		Eigen::Vector3t<T> point3DTrans = rmat * Eigen::Vector3t<T>(T(point3D.x()), T(point3D.y()), T(point3D.z())) + Eigen::Vector3t<T>(boardT[0], boardT[1], boardT[2]);
		Eigen::Vector2t<T> proj2D = cam.project(forward(point3DTrans, pl));

		residual[0] = proj2D.x() - T(point2D.x());
		residual[1] = proj2D.y() - T(point2D.y());

		return true;
	}

	Eigen::Vector2d point2D;
	Eigen::Vector3d point3D;
	const camera& cam;

	CostFunctor(const Eigen::Vector2d& point2D, const Eigen::Vector3d& point3D, const camera& cam)
		: point2D(point2D), point3D(point3D), cam(cam) {
	}

	static ceres::CostFunction* Create(const Eigen::Vector2d& point2D, const Eigen::Vector3d& point3D, const camera& cam) {
		return (new ceres::NumericDiffCostFunction<CostFunctor, ceres::CENTRAL, 2, 3, 3, 2, 1>(new CostFunctor(point2D, point3D, cam)));
	}
};

int main(int argc, char* argv[]) {
	if (argc < 9) {
		std::cout << "Usage: camMat.txt, point2D.txt, chessX, chessY, scale, image, out nd.txt, result image" << std::endl;
		return 0;
	}

	// Read camera parameter
	std::ifstream ifs_cam(argv[1]);
	if (!ifs_cam) {
		std::cout << "CamMat: " << argv[1] << " not found." << std::endl;
		return 0;
	}
	cv::Mat camMat(3, 3, CV_32F);
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			ifs_cam >> camMat.at<float>(i, j);
		}
	}

	cv::Mat dist = cv::Mat::zeros(1, 4, CV_32F);
	realCamera cam(camMat, dist);

	const int chessX = std::stoi(argv[3]);
	const int chessY = std::stoi(argv[4]);
	const float chessScale = std::stof(argv[5]);

	std::vector<cv::Point2d> point2DList;
	std::vector<cv::Point3d> point3DList;
	std::ifstream ifs_point(argv[2]);
	if (!ifs_point) {
		std::cout << "point2D: " << argv[2] << " not found." << std::endl;
		return 0;
	}
	for (int y = 0; y < chessY; y++) {
		for (int x = 0; x < chessX; x++) {
			float posX, posY;
			ifs_point >> posX >> posY;
			point2DList.emplace_back(posX, posY);
			point3DList.emplace_back(x * chessScale, y * chessScale, 0.0);
		}
	}

	// Parameter preparation
	cv::Mat rvec, tvec;
	cv::solvePnP(point3DList, point2DList, camMat, dist, rvec, tvec);

	double boardR[3] = { rvec.at<double>(0, 0), rvec.at<double>(1, 0), rvec.at<double>(2, 0) };
	double boardT[3] = { tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0) };
	double planeN[2] = { 0.0, 0.0 };
	double planeD = tvec.at<double>(2, 0) / 2.0;

	// Bundle adjustment
	Problem problem;
	for (int i = 0; i < point2DList.size(); i++) {
		Eigen::Vector2d point2D(point2DList.at(i).x, point2DList.at(i).y);
		Eigen::Vector3d point3D(point3DList.at(i).x, point3DList.at(i).y, point3DList.at(i).z);
		ceres::CostFunction* cost_function = CostFunctor::Create(point2D, point3D, cam);
		problem.AddResidualBlock(cost_function, nullptr, boardR, boardT, planeN, &planeD);
	}

	Solver::Options options;
	options.minimizer_progress_to_stdout = false;
	options.linear_solver_type = ceres::LinearSolverType::DENSE_SCHUR;

	Solver::Summary summary;
	Solve(options, &problem, &summary);

	std::cout << summary.BriefReport() << std::endl;

	std::cout << "boardR: " << boardR[0] << ", " << boardR[1] << ", " << boardR[2] << std::endl;
	std::cout << "boardT: " << boardT[0] << ", " << boardT[1] << ", " << boardT[2] << std::endl;
	std::cout << "planeN: " << planeN[0] << ", " << planeN[1] << std::endl;
	std::cout << "planeD: " << planeD << std::endl;

	auto img = cv::imread(argv[6]);
	double sum_error = 0.0;
	plane<double> pl(Eigen::Vector3d(planeN[0], planeN[1], 1).normalized(), planeD);
	for (int i = 0; i < point2DList.size(); i++) {
		Eigen::Vector2d point2D(point2DList.at(i).x, point2DList.at(i).y);
		Eigen::Vector3d point3D(point3DList.at(i).x, point3DList.at(i).y, point3DList.at(i).z);

		Eigen::Vector3d rvec(boardR[0], boardR[1], boardR[2]);
		Eigen::Matrix3d rmat;
		double theta = rvec.norm();
		if (theta < DBL_EPSILON) {
			rmat.setIdentity();
		} else {
			double c = cos(theta);
			double s = sin(theta);

			Eigen::Matrix3d rrt = rvec * rvec.transpose();
			Eigen::Matrix3d r_x;
			r_x << 0, -rvec.z(), rvec.y(), rvec.z(), 0, -rvec.x(), -rvec.y(), rvec.x(), 0;

			rvec /= theta;
			rmat = Eigen::Matrix3d::Identity() * c + (1.0 - c) * rrt + s * r_x;
		}

		Eigen::Vector3d point3DTrans = rmat * point3D + Eigen::Vector3d(boardT[0], boardT[1], boardT[2]);
		Eigen::Vector2d proj2D = cam.project(forward(point3DTrans, pl));

		sum_error += (proj2D - point2D).squaredNorm();
		cv::circle(img, cv::Point(point2D.x(), point2D.y()), 3, cv::Scalar(0, 255, 0));
		cv::circle(img, cv::Point(proj2D.x(), proj2D.y()), 3, cv::Scalar(0, 0, 255));
	}
	std::cout << "RMSE: " << sqrt(sum_error / point2DList.size()) << std::endl;

	cv::imwrite(argv[8], img);

	std::ofstream ofs(argv[7]);
	ofs << planeN[0] << " " << planeN[1] << std::endl;
	ofs << planeD << std::endl;
}
