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
	template <typename T> bool operator()(const T* const boardR, const T* const boardT, const T* const planeN, const T* const planeD_inv, const T* const camParam, T* residual) const {
		plane<T> pl(Eigen::Vector3t<T>(planeN[0], planeN[1], T(1)).normalized(), T(1) / planeD_inv[0]);

//		Eigen::Vector3t<T> rvec(boardR[0], boardR[1], boardR[2]);
		cv::Mat rvec = (cv::Mat_<T>(3, 1) << boardR[0], boardR[1], boardR[2]);
		Eigen::Matrix3t<T> rmat;
		cv::Mat rmat_;
		cv::Rodrigues(rvec, rmat_);
		cv::cv2eigen(rmat_, rmat);

		Eigen::Vector3t<T> point3DTrans = rmat * Eigen::Vector3t<T>(T(point3D.x()), T(point3D.y()), T(point3D.z())) + Eigen::Vector3t<T>(boardT[0], boardT[1], boardT[2]);

		Eigen::Matrix3t<T> camMat;
		camMat << camParam[0], T(0), camParam[1], T(0), camParam[2], camParam[3], T(0), T(0), T(1);
		Eigen::Vector3t<T> proj2D = camMat * forward(point3DTrans, pl);
		proj2D /= proj2D.z();

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
		return (new ceres::NumericDiffCostFunction<CostFunctor, ceres::CENTRAL, 2, 3, 3, 2, 1, 4>(new CostFunctor(point2D, point3D, cam)));
	}
};

int main(int argc, char* argv[]) {
	if (argc < 8) {
		std::cout << "Usage: camMat.txt, Num Images, point2D format, point3D format, image format, out nd.txt, result image format" << std::endl;
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

	const int numImages = std::stoi(argv[2]);

	std::vector<std::vector<cv::Point2d>> point2DList(numImages);
	std::vector<std::vector<cv::Point3d>> point3DList(numImages);
	for (int i = 0; i < numImages; i++) {
		char filename[FILENAME_MAX];
		sprintf_s(filename, argv[3], i);
		std::ifstream ifs_point2d(filename);
		if (!ifs_point2d) {
			std::cout << "point2D: " << filename << " not found." << std::endl;
			continue;
		}

		sprintf_s(filename, argv[4], i);
		std::ifstream ifs_point3d(filename);
		if (!ifs_point3d) {
			std::cout << "point3D: " << filename << " not found." << std::endl;
			continue;
		}

		while (true) {
			float x, y, X, Y, Z;
			ifs_point2d >> x >> y;
			ifs_point3d >> X >> Y >> Z;
			if (ifs_point2d.eof() || ifs_point3d.eof()) {
				break;
			}
			point2DList.at(i).emplace_back(x, y);
			point3DList.at(i).emplace_back(X, Y, Z);
		}
	}

	// Parameter preparation
	std::vector<cv::Mat> rvecs(numImages), tvecs(numImages);
	double minDistance = DBL_MAX;
	for (int i = 0; i < numImages; i++) {
		if (point2DList.at(i).empty()) {
			continue;
		}
		cv::solvePnP(point3DList.at(i), point2DList.at(i), camMat, dist, rvecs.at(i), tvecs.at(i));
		minDistance = std::min(minDistance, tvecs.at(i).at<double>(2, 0));
	}

	std::unique_ptr<double[]> boardR(new double[3 * numImages]);
	std::unique_ptr<double[]> boardT(new double[3 * numImages]);
	for (int i = 0; i < numImages; i++) {
		if (point2DList.at(i).empty()) {
			continue;
		}
		boardR[i * 3 + 0] = rvecs.at(i).at<double>(0, 0);
		boardR[i * 3 + 1] = rvecs.at(i).at<double>(1, 0);
		boardR[i * 3 + 2] = rvecs.at(i).at<double>(2, 0);
		boardT[i * 3 + 0] = tvecs.at(i).at<double>(0, 0);
		boardT[i * 3 + 1] = tvecs.at(i).at<double>(1, 0);
		boardT[i * 3 + 2] = tvecs.at(i).at<double>(2, 0);
	}
	double planeN[2] = { 0.0, 0.0 };
	double planeD_inv = 1.0 / (minDistance / 2);
	double camParam[4] = { camMat.at<float>(0, 0), camMat.at<float>(0, 2), camMat.at<float>(1, 1), camMat.at<float>(1, 2) };

	// Bundle adjustment
	Problem problem;
	for (int i = 0; i < numImages; i++) {
		for (int j = 0; j < point2DList.at(i).size(); j++) {
			Eigen::Vector2d point2D(point2DList.at(i).at(j).x, point2DList.at(i).at(j).y);
			Eigen::Vector3d point3D(point3DList.at(i).at(j).x, point3DList.at(i).at(j).y, point3DList.at(i).at(j).z);
			ceres::CostFunction* cost_function = CostFunctor::Create(point2D, point3D, cam);
			problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1.0), &boardR[i * 3], &boardT[i * 3], planeN, &planeD_inv, camParam);
		}
	}
	problem.SetParameterLowerBound(&planeD_inv, 0, 0.0);
	problem.SetParameterBlockConstant(camParam);

	Solver::Options options;
	options.minimizer_progress_to_stdout = true;
	options.function_tolerance = 1e-32;
	options.gradient_tolerance = 1e-32;
	options.parameter_tolerance = 1e-32;
	options.max_num_iterations = 1000;
	options.gradient_check_numeric_derivative_relative_step_size = 1e-10;
	options.linear_solver_type = ceres::LinearSolverType::DENSE_SCHUR;

	Solver::Summary summary;
	Solve(options, &problem, &summary);

	std::cout << summary.FullReport() << std::endl;

	double sum_error = 0.0;
	int errorCount = 0;
	plane<double> pl(Eigen::Vector3d(planeN[0], planeN[1], 1).normalized(), 1.0 / planeD_inv);
	for (int i = 0; i < numImages; i++) {
		char filename[FILENAME_MAX];
		sprintf_s(filename, argv[5], i);
		auto img = cv::imread(filename);
		for (int j = 0; j < point2DList.at(i).size(); j++) {
			Eigen::Vector2d point2D(point2DList.at(i).at(j).x, point2DList.at(i).at(j).y);
			Eigen::Vector3d point3D(point3DList.at(i).at(j).x, point3DList.at(i).at(j).y, point3DList.at(i).at(j).z);

//			Eigen::Vector3d rvec(boardR[i * 3 + 0], boardR[i * 3 + 1], boardR[i * 3 + 2]);
			cv::Mat rvec = (cv::Mat_<double>(3, 1) << boardR[i * 3 + 0], boardR[i * 3 + 1], boardR[i * 3 + 2]);
			Eigen::Matrix3d rmat;
			cv::Mat rmat_;
			cv::Rodrigues(rvec, rmat_);
			cv::cv2eigen(rmat_, rmat);

			Eigen::Vector3d point3DTrans = rmat * point3D + Eigen::Vector3d(boardT[i * 3 + 0], boardT[i * 3 + 1], boardT[i * 3 + 2]);
			Eigen::Vector2d proj2D = cam.project(forward(point3DTrans, pl));

			sum_error += (proj2D - point2D).squaredNorm();
			errorCount++;
			cv::circle(img, cv::Point(point2D.x(), point2D.y()), 3, cv::Scalar(0, 255, 0));
			cv::circle(img, cv::Point(proj2D.x(), proj2D.y()), 3, cv::Scalar(0, 0, 255));
		}

		sprintf_s(filename, argv[7], i);
		cv::imwrite(filename, img);
	}
	std::cout << "RMSE: " << sqrt(sum_error / errorCount) << std::endl;

	std::ofstream ofs(argv[6]);
	ofs << planeN[0] << " " << planeN[1] << std::endl;
	ofs << 1.0 / planeD_inv << std::endl;
}
