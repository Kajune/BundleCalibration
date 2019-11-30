#define _USE_MATH_DEFINES
#include <cmath>
#include <fstream>
#include <memory>
#include <random>

#include "..\EigenSupport.h"
#include "..\plyio.hpp"
#include "ceres\ceres.h"
#include "glog\logging.h"

using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

// カメラ画像上のpos2Dを空間中に逆投影するベクトル
template <typename T>
Eigen::Vector3t<T> castRay(const Eigen::Vector2t<T>& pos2D, const Eigen::Matrix3t<T>& camMat) {
	return Eigen::Matrix3t<T>(camMat.inverse()) * Eigen::Vector3t<T>(T(pos2D.x()), T(pos2D.y()), T(1));
}

// 回転ベクトルから回転行列への変換
template <typename T>
Eigen::Matrix3t<T> rotVecToRotMat(const Eigen::Vector3t<T>& rotVec) {
	Eigen::AngleAxis<T> angleX(rotVec.x(), Eigen::Vector3t<T>::UnitX());
	Eigen::AngleAxis<T> angleY(rotVec.y(), Eigen::Vector3t<T>::UnitY());
	Eigen::AngleAxis<T> angleZ(rotVec.z(), Eigen::Vector3t<T>::UnitZ());

	Eigen::Quaternion<T> q = angleX * angleY * angleZ;

	return q.matrix();
}

// ベクトルと平面の交点
template <typename T>
Eigen::Vector3t<T> intersectPlane(const Eigen::Vector3t<T>& origin, const Eigen::Vector3t<T>& dir, const Eigen::Vector3t<T>& planeOrigin, const Eigen::Vector3t<T>& planeNormal) {
	return origin - dir * (planeNormal.dot(origin - planeOrigin) / planeNormal.dot(dir));
}

struct CostFunctor {
	template <typename T> bool operator()(const T* const plane, const T* const projParam, const T* const projR, const T* const projT, T* residual) const {
		Eigen::Vector3t<T> camVecT(T(camVec.x()), T(camVec.y()), T(camVec.z()));

		// 仮定した平面とカメラ光線ベクトルの交点
		Eigen::Vector3t<T> planePos = intersectPlane(Eigen::Vector3t<T>{T(0), T(0), T(0)}, camVecT, 
													 Eigen::Vector3t<T>{T(0), T(0), plane[2]}, Eigen::Vector3t<T>{plane[0], plane[1], T(1)}.normalized());
	
		// RTの計算
		Eigen::Vector3t<T> transVec(projT[0], projT[1], projT[2]);
		Eigen::Matrix3t<T> rotMat = rotVecToRotMat(Eigen::Vector3t<T>{ projR[0], projR[1], projR[2] });

		Eigen::Matrix3t<T> projMat;
		projMat << projParam[0], T(0), projParam[1], T(0), projParam[2], projParam[3], T(0), T(0), T(1);

		// プロジェクタ平面に再投影
		Eigen::Vector3t<T> projPos3D = rotMat * planePos + transVec;
		Eigen::Vector3t<T> projPos2D = projMat * projPos3D;

		residual[0] = projPos2D.x() / projPos2D.z() - T(projX);

		return true;
	}

	Eigen::Vector3d camVec;
	double projX;

	CostFunctor(const Eigen::Vector3d& camVec, double projX)
		: camVec(camVec), projX(projX) {
	}

	static ceres::CostFunction* Create(const Eigen::Vector3d& camVec, double projX) {
		return (new ceres::AutoDiffCostFunction<CostFunctor, 1, 3, 4, 3, 3>(new CostFunctor(camVec, projX)));
	}
};

int main(int argc, char* argv[]) {
/*	if (argc < 5) {
		std::cout << "Usage: numImages, camParam.txt, initProjParam.txt, correspondence file format" << std::endl;
		return 0;
	}

	const int numImages = std::stoi(argv[1]);

	// Read camera parameter
	std::ifstream ifs_cam(argv[2]);
	Eigen::Matrix3d camMat;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			ifs_cam >> camMat(i, j);
		}
	}

	// Read proj parameter
	std::ifstream ifs_proj(argv[3]);
	Eigen::Matrix3d projMat;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			ifs_proj >> projMat(i, j);
		}
	}

	// Read correspondence
	// カメラ画像上での画素と対応するプロジェクタ画像上でのx座標のリスト
	std::vector<std::vector<std::pair<Eigen::Vector2d, double>>> pointList(numImages);
	for (int i = 0; i < numImages; i++) {
		char filename[FILENAME_MAX];
		sprintf_s(filename, argv[4], i);
		std::ifstream ifs_pos(filename);
		while (true) {
			Eigen::Vector2d camPos;
			double projX;
			ifs_pos >> camPos.x() >> camPos.y() >> projX;
			if (ifs_pos.eof()) {
				break;
			}
			pointList.at(i).emplace_back(std::make_pair(camPos, projX));
		}
	}*/

	// テスト用
	const int numImages = 3;

	Eigen::Matrix3d camMat;
	camMat << 4000, 0, 1024, 0, 3800, 1024, 0, 0, 1;

	Eigen::Matrix3d projMat_gt;
	projMat_gt << 3500, 0, 1048, 0, 3200, 1000, 0, 0, 1;

	Eigen::Matrix3d projMat;
	projMat << 4000, 0, 1024, 0, 4000, 1024, 0, 0, 1;

	Eigen::Matrix3d rotMat_gt = rotVecToRotMat(Eigen::Vector3d(2 * M_PI / 180.0, -20 * M_PI / 180.0, 1 * M_PI / 180.0));
	Eigen::Vector3d trans_gt(0.3, 0.03, 0.05);
	
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<> dist_depth(0.8, 1.2), dist_normal(-0.5, 0.5), dist_pixel(0, 2048);
	std::vector<std::vector<std::pair<Eigen::Vector2d, double>>> pointList(numImages);
	std::vector<Eigen::Vector3d> planeParams;
	for (int i = 0; i < numImages; i++) {
		Eigen::Vector3d planeParam(dist_normal(mt), dist_normal(mt), dist_depth(mt));
		Eigen::Vector3d planeNormal = Eigen::Vector3d(planeParam.x(), planeParam.y(), 1).normalized();
		Eigen::Vector3d planeOrigin(0, 0, planeParam.z());
		planeParams.emplace_back(planeParam);

		for (int j = 0; j < 100; j++) {
			Eigen::Vector2d pos2d(dist_pixel(mt), dist_pixel(mt));
			Eigen::Vector3d pos3d = intersectPlane(Eigen::Vector3d(0, 0, 0), castRay(pos2d, camMat), planeOrigin, planeNormal);
			Eigen::Vector3d pos3d_trans = rotMat_gt * pos3d + trans_gt;
			Eigen::Vector3d projPos = projMat_gt * pos3d_trans;
			if (projPos.x() / projPos.z() < 0 || 2048 <= projPos.x() / projPos.z()) {
				continue;
			}
			pointList.at(i).emplace_back(std::make_pair(pos2d, projPos.x() / projPos.z()));
		}
	}

	// Parameter preparation
	std::unique_ptr<double[]> planeParam(new double[3 * numImages]);
	for (int i = 0; i < numImages; i++) {
		planeParam[i * 3 + 0] = 0.0;
		planeParam[i * 3 + 1] = 0.0;
		planeParam[i * 3 + 2] = 1.0;
	}
	double projParam[4] = { projMat(0, 0), projMat(0, 2), projMat(1, 1), projMat(1, 2) };
	double projR[3] = { 0.0, 0.0, 0.0 };
	double projT[3] = { 1.0, 0.0, 0.0 };

	// Bundle adjustment
	Problem problem;
	for (int i = 0; i < numImages; i++) {
		for (const auto& it : pointList.at(i)) {
			ceres::CostFunction* cost_function = CostFunctor::Create(castRay(it.first, camMat), it.second);
			problem.AddResidualBlock(cost_function, nullptr, &planeParam[i * 3], projParam, projR, projT);
		}
	}

	Solver::Options options;
	options.minimizer_progress_to_stdout = true;
	options.function_tolerance = 1e-32;
	options.gradient_tolerance = 1e-32;
	options.parameter_tolerance = 1e-32;
	options.linear_solver_type = ceres::LinearSolverType::DENSE_SCHUR;

	Solver::Summary summary;
	Solve(options, &problem, &summary);

	std::cout << summary.FullReport() << std::endl;

	for (int i = 0; i < numImages; i++) {
		std::cout << planeParam[i * 3 + 0] - planeParams.at(i).x() << ", " << planeParam[i * 3 + 1] - planeParams.at(i).y() << ", " << planeParam[i * 3 + 2] - planeParams.at(i).z() << std::endl;
	}
	std::cout << projParam[0] << ", " << projParam[1] << ", " << projParam[2] << ", " << projParam[3] << std::endl;
	std::cout << projR[0] << ", " << projR[1] << ", " << projR[2] << ", " << projT[0] << ", " << projT[1] << ", " << projT[2] << std::endl;
}
