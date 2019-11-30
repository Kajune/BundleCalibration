#pragma once

#include "EigenSupport.h"
#include <fstream>
#include <vector>

using color = Eigen::Vector3d;

void writeHeader(std::ofstream& ofs, int pointNum, bool color, int edgeNum = 0, int faceNum = 0) {
	ofs << "ply" << std::endl;
	ofs << "format ascii 1.0" << std::endl;
	ofs << "element vertex " << pointNum << std::endl;
	ofs << "property float x" << std::endl;
	ofs << "property float y" << std::endl;
	ofs << "property float z" << std::endl;
	if (color) {
		ofs << "property uchar red" << std::endl;
		ofs << "property uchar green" << std::endl;
		ofs << "property uchar blue" << std::endl;
	}
	if (edgeNum) {
		ofs << "element edge " << edgeNum << std::endl;
		ofs << "property int vertex1" << std::endl;
		ofs << "property int vertex2" << std::endl;
		if (color) {
			ofs << "property uchar red" << std::endl;
			ofs << "property uchar green" << std::endl;
			ofs << "property uchar blue" << std::endl;
		}
	}
	if (faceNum) {
		ofs << "element face " << faceNum << std::endl;
		ofs << "property list uchar int vertex_indices" << std::endl;
	}
	ofs << "end_header" << std::endl;
}

template <typename T>
void writePoint(std::ofstream& ofs, const Eigen::Vector3t<T>& point) {
	ofs << point.x() << " " << point.y() << " " << point.z() << std::endl;
}

template <typename T>
void writePoint(std::ofstream& ofs, const Eigen::Vector3t<T>& point, const color& col) {
	ofs << point.x() << " " << point.y() << " " << point.z() << " " << col.x() << " " << col.y() << " " << col.z() << std::endl;
}

template <typename T>
void writePoints(std::ofstream& ofs, const std::vector<Eigen::Vector3t<T>>& points) {
	for (const auto& it : points) {
		ofs << it.x() << " " << it.y() << " " << it.z() << std::endl;
	}
}

template <typename T>
void writePoints(std::ofstream& ofs, const std::vector<Eigen::Vector3t<T>>& points, const color& col) {
	for (const auto& it : points) {
		ofs << it.x() << " " << it.y() << " " << it.z() << " " << col.x() << " " << col.y() << " " << col.z() << std::endl;
	}
}

void writeEdges(std::ofstream& ofs, const std::vector<std::pair<int, int>>& edges) {
	for (const auto& it : edges) {
		ofs << it.first << " " << it.second << std::endl;
	}
}

void writeEdges(std::ofstream& ofs, const std::vector<std::pair<int, int>>& edges, const color& col) {
	for (const auto& it : edges) {
		ofs << it.first << " " << it.second << " " << col.x() << " " << col.y() << " " << col.z() << std::endl;
	}
}

void writeFaces(std::ofstream& ofs, const std::vector<std::vector<int>>& faces) {
	for (const auto& it : faces) {
		ofs << it.size() << " ";
		for (const auto& it2 : it) {
			ofs << it2 << " ";
		}
		ofs << std::endl;
	}
}

template <typename T>
void writePly(std::ofstream& ofs, const std::vector<Eigen::Vector3t<T>>& points) {
	ofs << "ply" << std::endl;
	ofs << "format ascii 1.0" << std::endl;
	ofs << "element vertex " << points.size() << std::endl;
	ofs << "property float x" << std::endl;
	ofs << "property float y" << std::endl;
	ofs << "property float z" << std::endl;
	ofs << "end_header" << std::endl;
	writePoints(ofs, points);
}

double calcRMSE(const std::vector<Eigen::Vector3d>& points, const std::vector<Eigen::Vector3d>& points_gt) {
	double sum = 0;
	for (int i = 0; i < points.size(); i++) {
		sum += (points.at(i) - points_gt.at(i)).squaredNorm();
	}
	return sqrt(sum / points.size());
}

double calcRMSE(const std::vector<Eigen::Vector3d>& points, const std::vector<Eigen::Vector3d>& points_gt, const std::vector<int>& mask) {
	double sum = 0;
	int count = 0;
	for (int i = 0; i < points.size(); i++) {
		if (mask.at(i)) {
			continue;
		}
		sum += (points.at(i) - points_gt.at(i)).squaredNorm();
		count++;
	}
	return sqrt(sum / count);
}
