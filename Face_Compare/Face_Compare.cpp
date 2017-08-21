// Face_Compare.cpp : ���� DLL Ӧ�ó���ĵ���������


/*

��ǰ�汾��1.0.0.2

*/
//

#include "stdafx.h"


#pragma once
#include<iostream>
using namespace std;


#ifdef _WIN32
#pragma once
#include <opencv2/core/version.hpp>
#include <opencv2/core.hpp>
#define CV_VERSION_ID CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) \
  CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#define cvLIB(name) "opencv_" name CV_VERSION_ID "d"
#else
#define cvLIB(name) "opencv_" name CV_VERSION_ID
#endif //_DEBUG

#pragma comment( lib, cvLIB("core") )
#pragma comment( lib, cvLIB("imgproc") )
#pragma comment( lib, cvLIB("highgui") )

#endif //_WIN32

#if defined(__unix__) || defined(__APPLE__)

#ifndef fopen_s

#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),(mode)))==NULL

#endif //fopen_s

#endif //__unix

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "face_identification.h"
#include "recognizer.h"
#include "face_detection.h"
#include "face_alignment.h"

#include "math_functions.h"

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <io.h>
#include <windows.h>

using namespace seeta;

#define TEST(major, minor) major##_##minor##_Tester()
#define EXPECT_NE(a, b) if ((a) == (b)) std::cout << "ERROR: "
#define EXPECT_EQ(a, b) if ((a) != (b)) std::cout << "ERROR: "


//�ٽ�ֵ
#define RIM_VALUE 0.6
//model���·��
#define FA_MODEL_REL "/model/seeta_fa_v1.1.bin"
#define FD_MODEL_REL "/model/seeta_fd_frontal_v1.0.bin"
#define FR_MODEL_REL "/model/seeta_fr_v1.0.bin"




extern "C" _declspec(dllexport) int compare(
	const char* szPhoto1Path,
	const char* szPhoto2Path,
	float &fsimilarity,
	char* szErrMsg)
{
	//**********************************

	//�õ���ǰexe����Ŀ¼
	char strModel[256];
	GetModuleFileName(NULL, strModel, 256);
	std::string currentPath = strModel;
	currentPath = currentPath.substr(0, currentPath.find_last_of("\\"));


	//�ж�·������
	if (_access(szPhoto1Path, 0) == -1)
	{
		strcpy(szErrMsg, "����֤��Ƭ��");
		fsimilarity = 0;
		return 0;
	}
	if(_access(szPhoto2Path, 0) == -1)
	{
		//std::cout << "image not exist!" << std::endl;
		strcpy(szErrMsg, "��������Ƭ��");
		fsimilarity = 0;
		return 0;
	}


	//**********************************
	//�ж�model �Ƿ����
	if (_access((currentPath + FD_MODEL_REL).c_str(), 0) == -1
		|| _access((currentPath + FA_MODEL_REL).c_str(), 0) == -1
		|| _access((currentPath + FR_MODEL_REL).c_str(), 0) == -1)
	{
		//std::cout << "model not exist!" << std::endl;
		strcpy(szErrMsg, "����ģ�ͼ���ʧ�ܡ�");
		fsimilarity = 0;
		return 0;
	}
	//�ж�ָ��szErrMsg�Ƿ�Ϊ��


	// Initialize face detection model
	seeta::FaceDetection detector((currentPath + FD_MODEL_REL).c_str());
	detector.SetMinFaceSize(40);
	detector.SetScoreThresh(2.f);
	detector.SetImagePyramidScaleFactor(0.8f);
	detector.SetWindowStep(4, 4);

	// Initialize face alignment model 
	seeta::FaceAlignment point_detector((currentPath + FA_MODEL_REL).c_str());

	// Initialize face Identification model 
	FaceIdentification face_recognizer((currentPath + FR_MODEL_REL).c_str());

	//load image
	cv::Mat gallery_img_color = cv::imread(szPhoto1Path, 1); //��һ��
	 //�ж�ͼƬ���سɹ�
	 //*************************
	if (!gallery_img_color.data)
	{
		strcpy(szErrMsg, "��֤��Ƭ����ʧ�ܡ�");
		fsimilarity = 0;
		return 0;
	}
	//**************************
	cv::Mat gallery_img_gray;
	cv::cvtColor(gallery_img_color, gallery_img_gray, CV_BGR2GRAY);

	cv::Mat probe_img_color = cv::imread(szPhoto2Path, 1);   //�ڶ���
	 //�ж�ͼƬ���سɹ�
	 //*************************
	if (!probe_img_color.data)
	{
		strcpy(szErrMsg, "������Ƭ����ʧ�ܡ�");
		fsimilarity = 0;
		return 0;
	}
	//**************************
	cv::Mat probe_img_gray;
	cv::cvtColor(probe_img_color, probe_img_gray, CV_BGR2GRAY);

	ImageData gallery_img_data_color(gallery_img_color.cols, gallery_img_color.rows, gallery_img_color.channels());
	gallery_img_data_color.data = gallery_img_color.data;

	ImageData gallery_img_data_gray(gallery_img_gray.cols, gallery_img_gray.rows, gallery_img_gray.channels());
	gallery_img_data_gray.data = gallery_img_gray.data;

	ImageData probe_img_data_color(probe_img_color.cols, probe_img_color.rows, probe_img_color.channels());
	probe_img_data_color.data = probe_img_color.data;

	ImageData probe_img_data_gray(probe_img_gray.cols, probe_img_gray.rows, probe_img_gray.channels());
	probe_img_data_gray.data = probe_img_gray.data;

	// Detect faces
	std::vector<seeta::FaceInfo> gallery_faces = detector.Detect(gallery_img_data_gray);
	int32_t gallery_face_num = static_cast<int32_t>(gallery_faces.size());

	std::vector<seeta::FaceInfo> probe_faces = detector.Detect(probe_img_data_gray);
	int32_t probe_face_num = static_cast<int32_t>(probe_faces.size());

	if (gallery_face_num == 0)
	{
		strcpy(szErrMsg, "��֤��Ƭ����������");
		fsimilarity = 0;
		return 0;
	}
	if(probe_face_num == 0)
	{
		//std::cout << "Faces are not detected.";
		strcpy(szErrMsg, "������Ƭ����������");
		fsimilarity = 0;
		return 0;
	}

	// Detect 5 facial landmarks
	seeta::FacialLandmark gallery_points[5];
	point_detector.PointDetectLandmarks(gallery_img_data_gray, gallery_faces[0], gallery_points);

	seeta::FacialLandmark probe_points[5];
	point_detector.PointDetectLandmarks(probe_img_data_gray, probe_faces[0], probe_points);

	for (int i = 0; i<5; i++)
	{
		cv::circle(gallery_img_color, cv::Point(gallery_points[i].x, gallery_points[i].y), 2,
			CV_RGB(0, 255, 0));
		cv::circle(probe_img_color, cv::Point(probe_points[i].x, probe_points[i].y), 2,
			CV_RGB(0, 255, 0));
	}
	//cv::imwrite("gallery_point_result.jpg", gallery_img_color);
	//cv::imwrite("probe_point_result.jpg", probe_img_color);

	// Extract face identity feature
	float gallery_fea[2048];
	float probe_fea[2048];
	face_recognizer.ExtractFeatureWithCrop(gallery_img_data_color, gallery_points, gallery_fea);
	face_recognizer.ExtractFeatureWithCrop(probe_img_data_color, probe_points, probe_fea);

	// Caculate similarity of two faces
	float sim = face_recognizer.CalcSimilarity(gallery_fea, probe_fea);
	if (sim > RIM_VALUE)
	{
		strcpy(szErrMsg, "��֤ͨ����");
	}
	else {
		strcpy(szErrMsg, "��֤ʧ�ܡ�");
	}

	fsimilarity = sim;
	//std::cout << "sim " << sim << std::endl;
	return 1;
}
