#pragma once
// SLIC_rgbd.h: interface for the SLIC class.
//===========================================================================
// This code implements the RGB-D superpixel method described in [1],
// which is extended from [2].
//
// [1] Jingfan Guo, Tongwei Ren, and Jia Bei. Salient object detection for 
//     RGB-D image via saliency evolution. ICME, IEEE, 2016.
// [2] Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, 
//     Pascal Fua, and Sabine Susstrunk. SLIC Superpixels. EPFL Technical 
//     Report no. 149300, June 2010.
//
//===========================================================================
//  Edited by Jingfan Guo [MAGUS, NJU] in 2016.
//===========================================================================
//	Copyright (c) 2012 Radhakrishna Achanta [EPFL]. All rights reserved.
//===========================================================================
//////////////////////////////////////////////////////////////////////

#include <vector>
#include <string>
#include <algorithm>
using namespace std;


class SLIC
{
public:
	SLIC();
	virtual ~SLIC();
	//============================================================================
	// Superpixel segmentation for a given step size (superpixel size ~= step*step)
	//============================================================================
	void DoSuperpixelSegmentation_ForGivenSuperpixelSize(
		const unsigned char*                            ubuff,//Each 32 bit unsigned int contains ARGB pixel values.
		const int					width,
		const int					height,
		int*&						klabels,
		int&						numlabels,
		const int&					superpixelsize,
		const double&                                   compactness);
	//============================================================================
	// Superpixel segmentation for a given number of superpixels
	//============================================================================
	void DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(
		const unsigned char*                             ubuff,
		const unsigned char*                             ubuff_d,
		const int					width,
		const int					height,
		int*&						klabels,
		int&						numlabels,
		const int&					K,//required number of superpixels
		const double&                                   compactness);//10-20 is a good value for CIELAB space
																	 //============================================================================
																	 // Save superpixel labels in a text file in raster scan order
																	 //============================================================================
	void SaveSuperpixelLabels(
		const int*&					labels,
		const int&					width,
		const int&					height,
		const string&				filename,
		const string&				path);
	//============================================================================
	// Function to draw boundaries around superpixels of a given 'color'.
	// Can also be used to draw boundaries around supervoxels, i.e layer by layer.
	//============================================================================
	void DrawContoursAroundSegments(
		unsigned char*				segmentedImage,
		int*&						labels,
		const int&					width,
		const int&					height,
		const unsigned int&			color);

private:
	//============================================================================
	// The main SLIC algorithm for generating superpixels
	//============================================================================
	void PerformSuperpixelSLIC(
		vector<double>&				kseedsl,
		vector<double>&				kseedsa,
		vector<double>&				kseedsb,
		vector<double>&				kseedsx,
		vector<double>&				kseedsy,
		vector<double>&				kseedsd,
		int*&						klabels,
		const int&					STEP,
		const vector<double>&		edgemag,
		const double&				m = 10.0);
	//============================================================================
	// Pick seeds for superpixels when step size of superpixels is given.
	//============================================================================
	void GetLABXYDSeeds_ForGivenStepSize(
		vector<double>&				kseedsl,
		vector<double>&				kseedsa,
		vector<double>&				kseedsb,
		vector<double>&				kseedsx,
		vector<double>&				kseedsy,
		vector<double>&				kseedsd,
		const int&					STEP,
		const bool&					perturbseeds,
		const vector<double>&		edgemag);
	//============================================================================
	// Move the superpixel seeds to low gradient positions to avoid putting seeds
	// at region boundaries.
	//============================================================================
	void PerturbSeeds(
		vector<double>&				kseedsl,
		vector<double>&				kseedsa,
		vector<double>&				kseedsb,
		vector<double>&				kseedsx,
		vector<double>&				kseedsy,
		vector<double>&				kseedsd,
		const vector<double>&		edges);
	//============================================================================
	// Detect color edges, to help PerturbSeeds()
	//============================================================================
	void DetectLabEdges(
		const double*				lvec,
		const double*				avec,
		const double*				bvec,
		const int&					width,
		const int&					height,
		vector<double>&				edges);
	//============================================================================
	// sRGB to XYZ conversion; helper for RGB2LAB()
	//============================================================================
	void RGB2XYZ(
		const int&					sR,
		const int&					sG,
		const int&					sB,
		double&						X,
		double&						Y,
		double&						Z);
	//============================================================================
	// sRGB to CIELAB conversion (uses RGB2XYZ function)
	//============================================================================
	void RGB2LAB(
		const int&					sR,
		const int&					sG,
		const int&					sB,
		double&						lval,
		double&						aval,
		double&						bval);
	//============================================================================
	// sRGB to CIELAB conversion for 2-D images
	//============================================================================
	void DoRGBtoLABConversion(
		const unsigned char*		ubuff,
		double*&					lvec,
		double*&					avec,
		double*&					bvec);
	//============================================================================
	// sRGB to CIELAB conversion for 3-D volumes
	//============================================================================
	void DoRGBtoLABConversion(
		unsigned int**&				ubuff,
		double**&					lvec,
		double**&					avec,
		double**&					bvec);
	//============================================================================
	// Post-processing of SLIC segmentation, to avoid stray labels.
	//============================================================================
	void EnforceLabelConnectivity(
		const int*					labels,
		const int					width,
		const int					height,
		int*&						nlabels,//input labels that need to be corrected to remove stray labels
		int&						numlabels,//the number of labels changes in the end if segments are removed
		const int&					K); //the number of superpixels desired by the user

private:
	int										m_width;
	int										m_height;

	double*                                 m_dvec;

	double*									m_lvec;
	double*									m_avec;
	double*									m_bvec;
};
