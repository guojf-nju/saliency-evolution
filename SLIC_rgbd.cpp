// SLIC_rgbd.cpp: implementation for the SLIC class.
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

#include <cfloat>
#include <cmath>
#include <iostream>
#include <fstream>
#include "SLIC_rgbd.h"
#include "mex.h"

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

SLIC::SLIC()
{
	m_lvec = NULL;
	m_avec = NULL;
	m_bvec = NULL;
	m_dvec = NULL;
}

SLIC::~SLIC()
{
	if (m_lvec) delete[] m_lvec;
	if (m_avec) delete[] m_avec;
	if (m_bvec) delete[] m_bvec;
	if (m_dvec) delete[] m_dvec;
}

//==============================================================================
///	RGB2XYZ
///
/// sRGB (D65 illuninant assumption) to XYZ conversion
//==============================================================================
void SLIC::RGB2XYZ(
	const int&		sR,
	const int&		sG,
	const int&		sB,
	double&			X,
	double&			Y,
	double&			Z)
{
	double R = sR / 255.0;
	double G = sG / 255.0;
	double B = sB / 255.0;

	double r, g, b;

	if (R <= 0.04045)	r = R / 12.92;
	else				r = pow((R + 0.055) / 1.055, 2.4);
	if (G <= 0.04045)	g = G / 12.92;
	else				g = pow((G + 0.055) / 1.055, 2.4);
	if (B <= 0.04045)	b = B / 12.92;
	else				b = pow((B + 0.055) / 1.055, 2.4);

	X = r*0.4124564 + g*0.3575761 + b*0.1804375;
	Y = r*0.2126729 + g*0.7151522 + b*0.0721750;
	Z = r*0.0193339 + g*0.1191920 + b*0.9503041;
}

//===========================================================================
///	RGB2LAB
//===========================================================================
void SLIC::RGB2LAB(const int& sR, const int& sG, const int& sB, double& lval, double& aval, double& bval)
{
	//------------------------
	// sRGB to XYZ conversion
	//------------------------
	double X, Y, Z;
	RGB2XYZ(sR, sG, sB, X, Y, Z);

	//------------------------
	// XYZ to LAB conversion
	//------------------------
	double epsilon = 0.008856;	//actual CIE standard
	double kappa = 903.3;		//actual CIE standard

	double Xr = 0.950456;	//reference white
	double Yr = 1.0;		//reference white
	double Zr = 1.088754;	//reference white

	double xr = X / Xr;
	double yr = Y / Yr;
	double zr = Z / Zr;

	double fx, fy, fz;
	if (xr > epsilon)	fx = pow(xr, 1.0 / 3.0);
	else				fx = (kappa*xr + 16.0) / 116.0;
	if (yr > epsilon)	fy = pow(yr, 1.0 / 3.0);
	else				fy = (kappa*yr + 16.0) / 116.0;
	if (zr > epsilon)	fz = pow(zr, 1.0 / 3.0);
	else				fz = (kappa*zr + 16.0) / 116.0;

	lval = 116.0*fy - 16.0;
	aval = 500.0*(fx - fy);
	bval = 200.0*(fy - fz);
}

//===========================================================================
///	DoRGBtoLABConversion
///
///	For whole image: overlaoded floating point version
//===========================================================================
void SLIC::DoRGBtoLABConversion(
	/*const unsigned int*&		ubuff,*/
	const unsigned char *		ubuff,
	double*&					lvec,
	double*&					avec,
	double*&					bvec)
{
	int sz = m_width*m_height;
	lvec = new double[sz];
	avec = new double[sz];
	bvec = new double[sz];

	for (int j = 0; j < sz; j++)
	{
		int r = (ubuff[j] >> 16) & 0xFF;
		int g = (ubuff[j] >>  8) & 0xFF;
		int b = (ubuff[j]      ) & 0xFF;

		RGB2LAB(r, g, b, lvec[j], avec[j], bvec[j]);
	}
}

//=================================================================================
/// DrawContoursAroundSegments
///
/// Internal contour drawing option exists. One only needs to comment the if
/// statement inside the loop that looks at neighbourhood.
//=================================================================================
void SLIC::DrawContoursAroundSegments(
	unsigned char *			ubuff,
	int*&					labels,
	const int&				width,
	const int&				height,
	const unsigned int&				color)
{
	const int dx8[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };


	int sz = width*height;
	vector<bool> istaken(sz, false);
	vector<int> contourx(sz); vector<int> contoury(sz);
	int mainindex(0); int cind(0);
	for (int j = 0; j < height; j++)
	{
		for (int k = 0; k < width; k++)
		{
			int np(0);
			for (int i = 0; i < 8; i++)
			{
				int x = k + dx8[i];
				int y = j + dy8[i];

				if ((x >= 0 && x < width) && (y >= 0 && y < height))
				{
					int index = y*width + x;

					//if( false == istaken[index] )//comment this to obtain internal contours
					{
						if (labels[mainindex] != labels[index]) np++;
					}
				}
			}
			if (np > 1)
			{
				contourx[cind] = k;
				contoury[cind] = j;
				istaken[mainindex] = true;
				//img[mainindex] = color;
				cind++;
			}
			mainindex++;
		}
	}

	int numboundpix = cind;//int(contourx.size());
	for (int j = 0; j < numboundpix; j++)
	{
		int ii = contoury[j] * width + contourx[j];
		ubuff[3 * ii] = 0xff;
		ubuff[3 * ii + 1] = 0xff;
		ubuff[3 * ii + 2] = 0xff;

		for (int n = 0; n < 8; n++)
		{
			int x = contourx[j] + dx8[n];
			int y = contoury[j] + dy8[n];
			if ((x >= 0 && x < width) && (y >= 0 && y < height))
			{
				int ind = y*width + x;
				if (!istaken[ind])
				{
					ubuff[3 * ind] = 0;
					ubuff[3 * ind + 1] = 0;
					ubuff[3 * ind + 2] = 0;
				}
			}
		}
	}
}


//==============================================================================
///	DetectLabEdges
//==============================================================================
void SLIC::DetectLabEdges(
	const double*				lvec,
	const double*				avec,
	const double*				bvec,
	const int&					width,
	const int&					height,
	vector<double>&				edges)
{
	int sz = width*height;

	edges.resize(sz, 0);
	for (int j = 1; j < height - 1; j++)
	{
		for (int k = 1; k < width - 1; k++)
		{
			int i = j*width + k;

			double dx = (lvec[i - 1] - lvec[i + 1])*(lvec[i - 1] - lvec[i + 1]) +
				(avec[i - 1] - avec[i + 1])*(avec[i - 1] - avec[i + 1]) +
				(bvec[i - 1] - bvec[i + 1])*(bvec[i - 1] - bvec[i + 1]);

			double dy = (lvec[i - width] - lvec[i + width])*(lvec[i - width] - lvec[i + width]) +
				(avec[i - width] - avec[i + width])*(avec[i - width] - avec[i + width]) +
				(bvec[i - width] - bvec[i + width])*(bvec[i - width] - bvec[i + width]);

			//edges[i] = fabs(dx) + fabs(dy);
			edges[i] = dx*dx + dy*dy;
		}
	}
}

//===========================================================================
///	PerturbSeeds
//===========================================================================
void SLIC::PerturbSeeds(
	vector<double>&				kseedsl,
	vector<double>&				kseedsa,
	vector<double>&				kseedsb,
	vector<double>&				kseedsx,
	vector<double>&				kseedsy,
	vector<double>&				kseedsd,
	const vector<double>&                   edges)
{
	const int dx8[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };

	int numseeds = kseedsl.size();

	for (int n = 0; n < numseeds; n++)
	{
		int ox = (int)kseedsx[n];//original x
		int oy = (int)kseedsy[n];//original y
		int oind = oy*m_width + ox;

		int storeind = oind;
		for (int i = 0; i < 8; i++)
		{
			int nx = ox + dx8[i];//new x
			int ny = oy + dy8[i];//new y

			if (nx >= 0 && nx < m_width && ny >= 0 && ny < m_height)
			{
				int nind = ny*m_width + nx;
				if (edges[nind] < edges[storeind])
				{
					storeind = nind;
				}
			}
		}
		if (storeind != oind)
		{
			kseedsx[n] = storeind%m_width;
			kseedsy[n] = storeind / m_width;
			kseedsd[n] = m_dvec[storeind];
			kseedsl[n] = m_lvec[storeind];
			kseedsa[n] = m_avec[storeind];
			kseedsb[n] = m_bvec[storeind];
		}
	}
}


//===========================================================================
///	GetLABXYSeeds_ForGivenStepSize
///
/// The k seed values are taken as uniform spatial pixel samples.
//===========================================================================
void SLIC::GetLABXYDSeeds_ForGivenStepSize(
	vector<double>&				kseedsl,
	vector<double>&				kseedsa,
	vector<double>&				kseedsb,
	vector<double>&				kseedsx,
	vector<double>&				kseedsy,
	vector<double>&				kseedsd,
	const int&					STEP,
	const bool&					perturbseeds,
	const vector<double>&       edgemag)
{
	const bool hexgrid = false;
	int numseeds(0);
	int n(0);

	//int xstrips = m_width/STEP;
	//int ystrips = m_height/STEP;
	int xstrips = (int)(0.5 + double(m_width) / double(STEP));
	int ystrips = (int)(0.5 + double(m_height) / double(STEP));

	int xerr = m_width - STEP*xstrips; if (xerr < 0) { xstrips--; xerr = m_width - STEP*xstrips; }
	int yerr = m_height - STEP*ystrips; if (yerr < 0) { ystrips--; yerr = m_height - STEP*ystrips; }

	double xerrperstrip = double(xerr) / double(xstrips);
	double yerrperstrip = double(yerr) / double(ystrips);

	int xoff = STEP / 2;
	int yoff = STEP / 2;
	//-------------------------
	numseeds = xstrips*ystrips;
	//-------------------------
	kseedsl.resize(numseeds);
	kseedsa.resize(numseeds);
	kseedsb.resize(numseeds);
	kseedsx.resize(numseeds);
	kseedsy.resize(numseeds);
	kseedsd.resize(numseeds);

	for (int y = 0; y < ystrips; y++)
	{
		int ye = int(y*yerrperstrip);
		for (int x = 0; x < xstrips; x++)
		{
			int xe = int(x*xerrperstrip);
			int seedx = (x*STEP + xoff + xe);
			if (hexgrid) { seedx = x*STEP + (xoff << (y & 0x1)) + xe; seedx = min(m_width - 1, seedx); }//for hex grid sampling
			int seedy = (y*STEP + yoff + ye);
			int i = seedy*m_width + seedx;

			kseedsl[n] = m_lvec[i];
			kseedsa[n] = m_avec[i];
			kseedsb[n] = m_bvec[i];
			kseedsx[n] = seedx;
			kseedsy[n] = seedy;
			kseedsd[n] = m_dvec[i];
			n++;
		}
	}


	if (perturbseeds)
	{
		PerturbSeeds(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, kseedsd, edgemag);
	}
}

//===========================================================================
///	PerformSuperpixelSLIC
///
///	Performs k mean segmentation. It is fast because it looks locally, not
/// over the entire image.
//===========================================================================
void SLIC::PerformSuperpixelSLIC(
	vector<double>&				kseedsl,
	vector<double>&				kseedsa,
	vector<double>&				kseedsb,
	vector<double>&				kseedsx,
	vector<double>&				kseedsy,
	vector<double>&             kseedsd,
	int*&					klabels,
	const int&				STEP,
	const vector<double>&                   edgemag,
	const double&				M)
{
	int sz = m_width*m_height;
	const int numk = kseedsl.size();
	//----------------
	int offset = STEP;
	//if(STEP < 8) offset = STEP*1.5;//to prevent a crash due to a very small step size
	//----------------

	vector<double> clustersize(numk, 0);
	vector<double> inv(numk, 0);//to store 1/clustersize[k] values

	vector<double> sigmal(numk, 0);
	vector<double> sigmaa(numk, 0);
	vector<double> sigmab(numk, 0);
	vector<double> sigmax(numk, 0);
	vector<double> sigmay(numk, 0);
	vector<double> sigmad(numk, 0);
	vector<double> distvec(sz, DBL_MAX);

	double invwt = 1.0 / ((STEP / M)*(STEP / M));

	int x1, y1, x2, y2;
	double l, a, b, d;
	double dist;
	double distxyd;
	for (int itr = 0; itr < 10; itr++)
	{
		distvec.assign(sz, DBL_MAX);
		for (int n = 0; n < numk; n++)
		{
			y1 = (int)max(0.0, kseedsy[n] - offset);
			y2 = (int)min((double)m_height, kseedsy[n] + offset);
			x1 = (int)max(0.0, kseedsx[n] - offset);
			x2 = (int)min((double)m_width, kseedsx[n] + offset);


			for (int y = y1; y < y2; y++)
			{
				for (int x = x1; x < x2; x++)
				{
					int i = y*m_width + x;

					l = m_lvec[i];
					a = m_avec[i];
					b = m_bvec[i];
					d = m_dvec[i];
                    
					dist = (l - kseedsl[n])*(l - kseedsl[n]) +
						(a - kseedsa[n])*(a - kseedsa[n]) +
						(b - kseedsb[n])*(b - kseedsb[n]);

					distxyd = (x - kseedsx[n])*(x - kseedsx[n]) +
						(y - kseedsy[n])*(y - kseedsy[n]) +
						(d - kseedsd[n])*(d - kseedsd[n]);

					//------------------------------------------------------------------------
					dist += distxyd*invwt;//dist = sqrt(dist) + sqrt(distxy*invwt);//this is more exact
										  //------------------------------------------------------------------------
					if (dist < distvec[i])
					{
						distvec[i] = dist;
						klabels[i] = n;
					}
				}
			}
		}
		//-----------------------------------------------------------------
		// Recalculate the centroid and store in the seed values
		//-----------------------------------------------------------------
		//instead of reassigning memory on each iteration, just reset.

		sigmal.assign(numk, 0);
		sigmaa.assign(numk, 0);
		sigmab.assign(numk, 0);
		sigmax.assign(numk, 0);
		sigmay.assign(numk, 0);
		sigmad.assign(numk, 0);
		clustersize.assign(numk, 0);
		//------------------------------------
		//edgesum.assign(numk, 0);
		//------------------------------------

		{int ind(0);
		for (int r = 0; r < m_height; r++)
		{
			for (int c = 0; c < m_width; c++)
			{
				sigmal[klabels[ind]] += m_lvec[ind];
				sigmaa[klabels[ind]] += m_avec[ind];
				sigmab[klabels[ind]] += m_bvec[ind];
				sigmax[klabels[ind]] += c;
				sigmay[klabels[ind]] += r;
				sigmad[klabels[ind]] += m_dvec[ind];
				//------------------------------------
				//edgesum[klabels[ind]] += edgemag[ind];
				//------------------------------------
				clustersize[klabels[ind]] += 1.0;
				ind++;
			}
		}}

		{for (int k = 0; k < numk; k++)
		{
			if (clustersize[k] <= 0) clustersize[k] = 1;
			inv[k] = 1.0 / clustersize[k];//computing inverse now to multiply, than divide later
		}}

		{for (int k = 0; k < numk; k++)
		{
			kseedsl[k] = sigmal[k] * inv[k];
			kseedsa[k] = sigmaa[k] * inv[k];
			kseedsb[k] = sigmab[k] * inv[k];
			kseedsx[k] = sigmax[k] * inv[k];
			kseedsy[k] = sigmay[k] * inv[k];
			kseedsd[k] = sigmad[k] * inv[k];
			//------------------------------------
			//edgesum[k] *= inv[k];
			//------------------------------------
		}}
	}
}

//===========================================================================
///	SaveSuperpixelLabels
///
///	Save labels in raster scan order.
//===========================================================================
void SLIC::SaveSuperpixelLabels(
	const int*&					labels,
	const int&					width,
	const int&					height,
	const string&				filename,
	const string&				path)
{
#ifdef WINDOWS
	char fname[256];
	char extn[256];
	_splitpath(filename.c_str(), NULL, NULL, fname, extn);
	string temp = fname;
	string finalpath = path + temp + string(".dat");
#else
	string nameandextension = filename;
	size_t pos = filename.find_last_of("/");
	if (pos != string::npos)//if a slash is found, then take the filename with extension
	{
		nameandextension = filename.substr(pos + 1);
	}
	string newname = nameandextension.replace(nameandextension.rfind(".") + 1, 3, "dat");//find the position of the dot and replace the 3 characters following it.
	string finalpath = path + newname;
#endif

	int sz = width*height;
	ofstream outfile;
	outfile.open(finalpath.c_str(), ios::binary);
	for (int i = 0; i < sz; i++)
	{
		outfile.write((const char*)&labels[i], sizeof(int));
	}
	outfile.close();
}

//===========================================================================
///	EnforceLabelConnectivity
///
///		1. finding an adjacent label for each new component at the start
///		2. if a certain component is too small, assigning the previously found
///		    adjacent label to this component, and not incrementing the label.
//===========================================================================
void SLIC::EnforceLabelConnectivity(
	const int*					labels,//input labels that need to be corrected to remove stray labels
	const int					width,
	const int					height,
	int*&						nlabels,//new labels
	int&						numlabels,//the number of labels changes in the end if segments are removed
	const int&					K) //the number of superpixels desired by the user
{
	//	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	//	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

	const int dx4[4] = { -1, 0, 1, 0 };
	const int dy4[4] = { 0, -1, 0, 1 };

	const int sz = width*height;
	const int SUPSZ = sz / K;
	//nlabels.resize(sz, -1);
	for (int i = 0; i < sz; i++) nlabels[i] = -1;
	int label(0);
	int* xvec = new int[sz];
	int* yvec = new int[sz];
	int oindex(0);
	int adjlabel(0);//adjacent label
	for (int j = 0; j < height; j++)
	{
		for (int k = 0; k < width; k++)
		{
			if (0 > nlabels[oindex])
			{
				nlabels[oindex] = label;
				//--------------------
				// Start a new segment
				//--------------------
				xvec[0] = k;
				yvec[0] = j;
				//-------------------------------------------------------
				// Quickly find an adjacent label for use later if needed
				//-------------------------------------------------------
				{for (int n = 0; n < 4; n++)
				{
					int x = xvec[0] + dx4[n];
					int y = yvec[0] + dy4[n];
					if ((x >= 0 && x < width) && (y >= 0 && y < height))
					{
						int nindex = y*width + x;
						if (nlabels[nindex] >= 0) adjlabel = nlabels[nindex];
					}
				}}

				int count(1);
				for (int c = 0; c < count; c++)
				{
					for (int n = 0; n < 4; n++)
					{
						int x = xvec[c] + dx4[n];
						int y = yvec[c] + dy4[n];

						if ((x >= 0 && x < width) && (y >= 0 && y < height))
						{
							int nindex = y*width + x;

							if (0 > nlabels[nindex] && labels[oindex] == labels[nindex])
							{
								xvec[count] = x;
								yvec[count] = y;
								nlabels[nindex] = label;
								count++;
							}
						}

					}
				}
				//-------------------------------------------------------
				// If segment size is less then a limit, assign an
				// adjacent label found before, and decrement label count.
				//-------------------------------------------------------
				if (count <= SUPSZ >> 2)
				{
					for (int c = 0; c < count; c++)
					{
						int ind = yvec[c] * width + xvec[c];
						nlabels[ind] = adjlabel;
					}
					label--;
				}
				label++;
			}
			oindex++;
		}
	}
	numlabels = label;

	if (xvec) delete[] xvec;
	if (yvec) delete[] yvec;
}

//===========================================================================
///	DoSuperpixelSegmentation_ForGivenSuperpixelSize
///
/// The input parameter ubuff conains RGB values in a 32-bit unsigned integers
/// as follows:
///
/// [1 1 1 1 1 1 1 1]  [1 1 1 1 1 1 1 1]  [1 1 1 1 1 1 1 1]  [1 1 1 1 1 1 1 1]
///
///        Nothing              R                 G                  B
///
/// The RGB values are accessed from (and packed into) the unsigned integers
/// using bitwise operators as can be seen in the function DoRGBtoLABConversion().
///
/// compactness value depends on the input pixels values. For instance, if
/// the input is greyscale with values ranging from 0-100, then a compactness
/// value of 20.0 would give good results. A greater value will make the
/// superpixels more compact while a smaller value would make them more uneven.
///
/// The labels can be saved if needed using SaveSuperpixelLabels()
//===========================================================================
void SLIC::DoSuperpixelSegmentation_ForGivenSuperpixelSize(
	const unsigned char*        ubuff,
	const int					width,
	const int					height,
	int*&						klabels,
	int&						numlabels,
	const int&					superpixelsize,
	const double&               compactness)
{
	//------------------------------------------------
	const int STEP = int(sqrt(double(superpixelsize)) + 0.5);
	//------------------------------------------------
	vector<double> kseedsl(0);
	vector<double> kseedsa(0);
	vector<double> kseedsb(0);
	vector<double> kseedsx(0);
	vector<double> kseedsy(0);
	vector<double> kseedsd(0);

	//--------------------------------------------------
	m_width = width;
	m_height = height;
	int sz = m_width*m_height;
	//klabels.resize( sz, -1 );
	//--------------------------------------------------
	//klabels = new int[sz];
	for (int s = 0; s < sz; s++) klabels[s] = -1;
	//--------------------------------------------------
	if (1)//LAB, the default option
	{
		DoRGBtoLABConversion(ubuff, m_lvec, m_avec, m_bvec);
	}
	else//RGB
	{
		m_lvec = new double[sz]; m_avec = new double[sz]; m_bvec = new double[sz];
		for (int i = 0; i < sz; i++)
		{
			m_lvec[i] = ubuff[i] >> 16 & 0xff;
			m_avec[i] = ubuff[i] >>  8 & 0xff;
			m_bvec[i] = ubuff[i]       & 0xff;
		}
	}
	//--------------------------------------------------
	bool perturbseeds(false);//perturb seeds is not absolutely necessary, one can set this flag to false
	vector<double> edgemag(0);
	if (perturbseeds) DetectLabEdges(m_lvec, m_avec, m_bvec, m_width, m_height, edgemag);

	GetLABXYDSeeds_ForGivenStepSize(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, kseedsd, STEP, perturbseeds, edgemag);

	PerformSuperpixelSLIC(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, kseedsd, klabels, STEP, edgemag, compactness);
	numlabels = kseedsl.size();

	int* nlabels = new int[sz];
	EnforceLabelConnectivity(klabels, m_width, m_height, nlabels, numlabels, int(double(sz) / double(STEP*STEP)));
	{for (int i = 0; i < sz; i++) klabels[i] = nlabels[i]; }
	if (nlabels) delete[] nlabels;
}

//===========================================================================
///	DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels
///
/// The input parameter ubuff conains RGB values in a 32-bit unsigned integers
/// as follows:
///
/// [1 1 1 1 1 1 1 1]  [1 1 1 1 1 1 1 1]  [1 1 1 1 1 1 1 1]  [1 1 1 1 1 1 1 1]
///
///        Nothing              R                 G                  B
///
/// The RGB values are accessed from (and packed into) the unsigned integers
/// using bitwise operators as can be seen in the function DoRGBtoLABConversion().
///
/// compactness value depends on the input pixels values. For instance, if
/// the input is greyscale with values ranging from 0-100, then a compactness
/// value of 20.0 would give good results. A greater value will make the
/// superpixels more compact while a smaller value would make them more uneven.
///
/// The labels can be saved if needed using SaveSuperpixelLabels()
//===========================================================================
void SLIC::DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(
	const unsigned char*        ubuff,
	const unsigned char*        ubuff_d,
	const int					width,
	const int					height,
	int*&						klabels,
	int&						numlabels,
	const int&					K,//required number of superpixels
	const double&                                   compactness)//weight given to spatial distance
{
	int sz = width*height;
	m_dvec = new double[sz];
	for (int j = 0; j < sz; j++)
	{
		m_dvec[j] = ubuff_d[j];
	}
	const int superpixelsize = int(0.5 + double(width*height) / double(K));
	DoSuperpixelSegmentation_ForGivenSuperpixelSize(ubuff, width, height, klabels, numlabels, superpixelsize, compactness);
}

//===========================================================================
/// Entry point to C/C++ MEX-file
//===========================================================================
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    unsigned char *img, *dep;
    size_t height, width, dHeight, dWidth, sz;
    int expectSpNum;
    
    if(nrhs != 3)
        mexErrMsgIdAndTxt( "MATLAB:SLIC_rgbd:invalidNumInputs",
              "Three inputs required.");
    if(nlhs > 1)
        mexErrMsgIdAndTxt( "MATLAB:SLIC_rgbd:maxlhs",
              "Too many output arguments.");
        
    height = mxGetM(prhs[0]);
    width = mxGetN(prhs[0]) / 3;
    sz = height * width;
    
    dHeight = mxGetM(prhs[1]);
    dWidth = mxGetN(prhs[1]);
    
    if(height != dHeight || width != dWidth)
        mexErrMsgIdAndTxt( "MATLAB:SLIC_rgbd:invalidImageSize",
              "RGB image and depth image should have the same size.");
    
    img = (unsigned char*) mxGetData(prhs[0]);
    dep = (unsigned char*) mxGetData(prhs[1]);
    expectSpNum = (int) mxGetScalar(prhs[2]);
    
    unsigned char *imgData = new unsigned char[sz*3];
    unsigned char *depData = new unsigned char[sz];
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            int mIdx = j * height + i; // Matlab column-major index
            int cIdx = i * width + j; // C/C++ row-major index
            unsigned char r = img[mIdx];
            unsigned char g = img[sz + mIdx];
            unsigned char b = img[2 * sz + mIdx];
            imgData[cIdx] = (r << 16) | (g << 8) | b;
            depData[cIdx] = dep[mIdx];
        }
    }
    
    int* labels = new int[sz];
    int actualSpNum(0);
    SLIC slic;
    slic.DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(imgData, depData, width, height, labels, actualSpNum, expectSpNum, 20.0);

    plhs[0] = mxCreateDoubleMatrix(height, width, mxREAL);
    double *output = mxGetPr(plhs[0]);
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            int mIdx = j * height + i; // Matlab column-major index
            int cIdx = i * width + j; // C/C++ row-major index
            output[mIdx] = (double) labels[cIdx] + 1;
        }
    }
}