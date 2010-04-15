/* Redistribution and use in source and binary forms, with or
 * without modification, are permitted provided that the following
 * conditions are met:
 * 	Redistributions of source code must retain the above
 * 	copyright notice, this list of conditions and the following
 * 	disclaimer.
 * 	Redistributions in binary form must reproduce the above
 * 	copyright notice, this list of conditions and the following
 * 	disclaimer in the documentation and/or other materials
 * 	provided with the distribution.
 * 	The name of Contributor may not be used to endorse or
 * 	promote products derived from this software without
 * 	specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
 * TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 * Copyright© 2009, Liu Liu All rights reserved.
 * 
 * OpenCV functions for MSER extraction
 * 
 * 1. there are two different implementation of MSER, one for grey image, one for color image
 * 2. the grey image algorithm is taken from: Linear Time Maximally Stable Extremal Regions;
 *    the paper claims to be faster than union-find method;
 *    it actually get 1.5~2m/s on my centrino L7200 1.2GHz laptop.
 * 3. the color image algorithm is taken from: Maximally Stable Colour Regions for Recognition and Match;
 *    it should be much slower than grey image method ( 3~4 times );
 *    the chi_table.h file is taken directly from paper's source code which is distributed under GPL.
 * 4. though the name is *contours*, the result actually is a list of point set.
 */

#include "cvmser.h"
#include "cxmisc.h"

typedef struct CvLinkedPoint
{
	struct CvLinkedPoint* prev;
	struct CvLinkedPoint* next;
	CvPoint pt;
}
CvLinkedPoint;

// the history of region grown
typedef struct CvMSERGrowHistory
{
	struct CvMSERGrowHistory* shortcut;
	struct CvMSERGrowHistory* child;
	int stable; // when it ever stabled before, record the size
	int val;
	int size;
}
CvMSERGrowHistory;

typedef struct CvMSERConnectedComp
{
	CvLinkedPoint* head;
	CvLinkedPoint* tail;
	CvMSERGrowHistory* history;
	unsigned long grey_level;
	int size;
	int dvar; // the derivative of last var
	float var; // the current variation (most time is the variation of one-step back)
}
CvMSERConnectedComp;

// Linear Time MSER claims by using bsf can get performance gain, here is the implementation
// however it seems that will not do any good in real world test
#if 0
#ifdef __GNUC__
inline unsigned char _BitScanForward(unsigned long * Index, unsigned long Mask)
{
	unsigned int EFlags = 0;
	__asm__ ( "bsf %[Mask], %[Index];"
		  "pushf;"
		  "pop %[EFlags];"
		  : [Index]"=r"(*Index), [EFlags]"=r"(EFlags)
		  : [Mask]"r"(Mask) );
	return !(EFlags & 0x40);
}
#define __INTRIN_ENABLED__
#elif _MSC_BUILD
#include <intrin.h>
#define __INTRIN_ENABLED__
#endif
#ifdef __INTRIN_ENABLED__
inline void _bitset(unsigned long * a, unsigned long b)
{
	*a |= 1<<b;
}
inline void _bitreset(unsigned long * a, unsigned long b)
{
	*a &= ~(1<<b);
}
#endif
#endif

CvMSERParams cvMSERParams( int delta, int min_area, int max_area, float max_variation, float min_diversity, int max_evolution, double area_threshold, double min_margin, int edge_blur_size )
{
	CvMSERParams params;
	params.delta = delta;
	params.min_area = min_area;
	params.max_area = max_area;
	params.max_variation = max_variation;
	params.min_diversity = min_diversity;
	params.max_evolution = max_evolution;
	params.area_threshold = area_threshold;
	params.min_margin = min_margin;
	params.edge_blur_size = edge_blur_size;
	return params;
}

// clear the connected component in stack
CV_INLINE static void
icvInitMSERComp( CvMSERConnectedComp* comp )
{
	comp->size = 0;
	comp->var = 0;
	comp->dvar = 1;
	comp->history = NULL;
}

// add history of size to a connected component
CV_INLINE static void
icvMSERNewHistory( CvMSERConnectedComp* comp,
		   CvMSERGrowHistory* history )
{
	history->child = history;
	if ( NULL == comp->history )
	{
		history->shortcut = history;
		history->stable = 0;
	} else {
		comp->history->child = history;
		history->shortcut = comp->history->shortcut;
		history->stable = comp->history->stable;
	}
	history->val = comp->grey_level;
	history->size = comp->size;
	comp->history = history;
}

// merging two connected component
CV_INLINE static void
icvMSERMergeComp( CvMSERConnectedComp* comp1,
		  CvMSERConnectedComp* comp2,
		  CvMSERConnectedComp* comp,
		  CvMSERGrowHistory* history )
{
	CvLinkedPoint* head;
	CvLinkedPoint* tail;
	comp->grey_level = comp2->grey_level;
	history->child = history;
	// select the winner by size
	if ( comp1->size >= comp2->size )
	{
		if ( NULL == comp1->history )
		{
			history->shortcut = history;
			history->stable = 0;
		} else {
			comp1->history->child = history;
			history->shortcut = comp1->history->shortcut;
			history->stable = comp1->history->stable;
		}
		if ( NULL != comp2->history && comp2->history->stable > history->stable )
			history->stable = comp2->history->stable;
		history->val = comp1->grey_level;
		history->size = comp1->size;
		// put comp1 to history
		comp->var = comp1->var;
		comp->dvar = comp1->dvar;
		if ( comp1->size > 0 && comp2->size > 0 )
		{
			comp1->tail->next = comp2->head;
			comp2->head->prev = comp1->tail;
		}
		head = ( comp1->size > 0 ) ? comp1->head : comp2->head;
		tail = ( comp2->size > 0 ) ? comp2->tail : comp1->tail;
		// always made the newly added in the last of the pixel list (comp1 ... comp2)
	} else {
		if ( NULL == comp2->history )
		{
			history->shortcut = history;
			history->stable = 0;
		} else {
			comp2->history->child = history;
			history->shortcut = comp2->history->shortcut;
			history->stable = comp2->history->stable;
		}
		if ( NULL != comp1->history && comp1->history->stable > history->stable )
			history->stable = comp1->history->stable;
		history->val = comp2->grey_level;
		history->size = comp2->size;
		// put comp2 to history
		comp->var = comp2->var;
		comp->dvar = comp2->dvar;
		if ( comp1->size > 0 && comp2->size > 0 )
		{
			comp2->tail->next = comp1->head;
			comp1->head->prev = comp2->tail;
		}
		head = ( comp2->size > 0 ) ? comp2->head : comp1->head;
		tail = ( comp1->size > 0 ) ? comp1->tail : comp2->tail;
		// always made the newly added in the last of the pixel list (comp2 ... comp1)
	}
	comp->head = head;
	comp->tail = tail;
	comp->history = history;
	comp->size = comp1->size + comp2->size;
}

CV_INLINE static float
icvMSERVariationCalc( CvMSERConnectedComp* comp,
		      int delta )
{
	CvMSERGrowHistory* history = comp->history;
	int val = comp->grey_level;
	if ( NULL != history )
	{
		CvMSERGrowHistory* shortcut = history->shortcut;
		while ( shortcut != shortcut->shortcut && shortcut->val + delta > val )
			shortcut = shortcut->shortcut;
		CvMSERGrowHistory* child = shortcut->child;
		while ( child != child->child && child->val + delta <= val )
		{
			shortcut = child;
			child = child->child;
		}
		// get the position of history where the shortcut->val <= delta+val and shortcut->child->val >= delta+val
		history->shortcut = shortcut;
		return (float)(comp->size-shortcut->size)/(float)shortcut->size;
		// here is a small modification of MSER where cal ||R_{i}-R_{i-delta}||/||R_{i-delta}||
		// in standard MSER, cal ||R_{i+delta}-R_{i-delta}||/||R_{i}||
		// my calculation is simpler and much easier to implement
	}
	return 1.;
}

CV_INLINE static bool
icvMSERStableCheck( CvMSERConnectedComp* comp,
		    CvMSERParams params )
{
	// tricky part: it actually check the stablity of one-step back
	if ( comp->history == NULL || comp->history->size <= params.min_area || comp->history->size >= params.max_area )
		return 0;
	float div = (float)(comp->history->size-comp->history->stable)/(float)comp->history->size;
	float var = icvMSERVariationCalc( comp, params.delta );
	int dvar = ( comp->var < var || comp->history->val + 1 < comp->grey_level );
	int stable = ( dvar && !comp->dvar && comp->var < params.max_variation && div > params.min_diversity );
	comp->var = var;
	comp->dvar = dvar;
	if ( stable )
		comp->history->stable = comp->history->size;
	return stable;
}

// add a pixel to the pixel list
CV_INLINE static void
icvAccumulateMSERComp( CvMSERConnectedComp* comp,
		       CvLinkedPoint* point )
{
	if ( comp->size > 0 )
	{
		point->prev = comp->tail;
		comp->tail->next = point;
		point->next = NULL;
	} else {
		point->prev = NULL;
		point->next = NULL;
		comp->head = point;
	}
	comp->tail = point;
	comp->size++;
}

// convert the point set to CvSeq
CV_INLINE static CvContour*
icvMSERToContour( CvMSERConnectedComp* comp,
		  CvMemStorage* storage )
{
	CvSeq* contour = cvCreateSeq( CV_SEQ_KIND_GENERIC|CV_32SC2, sizeof(CvContour), sizeof(CvPoint), storage );
	cvSeqPushMulti( contour, 0, comp->history->size );
	CvLinkedPoint* lpt = comp->head;
	for ( int i = 0; i < comp->history->size; i++ )
	{
		CvPoint* pt = CV_GET_SEQ_ELEM( CvPoint, contour, i );
		pt->x = lpt->pt.x;
		pt->y = lpt->pt.y;
		lpt = lpt->next;
	}
	return (CvContour*)contour;
}

// to preprocess src image to following format
// 32-bit image
// > 0 is available, < 0 is visited
// 17~19 bits is the direction
// 8~11 bits is the bucket it falls to (for BitScanForward)
// 0~8 bits is the color
static void
icvPreprocessMSER_8UC1( CvMat* src,
			CvMat* img,
			int*** heap_cur )
{
	int srccpt = src->step-src->cols;
	int cpt_1 = img->cols-src->cols-1;
	int* imgptr = img->data.i;

	int level_size[256];
	for ( int i = 0; i < 256; i++ )
		level_size[i] = 0;

	for ( int i = 0; i < src->cols+2; i++ )
	{
		*imgptr = -1;
		imgptr++;
	}
	imgptr += cpt_1-1;
	uchar* srcptr = src->data.ptr;
	for ( int i = 0; i < src->rows; i++ )
	{
		*imgptr = -1;
		imgptr++;
		for ( int j = 0; j < src->cols; j++ )
		{
			*srcptr = 0xff-*srcptr;
			level_size[*srcptr]++;
			*imgptr = ((*srcptr>>5)<<8)|(*srcptr);
			imgptr++;
			srcptr++;
		}
		*imgptr = -1;
		imgptr += cpt_1;
		srcptr += srccpt;
	}
	for ( int i = 0; i < src->cols+2; i++ )
	{
		*imgptr = -1;
		imgptr++;
	}

	heap_cur[0][0] = 0;
	for ( int i = 1; i < 256; i++ )
	{
		heap_cur[i] = heap_cur[i-1]+level_size[i-1]+1;
		heap_cur[i][0] = 0;
	}
}

static void
icvExtractMSER_8UC1_Pass( int* imgptr,
			  int*** heap_cur,
			  CvLinkedPoint* ptsptr,
			  CvMSERGrowHistory* histptr,
			  CvMSERConnectedComp* comptr,
			  int step,
			  int stepmask,
			  int stepgap,
			  CvMSERParams params,
			  int color,
			  CvSeq* contours,
			  CvMemStorage* storage )
{
	int* ioptr = imgptr;
	comptr->grey_level = 256;
	comptr++;
	comptr->grey_level = (*imgptr)&0xff;
	icvInitMSERComp( comptr );
	*imgptr |= 0x80000000;
	heap_cur += (*imgptr)&0xff;
	int dir[] = { 1, step, -1, -step };
#ifdef __INTRIN_ENABLED__
	unsigned long heapbit[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	unsigned long* bit_cur = heapbit+(((*imgptr)&0x700)>>8);
#endif
	for ( ; ; )
	{
		// take tour of all the 4 directions
		while ( ((*imgptr)&0x70000) < 0x40000 )
		{
			// get the neighbor
			int* imgptr_nbr = imgptr+dir[((*imgptr)&0x70000)>>16];
			if ( *imgptr_nbr >= 0 ) // if the neighbor is not visited yet
			{
				*imgptr_nbr |= 0x80000000; // mark it as visited
				if ( ((*imgptr_nbr)&0xff) < ((*imgptr)&0xff) )
				{
					// when the value of neighbor smaller than current
					// push current to boundary heap and make the neighbor to be the current one
					// create an empty comp
					(*heap_cur)++;
					**heap_cur = imgptr;
					*imgptr += 0x10000;
					heap_cur += ((*imgptr_nbr)&0xff)-((*imgptr)&0xff);
#ifdef __INTRIN_ENABLED__
					_bitset( bit_cur, (*imgptr)&0x1f );
					bit_cur += (((*imgptr_nbr)&0x700)-((*imgptr)&0x700))>>8;
#endif
					imgptr = imgptr_nbr;
					comptr++;
					icvInitMSERComp( comptr );
					comptr->grey_level = (*imgptr)&0xff;
					continue;
				} else {
					// otherwise, push the neighbor to boundary heap
					heap_cur[((*imgptr_nbr)&0xff)-((*imgptr)&0xff)]++;
					*heap_cur[((*imgptr_nbr)&0xff)-((*imgptr)&0xff)] = imgptr_nbr;
#ifdef __INTRIN_ENABLED__
					_bitset( bit_cur+((((*imgptr_nbr)&0x700)-((*imgptr)&0x700))>>8), (*imgptr_nbr)&0x1f );
#endif
				}
			}
			*imgptr += 0x10000;
		}
		int i = imgptr-ioptr;
		ptsptr->pt = cvPoint( i&stepmask, i>>stepgap );
		// get the current location
		icvAccumulateMSERComp( comptr, ptsptr );
		ptsptr++;
		// get the next pixel from boundary heap
		if ( **heap_cur )
		{
			imgptr = **heap_cur;
			(*heap_cur)--;
#ifdef __INTRIN_ENABLED__
			if ( !**heap_cur )
				_bitreset( bit_cur, (*imgptr)&0x1f );
#endif
		} else {
#ifdef __INTRIN_ENABLED__
			bool found_pixel = 0;
			unsigned long pixel_val;
			for ( int i = ((*imgptr)&0x700)>>8; i < 8; i++ )
			{
				if ( _BitScanForward( &pixel_val, *bit_cur ) )
				{
					found_pixel = 1;
					pixel_val += i<<5;
					heap_cur += pixel_val-((*imgptr)&0xff);
					break;
				}
				bit_cur++;
			}
			if ( found_pixel )
#else
			heap_cur++;
			unsigned long pixel_val = 0;
			for ( unsigned long i = ((*imgptr)&0xff)+1; i < 256; i++ )
			{
				if ( **heap_cur )
				{
					pixel_val = i;
					break;
				}
				heap_cur++;
			}
			if ( pixel_val )
#endif
			{
				imgptr = **heap_cur;
				(*heap_cur)--;
#ifdef __INTRIN_ENABLED__
				if ( !**heap_cur )
					_bitreset( bit_cur, pixel_val&0x1f );
#endif
				if ( pixel_val < comptr[-1].grey_level )
				{
					// check the stablity and push a new history, increase the grey level
					if ( icvMSERStableCheck( comptr, params ) )
					{
						CvContour* contour = icvMSERToContour( comptr, storage );
						contour->color = color;
						cvSeqPush( contours, &contour );
					}
					icvMSERNewHistory( comptr, histptr );
					comptr[0].grey_level = pixel_val;
					histptr++;
				} else {
					// keep merging top two comp in stack until the grey level >= pixel_val
					for ( ; ; )
					{
						comptr--;
						icvMSERMergeComp( comptr+1, comptr, comptr, histptr );
						histptr++;
						if ( pixel_val <= comptr[0].grey_level )
							break;
						if ( pixel_val < comptr[-1].grey_level )
						{
							// check the stablity here otherwise it wouldn't be an ER
							if ( icvMSERStableCheck( comptr, params ) )
							{
								CvContour* contour = icvMSERToContour( comptr, storage );
								contour->color = color;
								cvSeqPush( contours, &contour );
							}
							icvMSERNewHistory( comptr, histptr );
							comptr[0].grey_level = pixel_val;
							histptr++;
							break;
						}
					}
				}
			} else
				break;
		}
	}
}

static void
icvExtractMSER_8UC1( CvMat* src,
		     CvMat* mask,
		     CvSeq* contours,
		     CvMemStorage* storage,
		     CvMSERParams params )
{
	int step = 8;
	int stepgap = 3;
	while ( step < src->step+2 )
	{
		step <<= 1;
		stepgap++;
	}
	int stepmask = step-1;

	// to speedup the process, make the width to be 2^N
	CvMat* img = cvCreateMat( src->rows+2, step, CV_32SC1 );
	int* imgptr = img->data.i+step+1;

	// pre-allocate boundary heap
	int** heap = (int**)cvAlloc( (src->rows*src->cols+256)*sizeof(heap[0]) );
	int** heap_start[256];
	heap_start[0] = heap;

	// pre-allocate linked point and grow history
	CvLinkedPoint* pts = (CvLinkedPoint*)cvAlloc( src->rows*src->cols*sizeof(pts[0]) );
	CvMSERGrowHistory* history = (CvMSERGrowHistory*)cvAlloc( src->rows*src->cols*sizeof(history[0]) );
	CvMSERConnectedComp comp[257];

	// darker to brighter (MSER-)
	icvPreprocessMSER_8UC1( src, img, heap_start );
	icvExtractMSER_8UC1_Pass( imgptr, heap_start, pts, history, comp, step, stepmask, stepgap, params, -1, contours, storage );
	// brighter to darker (MSER+)
	icvPreprocessMSER_8UC1( src, img, heap_start );
	icvExtractMSER_8UC1_Pass( imgptr, heap_start, pts, history, comp, step, stepmask, stepgap, params, 1, contours, storage );

	// clean up
	cvFree( &history );
	cvFree( &heap );
	cvFree( &pts );
	cvReleaseMat( &img );
}

struct CvMSCRNode;

typedef struct CvTempMSCR
{
	CvMSCRNode* head;
	CvMSCRNode* tail;
	double m; // the margin used to prune area later
	int size;
} CvTempMSCR;

typedef struct CvMSCRNode
{
	CvMSCRNode* shortcut;
	// to make the finding of root less painful
	CvMSCRNode* prev;
	CvMSCRNode* next;
	// a point double-linked list
	CvTempMSCR* tmsr;
	// the temporary msr (set to NULL at every re-initialise)
	CvTempMSCR* gmsr;
	// the global msr (once set, never to NULL)
	int index;
	// the index of the node, at this point, it should be x at the first 16-bits, and y at the last 16-bits.
	int rank;
	int reinit;
	int size, sizei;
	double dt, di;
	double s;
} CvMSCRNode;

typedef struct CvMSCREdge
{
	double chi;
	CvMSCRNode* left;
	CvMSCRNode* right;
} CvMSCREdge;

#include "chi_table.h"

CV_INLINE static double
icvChisquaredDistance( uchar* x, uchar* y )
{
	return (double)((x[0]-y[0])*(x[0]-y[0]))/(double)(x[0]+y[0]+1e-10)+
	       (double)((x[1]-y[1])*(x[1]-y[1]))/(double)(x[1]+y[1]+1e-10)+
	       (double)((x[2]-y[2])*(x[2]-y[2]))/(double)(x[2]+y[2]+1e-10);
}

CV_INLINE static void
icvInitMSCRNode( CvMSCRNode* node )
{
	node->gmsr = node->tmsr = NULL;
	node->reinit = 0xffff;
	node->rank = 0;
	node->sizei = node->size = 1;
	node->prev = node->next = node->shortcut = node;
}

// the preprocess to get the edge list with proper gaussian blur
static void
icvPreprocessMSER_8UC3( CvMSCRNode* node,
			CvMSCREdge* edge,
			double* total,
			CvMat* src,
			CvMat* dx,
			CvMat* dy,
			int edge_blur_size )
{
	int srccpt = src->step-src->cols*3;
	uchar* srcptr = src->data.ptr;
	uchar* lastptr = src->data.ptr+3;
	double* dxptr = dx->data.db;
	for ( int i = 0; i < src->rows; i++ )
	{
		for ( int j = 0; j < src->cols-1; j++ )
		{
			*dxptr = icvChisquaredDistance( srcptr, lastptr );
			dxptr++;
			srcptr += 3;
			lastptr += 3;
		}
		srcptr += srccpt+3;
		lastptr += srccpt+3;
	}
	srcptr = src->data.ptr;
	lastptr = src->data.ptr+src->step;
	double* dyptr = dy->data.db;
	for ( int i = 0; i < src->rows-1; i++ )
	{
		for ( int j = 0; j < src->cols; j++ )
		{
			*dyptr = icvChisquaredDistance( srcptr, lastptr );
			dyptr++;
			srcptr += 3;
			lastptr += 3;
		}
		srcptr += srccpt;
		lastptr += srccpt;
	}
	// get dx and dy and blur it
	if ( edge_blur_size >= 1 )
	{
		cvSmooth( dx, dx, CV_GAUSSIAN, edge_blur_size, edge_blur_size );
		cvSmooth( dy, dy, CV_GAUSSIAN, edge_blur_size, edge_blur_size );
	}
	// assian dx, dy to proper edge list and initialize mscr node
	dxptr = dx->data.db;
	dyptr = dy->data.db;
	CvMSCRNode* nodeptr = node;
	icvInitMSCRNode( nodeptr );
	nodeptr->index = 0;
	*total += edge->chi = *dxptr;
	dxptr++;
	edge->left = nodeptr;
	edge->right = nodeptr+1;
	edge++;
	nodeptr++;
	for ( int i = 1; i < src->cols-1; i++ )
	{
		icvInitMSCRNode( nodeptr );
		nodeptr->index = i;
		*total += edge->chi = *dxptr;
		dxptr++;
		edge->left = nodeptr;
		edge->right = nodeptr+1;
		edge++;
		nodeptr++;
	}
	icvInitMSCRNode( nodeptr );
	nodeptr->index = src->cols-1;
	nodeptr++;
	for ( int i = 1; i < src->rows-1; i++ )
	{
		icvInitMSCRNode( nodeptr );
		nodeptr->index = i<<16;
		*total += edge->chi = *dyptr;
		dyptr++;
		edge->left = nodeptr-src->cols;
		edge->right = nodeptr;
		edge++;
		*total += edge->chi = *dxptr;
		dxptr++;
		edge->left = nodeptr;
		edge->right = nodeptr+1;
		edge++;
		nodeptr++;
		for ( int j = 1; j < src->cols-1; j++ )
		{
			icvInitMSCRNode( nodeptr );
			nodeptr->index = (i<<16)|j;
			*total += edge->chi = *dyptr;
			dyptr++;
			edge->left = nodeptr-src->cols;
			edge->right = nodeptr;
			edge++;
			*total += edge->chi = *dxptr;
			dxptr++;
			edge->left = nodeptr;
			edge->right = nodeptr+1;
			edge++;
			nodeptr++;
		}
		icvInitMSCRNode( nodeptr );
		nodeptr->index = (i<<16)|(src->cols-1);
		*total += edge->chi = *dyptr;
		dyptr++;
		edge->left = nodeptr-src->cols;
		edge->right = nodeptr;
		edge++;
		nodeptr++;
	}
	icvInitMSCRNode( nodeptr );
	nodeptr->index = (src->rows-1)<<16;
	*total += edge->chi = *dxptr;
	dxptr++;
	edge->left = nodeptr;
	edge->right = nodeptr+1;
	edge++;
	*total += edge->chi = *dyptr;
	dyptr++;
	edge->left = nodeptr-src->cols;
	edge->right = nodeptr;
	edge++;
	nodeptr++;
	for ( int i = 1; i < src->cols-1; i++ )
	{
		icvInitMSCRNode( nodeptr );
		nodeptr->index = ((src->rows-1)<<16)|i;
		*total += edge->chi = *dxptr;
		dxptr++;
		edge->left = nodeptr;
		edge->right = nodeptr+1;
		edge++;
		*total += edge->chi = *dyptr;
		dyptr++;
		edge->left = nodeptr-src->cols;
		edge->right = nodeptr;
		edge++;
		nodeptr++;
	}
	icvInitMSCRNode( nodeptr );
	nodeptr->index = ((src->rows-1)<<16)|(src->cols-1);
	*total += edge->chi = *dyptr;
	edge->left = nodeptr-src->cols;
	edge->right = nodeptr;
}

#define cmp_mscr_edge(edge1, edge2) \
	((edge1).chi < (edge2).chi)

static CV_IMPLEMENT_QSORT( icvQuickSortMSCREdge, CvMSCREdge, cmp_mscr_edge )

// to find the root of one region
CV_INLINE static CvMSCRNode*
icvFindMSCR( CvMSCRNode* x )
{
	CvMSCRNode* prev = x;
	CvMSCRNode* next;
	for ( ; ; )
	{
		next = x->shortcut;
		x->shortcut = prev;
		if ( next == x ) break;
		prev= x;
		x = next;
	}
	CvMSCRNode* root = x;
	for ( ; ; )
	{
		prev = x->shortcut;
		x->shortcut = root;
		if ( prev == x ) break;
		x = prev;
	}
	return root;
}

// the stable mscr should be:
// bigger than min_area and smaller than max_area
// differ from its ancestor more than min_diversity
CV_INLINE static bool
icvMSCRStableCheck( CvMSCRNode* x,
		    CvMSERParams params )
{
	if ( x->size <= params.min_area || x->size >= params.max_area )
		return 0;
	if ( x->gmsr == NULL )
		return 1;
	double div = (double)(x->size-x->gmsr->size)/(double)x->size;
	return div > params.min_diversity;
}

static void
icvExtractMSER_8UC3( CvMat* src,
		     CvMat* mask,
		     CvSeq* contours,
		     CvMemStorage* storage,
		     CvMSERParams params )
{
	CvMSCRNode* map = (CvMSCRNode*)cvAlloc( src->cols*src->rows*sizeof(map[0]) );
	int Ne = src->cols*src->rows*2-src->cols-src->rows;
	CvMSCREdge* edge = (CvMSCREdge*)cvAlloc( Ne*sizeof(edge[0]) );
	CvTempMSCR* mscr = (CvTempMSCR*)cvAlloc( src->cols*src->rows*sizeof(mscr[0]) );
	double emean = 0;
	CvMat* dx = cvCreateMat( src->rows, src->cols-1, CV_64FC1 );
	CvMat* dy = cvCreateMat( src->rows-1, src->cols, CV_64FC1 );
	icvPreprocessMSER_8UC3( map, edge, &emean, src, dx, dy, params.edge_blur_size );
	emean = emean / (double)Ne;
	icvQuickSortMSCREdge( edge, Ne, 0 );
	CvMSCREdge* edge_ub = edge+Ne;
	CvMSCREdge* edgeptr = edge;
	CvTempMSCR* mscrptr = mscr;
	// the evolution process
	for ( int i = 0; i < params.max_evolution; i++ )
	{
		double k = (double)i/(double)params.max_evolution*(TABLE_SIZE-1);
		int ti = floor(k);
		double reminder = k-ti;
		double thres = emean*(chitab3[ti]*(1-reminder)+chitab3[ti+1]*reminder);
		// to process all the edges in the list that chi < thres
		while ( edgeptr < edge_ub && edgeptr->chi < thres )
		{
			CvMSCRNode* lr = icvFindMSCR( edgeptr->left );
			CvMSCRNode* rr = icvFindMSCR( edgeptr->right );
			// get the region root (who is responsible)
			if ( lr != rr )
			{
				// rank idea take from: N-tree Disjoint-Set Forests for Maximally Stable Extremal Regions
				if ( rr->rank > lr->rank )
				{
					CvMSCRNode* tmp;
					CV_SWAP( lr, rr, tmp );
				} else if ( lr->rank == rr->rank ) {
					// at the same rank, we will compare the size
					if ( lr->size > rr->size )
					{
						CvMSCRNode* tmp;
						CV_SWAP( lr, rr, tmp );
					}
					lr->rank++;
				}
				rr->shortcut = lr;
				lr->size += rr->size;
				// join rr to the end of list lr (lr is a endless double-linked list)
				lr->prev->next = rr;
				lr->prev = rr->prev;
				rr->prev->next = lr;
				rr->prev = lr;
				// area threshold force to reinitialize
				if ( lr->size > (lr->size-rr->size)*params.area_threshold )
				{
					lr->sizei = lr->size;
					lr->reinit = i;
					if ( lr->tmsr != NULL )
					{
						lr->tmsr->m = lr->dt-lr->di;
						lr->tmsr = NULL;
					}
					lr->di = edgeptr->chi;
					lr->s = 1e10;
				}
				lr->dt = edgeptr->chi;
				if ( i > lr->reinit )
				{
					double s = (double)(lr->size-lr->sizei)/(lr->dt-lr->di);
					if ( s < lr->s )
					{
						// skip the first one and check stablity
						if ( i > lr->reinit+1 && icvMSCRStableCheck( lr, params ) )
						{
							if ( lr->tmsr == NULL )
							{
								lr->gmsr = lr->tmsr = mscrptr;
								mscrptr++;
							}
							lr->tmsr->size = lr->size;
							lr->tmsr->head = lr;
							lr->tmsr->tail = lr->prev;
							lr->tmsr->m = 0;
						}
						lr->s = s;
					}
				}
			}
			edgeptr++;
		}
		if ( edgeptr >= edge_ub )
			break;
	}
	for ( CvTempMSCR* ptr = mscr; ptr < mscrptr; ptr++ )
		// to prune area with margin less than min_margin
		if ( ptr->m > params.min_margin )
		{
			CvSeq* _contour = cvCreateSeq( CV_SEQ_KIND_GENERIC|CV_32SC2, sizeof(CvContour), sizeof(CvPoint), storage );
			cvSeqPushMulti( _contour, 0, ptr->size );
			CvMSCRNode* lpt = ptr->head;
			for ( int i = 0; i < ptr->size; i++ )
			{
				CvPoint* pt = CV_GET_SEQ_ELEM( CvPoint, _contour, i );
				pt->x = (lpt->index)&0xffff;
				pt->y = (lpt->index)>>16;
				lpt = lpt->next;
			}
			CvContour* contour = (CvContour*)_contour;
			contour->color = 0;
			cvSeqPush( contours, &contour );
		}
	cvReleaseMat( &dx );
	cvReleaseMat( &dy );
	cvFree( &mscr );
	cvFree( &edge );
	cvFree( &map );
}

void
cvExtractMSER( CvArr* _img,
	       CvArr* _mask,
	       CvSeq** _contours,
	       CvMemStorage* storage,
	       CvMSERParams params )
{
	CvMat srchdr, *src = cvGetMat( _img, &srchdr );
	CvMat maskhdr, *mask = _mask ? cvGetMat( _mask, &maskhdr ) : 0;
	CvSeq* contours = 0;

	CV_FUNCNAME( "cvExtractMSER" );

	__BEGIN__;

	CV_ASSERT(src != 0);
	CV_ASSERT(CV_MAT_TYPE(src->type) == CV_8UC1 || CV_MAT_TYPE(src->type) == CV_8UC3);
	CV_ASSERT(mask == 0 || (CV_ARE_SIZES_EQ(src, mask) && CV_MAT_TYPE(mask->type) == CV_8UC1));
	CV_ASSERT(storage != 0);

	contours = *_contours = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvSeq*), storage );

	// choose different method for different image type
	// for grey image, it is: Linear Time Maximally Stable Extremal Regions
	// for color image, it is: Maximally Stable Colour Regions for Recognition and Matching
	switch ( CV_MAT_TYPE(src->type) )
	{
		case CV_8UC1:
			icvExtractMSER_8UC1( src, mask, contours, storage, params );
			break;
		case CV_8UC3:
			icvExtractMSER_8UC3( src, mask, contours, storage, params );
			break;
	}

	__END__;
}
