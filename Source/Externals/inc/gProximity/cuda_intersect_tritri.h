/*
 *  gProximity Library.
 *  
 *  
 *  Copyright (C) 2010 University of North Carolina at Chapel Hill.
 *  All rights reserved.
 *  
 *  Permission to use, copy, modify, and distribute this software and its
 *  documentation for educational, research, and non-profit purposes, without
 *  fee, and without a written agreement is hereby granted, provided that the
 *  above copyright notice, this paragraph, and the following four paragraphs
 *  appear in all copies.
 *  
 *  Permission to incorporate this software into commercial products may be
 *  obtained by contacting the University of North Carolina at Chapel Hill.
 *  
 *  This software program and documentation are copyrighted by the University of
 *  North Carolina at Chapel Hill. The software program and documentation are
 *  supplied "as is", without any accompanying services from the University of
 *  North Carolina at Chapel Hill or the authors. The University of North
 *  Carolina at Chapel Hill and the authors do not warrant that the operation of
 *  the program will be uninterrupted or error-free. The end-user understands
 *  that the program was developed for research purposes and is advised not to
 *  rely exclusively on the program for any reason.
 *  
 *  IN NO EVENT SHALL THE UNIVERSITY OF NORTH CAROLINA AT CHAPEL HILL OR ITS
 *  EMPLOYEES OR THE AUTHORS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
 *  SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS,
 *  ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE
 *  UNIVERSITY OF NORTH CAROLINA AT CHAPEL HILL OR THE AUTHORS HAVE BEEN ADVISED
 *  OF THE POSSIBILITY OF SUCH DAMAGE.
 *  
 *  THE UNIVERSITY OF NORTH CAROLINA AT CHAPEL HILL AND THE AUTHORS SPECIFICALLY
 *  DISCLAIM ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE AND ANY
 *  STATUTORY WARRANTY OF NON-INFRINGEMENT. THE SOFTWARE PROVIDED HEREUNDER IS
 *  ON AN "AS IS" BASIS, AND THE UNIVERSITY OF NORTH CAROLINA AT CHAPEL HILL AND
 *  THE AUTHORS HAVE NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES,
 *  ENHANCEMENTS, OR MODIFICATIONS.
 *  
 *  Please send all BUG REPORTS to:
 *  
 *  geom@cs.unc.edu
 *  
 *  The authors may be contacted via:
 *  
 *  Christian Lauterbach, Qi Mo, Jia Pan and Dinesh Manocha
 *  Dept. of Computer Science
 *  Frederick P. Brooks Jr. Computer Science Bldg.
 *  3175 University of N.C.
 *  Chapel Hill, N.C. 27599-3175
 *  United States of America
 *  
 *  http://gamma.cs.unc.edu/GPUCOL/
 *  
 */
 
#ifndef __CUDA_INTERSECT_TRITRI_H_
#define __CUDA_INTERSECT_TRITRI_H_

#include "cuda_vectors.h"

/* sort so that a<=b */
#define SORT(a,b)       \
	if(a>b)    \
{          \
	float c; \
	c=a;     \
	a=b;     \
	b=c;     \
}

#define ISECT(VV0,VV1,VV2,D0,D1,D2,isect0,isect1) \
	isect0=VV0+(VV1-VV0)*D0/(D0-D1);    \
	isect1=VV0+(VV2-VV0)*D0/(D0-D2);

#define NEWCOMPUTE_INTERVALS(VV0,VV1,VV2,D0,D1,D2,D0D1,D0D2,A,B,C,X0,X1) \
{ \
	if(D0D1>0.0f) \
{ \
	/* here we know that D0D2<=0.0 */ \
	/* that is D0, D1 are on the same side, D2 on the other or on the plane */ \
	A=VV2; B=(VV0-VV2)*D2; C=(VV1-VV2)*D2; X0=D2-D0; X1=D2-D1; \
} \
		else if(D0D2>0.0f)\
{ \
	/* here we know that d0d1<=0.0 */ \
	A=VV1; B=(VV0-VV1)*D1; C=(VV2-VV1)*D1; X0=D1-D0; X1=D1-D2; \
} \
		else if(D1*D2>0.0f || D0!=0.0f) \
{ \
	/* here we know that d0d1<=0.0 or that D0!=0.0 */ \
	A=VV0; B=(VV1-VV0)*D0; C=(VV2-VV0)*D0; X0=D0-D1; X1=D0-D2; \
} \
		else if(D1!=0.0f) \
{ \
	A=VV1; B=(VV0-VV1)*D1; C=(VV2-VV1)*D1; X0=D1-D0; X1=D1-D2; \
} \
		else if(D2!=0.0f) \
{ \
	A=VV2; B=(VV0-VV2)*D2; C=(VV1-VV2)*D2; X0=D2-D0; X1=D2-D1; \
} \
		else \
{ \
	/* triangles are coplanar */ \
	return 0;\
} \
}

__device__ inline int triangleIntersection_(float3 const& U0, float3 const& U1, float3 const& U2, float3 const& V0, float3 const& V1, float3 const& V2)
{
	float3 N1, N2;
	float d1, d2;
	float du0, du1, du2, dv0, dv1, dv2;
	float du0du1, du0du2, dv0dv1, dv0dv2;
	float vp0, vp1, vp2;
	float up0, up1, up2;
	
	/* compute plane equation of triangle(V0,V1,V2) */
	{
		float3 E1, E2;
		f3s_sub(E1, V1, V0);
		f3s_sub(E2, V2, V0);
		f3s_cross(N1, E1, E2);
		d1 = -f3s_dot(N1, V0);
		/* plane equation 1: N1.X+d1=0 */
		
		/* put U0,U1,U2 into plane equation 1 to compute signed distances to the plane*/
		du0 = f3s_dot(N1, U0) + d1;
		du1 = f3s_dot(N1, U1) + d1;
		du2 = f3s_dot(N1, U2) + d1;
	}
	
	/* coplanarity robustness check */
#if USE_EPSILON_TEST==TRUE
	if(fabs(du0) < FLT_EPSILON) du0 = 0.0;
	if(fabs(du1) < FLT_EPSILON) du1 = 0.0;
	if(fabs(du2) < FLT_EPSILON) du2 = 0.0;
#endif
	du0du1 = du0 * du1;
	du0du2 = du0 * du2;
	
	if(du0du1 > 0.0f && du0du2 > 0.0f)  /* same sign on all of them + not equal 0 ? */
		return 0;                    /* no intersection occurs */
		
	/* compute plane of triangle (U0,U1,U2) */
	{
		float3 E1, E2;
		f3s_sub(E1, U1, U0);
		f3s_sub(E2, U2, U0);
		f3s_cross(N2, E1, E2);
		d2 = -f3s_dot(N2, U0);
		/* plane equation 2: N2.X+d2=0 */
		
		/* put V0,V1,V2 into plane equation 2 */
		dv0 = f3s_dot(N2, V0) + d2;
		dv1 = f3s_dot(N2, V1) + d2;
		dv2 = f3s_dot(N2, V2) + d2;
	}
	
#if USE_EPSILON_TEST==TRUE
	if(fabs(dv0) < FLT_EPSILON) dv0 = 0.0;
	if(fabs(dv1) < FLT_EPSILON) dv1 = 0.0;
	if(fabs(dv2) < FLT_EPSILON) dv2 = 0.0;
#endif
	
	dv0dv1 = dv0 * dv1;
	dv0dv2 = dv0 * dv2;
	
	if(dv0dv1 >= 0.0f && dv0dv2 >= 0.0f)  /* same sign on all of them + not equal 0 ? */
		return 0;                    /* no intersection occurs */
		
	{
		/* compute direction of intersection line */
		float3 D;
		f3s_cross(D, N1, N2);
		
		/* compute and index to the largest component of D */
		float bb, cc, aa;
		aa = (float)fabs(D.x);
		bb = (float)fabs(D.y);
		cc = (float)fabs(D.z);
		if(aa > bb && aa > cc)
		{
			/* this is the simplified projection onto L*/
			vp0 = V0.x;
			vp1 = V1.x;
			vp2 = V2.x;
			up0 = U0.x;
			up1 = U1.x;
			up2 = U2.x;
		}
		else if(bb > cc && bb > aa)
		{
			/* this is the simplified projection onto L*/
			vp0 = V0.y;
			vp1 = V1.y;
			vp2 = V2.y;
			up0 = U0.y;
			up1 = U1.y;
			up2 = U2.y;
		}
		else
		{
			/* this is the simplified projection onto L*/
			vp0 = V0.z;
			vp1 = V1.z;
			vp2 = V2.z;
			up0 = U0.z;
			up1 = U1.z;
			up2 = U2.z;
		}
	}
	
	float a, b, c, x0, x1;
	float d, e, f, y0, y1;
	/* compute interval for triangle 1 */
	NEWCOMPUTE_INTERVALS(vp0, vp1, vp2, dv0, dv1, dv2, dv0dv1, dv0dv2, a, b, c, x0, x1);
	
	/* compute interval for triangle 2 */
	NEWCOMPUTE_INTERVALS(up0, up1, up2, du0, du1, du2, du0du1, du0du2, d, e, f, y0, y1);
	
	float xx, yy, xxyy, tmp;
	xx = x0 * x1;
	yy = y0 * y1;
	xxyy = xx * yy;
	
	tmp = a * xxyy;
	
	float2 isect1, isect2;
	isect1.x = tmp + b * x1 * yy;
	isect1.y = tmp + c * x0 * yy;
	
	tmp = d * xxyy;
	isect2.x = tmp + e * xx * y1;
	isect2.y = tmp + f * xx * y0;
	
	SORT(isect1.x, isect1.y);
	SORT(isect2.x, isect2.y);
	
	if(isect1.y <= isect2.x || isect2.y <= isect1.x) return 0;
	return 1;
}



__device__ __inline__ int project6(float3 const& ax,
                                   float3 const& p1, float3 const& p2, float3 const& p3,
                                   float3 const& q1, float3 const& q2, float3 const& q3)
{
	float P1 = f3v_dot(ax, p1);
	float P2 = f3v_dot(ax, p2);
	float P3 = f3v_dot(ax, p3);
	float Q1 = f3v_dot(ax, q1);
	float Q2 = f3v_dot(ax, q2);
	float Q3 = f3v_dot(ax, q3);
	
	float mx1 = fmax(fmax(P1, P2), P3);
	float mn1 = fmin(fmin(P1, P2), P3);
	float mx2 = fmax(fmax(Q1, Q2), Q3);
	float mn2 = fmin(fmin(Q1, Q2), Q3);
	
	if(mn1 > mx2) return 0;
	if(mn2 > mx1) return 0;
	return 1;
}

__device__ int triangleIntersection(float3 const& P1, float3 const& P2, float3 const& P3,
									float3 const& Q1, float3 const& Q2, float3 const& Q3)
{

	// One triangle is (p1,p2,p3).  Other is (q1,q2,q3).
	// Edges are (e1,e2,e3) and (f1,f2,f3).
	// Normals are n1 and m1
	// Outwards are (g1,g2,g3) and (h1,h2,h3).
	//
	// We assume that the triangle vertices are in the same coordinate system.
	//
	// First thing we do is establish a new c.s. so that p1 is at (0,0,0).

	float3 p1, p2, p3;
	float3 q1, q2, q3;
	float3 e1, e2, e3;
	float3 f1, f2, f3;

	make_float3(0, 0, 0);
	f3s_sub(p2, P2, P1);
	f3s_sub(p3, P3, P1);

	f3s_sub(q1, Q1, P1);
	f3s_sub(q2, Q2, P1);
	f3s_sub(q3, Q3, P1);

	f3s_sub(e1, p2, p1);
	f3s_sub(e2, p3, p2);
	f3s_sub(e3, p1, p3);

	f3s_sub(f1, q2, q1);
	f3s_sub(f2, q3, q2);
	f3s_sub(f3, q1, q3);

	float3 t;
	float3 m1, n1;

	f3s_cross(n1, e1, e2);
	if(!project6(n1, p1, p2, p3, q1, q2, q3)) return 0;
	f3s_cross(m1, f1, f2);
	if(!project6(m1, p1, p2, p3, q1, q2, q3)) return 0;


	f3s_cross(t, e1, f1);
	if(!project6(t, p1, p2, p3, q1, q2, q3)) return 0;
	f3s_cross(t, e1, f2);
	if(!project6(t, p1, p2, p3, q1, q2, q3)) return 0;
	f3s_cross(t, e1, f3);
	if(!project6(t, p1, p2, p3, q1, q2, q3)) return 0;
	f3s_cross(t, e2, f1);
	if(!project6(t, p1, p2, p3, q1, q2, q3)) return 0;
	f3s_cross(t, e2, f2);
	if(!project6(t, p1, p2, p3, q1, q2, q3)) return 0;
	f3s_cross(t, e2, f3);
	if(!project6(t, p1, p2, p3, q1, q2, q3)) return 0;
	f3s_cross(t, e3, f1);
	if(!project6(t, p1, p2, p3, q1, q2, q3)) return 0;
	f3s_cross(t, e3, f2);
	if(!project6(t, p1, p2, p3, q1, q2, q3)) return 0;
	f3s_cross(t, e3, f3);
	if(!project6(t, p1, p2, p3, q1, q2, q3)) return 0;


	f3s_cross(t, e1, n1);
	if(!project6(t, p1, p2, p3, q1, q2, q3)) return 0;
	f3s_cross(t, e2, n1);
	if(!project6(t, p1, p2, p3, q1, q2, q3)) return 0;
	f3s_cross(t, e3, n1);
	if(!project6(t, p1, p2, p3, q1, q2, q3)) return 0;
	f3s_cross(t, f1, m1);
	if(!project6(t, p1, p2, p3, q1, q2, q3)) return 0;
	f3s_cross(t, f2, m1);
	if(!project6(t, p1, p2, p3, q1, q2, q3)) return 0;
	f3s_cross(t, f3, m1);
	if(!project6(t, p1, p2, p3, q1, q2, q3)) return 0;



	return 1;
}

__device__ int triangleIntersection2(float3 const& P1, float3 const& P2, float3 const& P3,
                                    float3 const& Q1, float3 const& Q2, float3 const& Q3)
{

	// One triangle is (p1,p2,p3).  Other is (q1,q2,q3).
	// Edges are (e1,e2,e3) and (f1,f2,f3).
	// Normals are n1 and m1
	// Outwards are (g1,g2,g3) and (h1,h2,h3).
	//
	// We assume that the triangle vertices are in the same coordinate system.
	//
	// First thing we do is establish a new c.s. so that p1 is at (0,0,0).
	
	float3 p1, p2, p3;
	float3 q1, q2, q3;
	float3 e1, e2, e3;
	float3 f1, f2, f3;
	float3 g1, g2, g3;
	float3 h1, h2, h3;
	float3 n1, m1;
	
	float3 ef11, ef12, ef13;
	float3 ef21, ef22, ef23;
	float3 ef31, ef32, ef33;
	
	p1 = make_float3(0, 0, 0);
	p2 = f3v_sub(P2, P1);
	p3 = f3v_sub(P3, P1);
	
	q1 = f3v_sub(Q1, P1);
	q2 = f3v_sub(Q2, P1);
	q3 = f3v_sub(Q3, P1);
	
	e1 = f3v_sub(p2, p1);
	e2 = f3v_sub(p3, p2);
	e3 = f3v_sub(p1, p3);
	
	f1 = f3v_sub(q2, q1);
	f2 = f3v_sub(q3, q2);
	f3 = f3v_sub(q1, q3);
	
	f3s_cross(n1, e1, e2);
	f3s_cross(m1, f1, f2);
	
	f3s_cross(g1, e1, n1);
	f3s_cross(g2, e2, n1);
	f3s_cross(g3, e3, n1);
	f3s_cross(h1, f1, m1);
	f3s_cross(h2, f2, m1);
	f3s_cross(h3, f3, m1);
	
	f3s_cross(ef11, e1, f1);
	f3s_cross(ef12, e1, f2);
	f3s_cross(ef13, e1, f3);
	f3s_cross(ef21, e2, f1);
	f3s_cross(ef22, e2, f2);
	f3s_cross(ef23, e2, f3);
	f3s_cross(ef31, e3, f1);
	f3s_cross(ef32, e3, f2);
	f3s_cross(ef33, e3, f3);
	
	// now begin the series of tests
	
	if(!project6(n1, p1, p2, p3, q1, q2, q3)) return 0;
	if(!project6(m1, p1, p2, p3, q1, q2, q3)) return 0;
	
	if(!project6(ef11, p1, p2, p3, q1, q2, q3)) return 0;
	if(!project6(ef12, p1, p2, p3, q1, q2, q3)) return 0;
	if(!project6(ef13, p1, p2, p3, q1, q2, q3)) return 0;
	if(!project6(ef21, p1, p2, p3, q1, q2, q3)) return 0;
	if(!project6(ef22, p1, p2, p3, q1, q2, q3)) return 0;
	if(!project6(ef23, p1, p2, p3, q1, q2, q3)) return 0;
	if(!project6(ef31, p1, p2, p3, q1, q2, q3)) return 0;
	if(!project6(ef32, p1, p2, p3, q1, q2, q3)) return 0;
	if(!project6(ef33, p1, p2, p3, q1, q2, q3)) return 0;
	
	if(!project6(g1, p1, p2, p3, q1, q2, q3)) return 0;
	if(!project6(g2, p1, p2, p3, q1, q2, q3)) return 0;
	if(!project6(g3, p1, p2, p3, q1, q2, q3)) return 0;
	if(!project6(h1, p1, p2, p3, q1, q2, q3)) return 0;
	if(!project6(h2, p1, p2, p3, q1, q2, q3)) return 0;
	if(!project6(h3, p1, p2, p3, q1, q2, q3)) return 0;
	
	return 1;
}

#if 0
// Kernel for computing collisions from list of triangle pairs
template <int nThreads, bool skipAdjacent>
__global__ void triangleCollision(uint3 *d_GL_triIndices, GPUVertex *d_GL_vertices, int2 *pairs, const int nPairs)
{
	const int myID = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	__shared__ int2 local_pairs[nThreads];
	
	// don't go beyond end of pair list
	if(myID < nPairs)
	{
		float3 u1, u2, u3, v1, v2, v3;
		
		// read in IDs for both triangles
		{
			int2 pair = pairs[myID];
			local_pairs[threadIdx.x] = pair;
			
			// read in triangles
			const uint3 idx  = d_GL_triIndices[pair.x];
			const uint3 idx2 = d_GL_triIndices[pair.y];
			
			// test whether adjacent
			if(skipAdjacent)
			{
				if((idx.x == idx2.x || idx.x == idx2.y || idx.x == idx2.z) ||
				        (idx.y == idx2.x || idx.y == idx2.y || idx.y == idx2.z) ||
				        (idx.z == idx2.x || idx.z == idx2.y || idx.z == idx2.z))
					return;
			}
			
			// not adjacent
			u1 = d_GL_vertices[idx.x].v;
			u2 = d_GL_vertices[idx.y].v;
			u3 = d_GL_vertices[idx.z].v;
			
			v1 = d_GL_vertices[idx2.x].v;
			v2 = d_GL_vertices[idx2.y].v;
			v3 = d_GL_vertices[idx2.z].v;
		}
		
		// intersect triangles
		if(triangleIntersection(u1, u2, u3, v1, v2, v3))
		{
			pairs[myID] = make_int2(-local_pairs[threadIdx.x].x, -local_pairs[threadIdx.x].y);
		}
	}
}
#endif

#endif