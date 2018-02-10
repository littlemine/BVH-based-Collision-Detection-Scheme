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
 
#ifndef __CUDA_VECTORS_H_
#define __CUDA_VECTORS_H_

#include <vector_types.h>
#include <vector_functions.h>

// Vector operations on float2
#define f2_dot(v1, v2) ((v1).x * (v2).x + (v1).y * (v2).y)
#define f2_sub(result, v1, v2) { result.x = (v1).x - (v2).x; result.y = (v1).y - (v2).y;}
#define f2_add(result, v1, v2) { result.x = (v1).x + (v2).x; result.y = (v1).y + (v2).y;}
#define f2_mul(result, v1, v2) { result.x = (v1).x + (v2).x; result.y = (v1).y * (v2).y;}

// Vector operations on float[3]
#define f3_assign(v2, v1) { (v2)[0] = (v1)[0]; (v2)[1] = (v1)[1]; (v2)[2] = (v1)[2]; }
#define f3_dot(v1, v2) ((v1)[0] * (v2)[0] + (v1)[1] * (v2)[1] + (v1)[2] * (v2)[2])
#define f3_sub(result, v1, v2) { result[0] = (v1)[0] - (v2)[0]; result[1] = (v1)[1] - (v2)[1]; result[2] = (v1)[2] - (v2)[2]; }
#define f3_add(result, v1, v2) { result[0] = (v1)[0] + (v2)[0]; result[1] = (v1)[1] + (v2)[1]; result[2] = (v1)[2] + (v2)[2]; }
#define f3_mul(result, v1, v2) { result[0] = (v1)[0] + (v2)[0]; result[1] = (v1)[1] * (v2)[1]; result[2] = (v1)[2] * (v2)[2]; }
#define f3_cross(result, v1, v2) { result[0] = (v1)[1]*(v2)[2] - (v1)[2]*(v2)[1]; \
	result[1] = (v1)[2]*(v2)[0] - (v1)[0]*(v2)[2]; \
	result[2] = (v1)[0]*(v2)[1] - (v1)[1]*(v2)[0]; }

// Vector operations on float3, make_float3() versions
#define f3v_assign(v2, v1) do { v2 = make_float3((v1).x, (v1).y, (v1).z); } while(0)
#define f3v_add(v1, v2) make_float3((v1).x+(v2).x, (v1).y+(v2).y, (v1).z+(v2).z)
#define f3v_mul(v1, v2) make_float3((v1).x*(v2).x, (v1).y*(v2).y, (v1).z*(v2).z)
#define f3v_sub(v1, v2) make_float3((v1).x-(v2).x, (v1).y-(v2).y, (v1).z-(v2).z)
#define f3v_dot(v1, v2) ((v1).x * (v2).x + (v1).y * (v2).y + (v1).z * (v2).z)

#define f3v_add1(v1, f) make_float3((v1).x+f, (v1).y+f, (v1).z+f)
#define f3v_mul1(v1, f) make_float3((v1).x*f, (v1).y*f, (v1).z*f)
#define f3v_sub1(v1, f) make_float3((v1).x-f, (v1).y-f, (v1).z-f)

#define f3v_cross(v1, v2) make_float3((v1).y*(v2).z - (v1).z*(v2).y,\
	(v1).z*(v2).x - (v1).x*(v2).z,\
	(v1).x*(v2).y - (v1).y*(v2).x)

#define f3v_len(v) ((v).x * (v).x + (v).y * (v).y + (v).z * (v).z)

// Vector operations on float3, normal versions
#define f3s_assign(v2, v1) { (v2).x = (v1).x; (v2).y = (v1).y; (v2).z = (v1).z; }
#define f3s_dot(v1, v2) ((v1).x * (v2).x + (v1).y * (v2).y + (v1).z * (v2).z)
#define f3s_sub(result, v1, v2) { result.x = (v1).x - (v2).x; result.y = (v1).y - (v2).y; result.z = (v1).z - (v2).z; }
#define f3s_add(result, v1, v2) { result.x = (v1).x + (v2).x; result.y = (v1).y + (v2).y; result.z = (v1).z + (v2).z; }
#define f3s_mul(result, v1, v2) { result.x = (v1).x * (v2).x; result.y = (v1).y * (v2).y; result.z = (v1).z * (v2).z; }
#define f3s_cross(result, v1, v2) { result.x = (v1).y*(v2).z - (v1).z*(v2).y; result.y = (v1).z*(v2).x - (v1).x*(v2).z; result.z = (v1).x*(v2).y - (v1).y*(v2).x; }

#define f3v_mulSymM3x3(matrix,vec) make_float3(matrix[0]*vec.x + matrix[1]*vec.y + matrix[2]*vec.z,\
	matrix[1]*vec.x + matrix[3]*vec.y + matrix[4]*vec.z,\
	matrix[2]*vec.x + matrix[4]*vec.y + matrix[5]*vec.z)

#define f3v_mulM3x3(matrix, vec) make_float3(matrix[0]*vec.x + matrix[1]*vec.y + matrix[2]*vec.z,\
	matrix[3]*vec.x + matrix[4]*vec.y + matrix[5]*vec.z,\
	matrix[6]*vec.x + matrix[7]*vec.y + matrix[8]*vec.z)

#define f3v_mulM3x3T(matrix, vec) make_float3(matrix[0]*vec.x + matrix[3]*vec.y + matrix[6]*vec.z,\
	matrix[1]*vec.x + matrix[4]*vec.y + matrix[7]*vec.z,\
	matrix[2]*vec.x + matrix[5]*vec.y + matrix[8]*vec.z)

#define MT3x3_mulM(m1, m2) {m1[0]*m2[0]+m1[3]*m2[3]+m1[6]*m2[6], m1[0]*m2[1]+m1[3]*m2[4]+m1[6]*m2[7], m1[0]*m2[2]+m1[3]*m2[5]+m1[6]*m2[8],\
	m1[1]*m2[0]+m1[4]*m2[3]+m1[7]*m2[6], m1[1]*m2[1]+m1[4]*m2[4]+m1[7]*m2[7], m1[1]*m2[2]+m1[4]*m2[5]+m1[7]*m2[8],\
	m1[2]*m2[0]+m1[5]*m2[3]+m1[8]*m2[6], m1[2]*m2[1]+m1[5]*m2[4]+m1[8]*m2[7], m1[2]*m2[2]+m1[5]*m2[5]+m1[8]*m2[8]}

#define M3x3_mulM(m1, m2) {m1[0]*m2[0]+m1[1]*m2[3]+m1[2]*m2[6], m1[0]*m2[1]+m1[1]*m2[4]+m1[2]*m2[7], m1[0]*m2[2]+m1[1]*m2[5]+m1[2]*m2[8],\
	m1[3]*m2[0]+m1[4]*m2[3]+m1[5]*m2[6], m1[3]*m2[1]+m1[4]*m2[4]+m1[5]*m2[7], m1[3]*m2[2]+m1[4]*m2[5]+m1[5]*m2[8],\
	m1[6]*m2[0]+m1[7]*m2[3]+m1[8]*m2[6], m1[6]*m2[1]+m1[7]*m2[4]+m1[8]*m2[7], m1[6]*m2[2]+m1[7]*m2[5]+m1[8]*m2[8]}


#endif
