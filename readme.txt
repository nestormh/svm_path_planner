Multiclass Support Vector Machine Path Planning
=======================================================================

Copyright 2013 Néstor Morales Hernández

Project homepage: 
- http://nmorales.webs.ull.es/MSVMPP/index.html
- http://verdino.isaatc.ull.es

If you use our software and you like it or have comments on its usefulness etc., we 
would love to hear from you at <nestor@isaatc.ull.es>. You may share with us
your experience and any possibilities that we may improve the work/code.

-------------------------------------------------------------------------

This source code is a method for path planning based on Multiclass Support Vector Machine. Unlike other Support Vector Machine path planning based methods, it generates a decision boundary for each object in the scene and then joins them together to form a graph, using the points in which the boundaries cross each other. This is completed with a visibility Relative Neighborhood Graph, which allows obtaining a short, smooth and safe path between any two points in the map, in case it exists. The method has been tested in real conditions in a real-time application, obtaining good results.

If you use this code for your own research, please cite us:

1.  Néstor Morales, Jonay Toledo, Leopoldo Acosta. Multiclass Support Vector Machine. (pending acceptance)

Also, you may like to cite also some of the papers from the code used in this project:

GPU-LIBSVM:

1.  A. Athanasopoulos, A. Dimou, V. Mezaris, I. Kompatsiaris, "GPU Acceleration for Support Vector Machines", Proc. 12th International Workshop on Image Analysis for Multimedia Interactive Services (WIAMIS 2011), Delft, The Netherlands, April 2011.

GPU-DT:

1.  Computing Two-dimensional Delaunay Triangulation Using Graphics Hardware
    G.D. Rong, T.S. Tan, Thanh-Tung Cao and Stephanus
    The 2008 ACM Symposium on Interactive 3D Graphics and Games, 
    15--17 Feb, Redwood City, CA, USA, pp. 89--97. 

2.  Parallel Banding Algorithm to Compute Exact Distance Transform with the GPU
    T.T. Cao, K. Tang, A. Mohamed, and T.S. Tan
    The 2010 ACM Symposium on Interactive 3D Graphics and Games, 19-21 Feb, 
    Washington DC, USA.

3.  Proof of Correctness of the Digital Delaunay Triangulation Algorithm
    T.T. Cao, H. Edelsbrunner, and T.S. Tan
    Manuscript Jan, 2010. 

4.  Computing Two-Dimensional Constrained Delaunay Triangulation Using Graphics Hardware    
    Meng Qi, Thanh-Tung Cao, Tiow-Seng Tan
    Technical report (TRB3/11 (March 2011))

1. Requirements
==============
- Boost libraries
- CUDA (tested on version 5.0)
- OpenCV
- Point Cloud Library (PCL) - Tested on version 1.6
- Computational Geometry Algorithms Library (CGAL)
- The GNU Multiple Precision Arithmetic Library (required by CGAL)
- GNU MPFR Library (required by CGAL)
- Robot Operating System (ROS) - Tested on Groovy [OPTIONAL]
- A GPU capable of running CUDA.

2. Tested
=========
This code has been tested on NVIDIA Geforce GT 640M on a i7-3630QM.

3. Acknowledgements
===================
We acknowledge that the code in the folder "GPULibSVM" is a modification of the code in the "GPU-accelerated LIBSVM", available at (http://mklab.iti.gr/project/GPU-LIBSVM). Also, the code in the folder "gpudt" is a modification of the code from the GPU-DT libraries, available at (http://www.comp.nus.edu.sg/~tants/delaunay2DDownload.html). Both libraries have been modified and adapted to work with our method.

4. License
===================
Copyright 2013 Néstor Morales Hernández <nestor@isaatc.ull.es>
 
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
 
    http://www.apache.org/licenses/LICENSE-2.0
 
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

5. Funding
===================
This work was supported by the project SAGENIA DPI2010-18349 and the funds from the Agencia Canaria de Investigación, Innovación y Sociedad de la Información (ACIISI), cofinanced by FEDER funds (EU).

----------------------------------------------------------------------------------
Néstor Morales, Jonay Toledo, Leopoldo Acosta
Departamento de Ingeniería de Sistemas y Automática y Arquitectura y Tecnología de Computadores (ISAATC)
Facultad de Física y Matemáticas. Avenida Astrofísico Francisco Sánchez, s/n
38200, La Laguna - Santa Cruz de Tenerife (SPAIN)
----------------------------------------------------------------------------------
Please send bugs and comments to: nestor@isaatc.ull.es
