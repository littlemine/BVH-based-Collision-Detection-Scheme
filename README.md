This is the collision detection package by littlemine (Xinlei Wang).

------------------------------------------------------------------------------
1. Configuration Instructions
------------------------------------------------------------------------------

This project is developed using Visual Studio 2015 and CUDA 9 (>= 8) on Windows platform. It is the source code of the article [Efficient BVH-based Collision Detection Scheme with Ordering and Restructuring](https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.13356).

All configurations are set in directory "Source\Project\Mine\setting\".
All the standalone benchmarks can be downloaded at [UNC Dynamic Scene Benchmarks](http://gamma.cs.unc.edu/DYNAMICB/).

Open "Source\Build\vs2015\Mine\Mine.sln", then build and run it. If the project loading fails, please modify the CUDA version number matching the installed one, and reload the project.

If you plan to use Visual Studio 2017, please make sure build and link the corresponding *assimp* libraries.

------------------------------------------------------------------------------
2. Credits
------------------------------------------------------------------------------

[I-Cloth](http://gamma.cs.unc.edu/CAMA/)

[gProximity](http://gamma.cs.unc.edu/GPUCOL/)

[Assimp](http://www.assimp.org/)

[UNC Dynamic Scene Collection](http://gamma.cs.unc.edu/DYNAMICB/)

------------------------------------------------------------------------------
3. Bug report
------------------------------------------------------------------------------

We would be interested in knowing more about your application as well as any
bugs you may encounter in the collision detection library. You can
report them by sending e-mail to wxlwxl1993@zju.edu.cn or tang_m@zju.edu.cn

------------------------------------------------------------------------------
4. Release date
------------------------------------------------------------------------------

2018/02/10
