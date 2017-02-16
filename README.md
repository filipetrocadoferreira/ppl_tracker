# Simple Multi-Camera Multi-Target Tracker
This code Implements a simple method to track multiple person in a multi (overlapping) cameras.
The code performs:
* Detection using Fastes Detector in the West (c++ version) - https://github.com/apennisi/fastestpedestriandetectorinthewest
* Transformation to World Coordinates
* Data Association using Hungarian Algorithm - https://github.com/mcximing/hungarian-algorithm-cpp
* Tracking in world coordinates using a version of SORT tracker - https://github.com/abewley/sort 


##v0.2
* new detector parameters (faster and better recall)
* no more "entry and leave points". Now constraints to generate/delete tracklet are more general (room dimensions)
* draw of the trajectories



##Requirements
* OpenCV
* OpenMP
* SSE
* Boost

## How to build

```bash
mkdir build
cd build
cmake ../
make -j\<number-of-cores+1\>
```

Save the files under /Files folder

```bash
/Files
 |__arc
 |__triangles
 |__...
```



##How to use



```bash
./tracker <number_of_test>
```
<number_of_test> - > number from 1 to 5 corresponding to one of each test. -1 to perform all tests.
Results are saved under /Results folder


