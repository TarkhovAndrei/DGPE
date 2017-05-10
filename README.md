# A solver for the discrete Gross-Pitaevskii equation in one, two and three dimensions

A tool for modelling of the system of coupled Bose-Einstein condensates on a lattice with periodic boundary conditions in 1D, 2D and 3D.

A Runge-Kutta 4th order algorithm is employed. Quadrupole precision numbers are used for all calculations. 

The standard routine for the largest Lyapunov exponent calculation is implemented, 
as well as a custom Loschmidt echo (imperfect time-reversal) routine, for the lattice of coupled Bose-Einstein condensates.

-----------------------------------------
Copyright <2017> <Andrei E. Tarkhov, Skolkovo Institute of Science and Technology,
https://github.com/TarkhovAndrei/DGPE>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following 2 conditions:

1) If any part of the present source code is used for any purposes followed by publication of obtained results,
the citation of the present code shall be provided according to the rule:

    "Andrei E. Tarkhov, Skolkovo Institute of Science and Technology,
    source code from the GitHub repository https://github.com/TarkhovAndrei/DGPE
    was used to obtain the presented results, 2017."

2) The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
