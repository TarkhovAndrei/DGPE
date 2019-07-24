# A solver for the Gross-Pitaevskii equation on a one-, two- or three- dimensional lattice

A tool for modelling the system of coupled Bose-Einstein condensates on a lattice with periodic boundary conditions in 1D, 2D and 3D.

The 4-th order Runge-Kutta algorithm with a fixed time step is employed. The code supports double and quadrupole precision numbers.

The standard routine for the largest Lyapunov exponent calculation is implemented, 
as well as a custom Loschmidt echo (imperfect time-reversal) routine, for the lattice of coupled Bose-Einstein condensates.

The code was used for obtaining numerical results for the papers:

Andrei E. Tarkhov, Sandro Wimberger, and Boris V. Fine, Phys. Rev. A 96, 023624 (2017), https://doi.org/10.1103/PhysRevA.96.023624

Andrei E. Tarkhov, Boris V. Fine, New J. Phys. 20 123021 (2018), https://doi.org/10.1088/1367-2630/aaf0b6

-----------------------------------------
Copyright <2019> <Andrei E. Tarkhov, Skolkovo Institute of Science and Technology,
https://github.com/TarkhovAndrei/DGPE>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following 2 conditions:

1) If any part of the present source code is used for any purposes followed by publication of obtained results,
the citation of the present code shall be provided according to the rule:

    "Andrei E. Tarkhov, Skolkovo Institute of Science and Technology,
    source code from the GitHub repository https://github.com/TarkhovAndrei/DGPE
    was used to obtain the presented results, 2019."

2) The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.