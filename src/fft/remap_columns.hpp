#ifndef FFT_REMAP_COLUMNS_HPP_
#define FFT_REMAP_COLUMNS_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file RemapColumns.hpp
//! \brief remap the domain to allow operations on global columns along each direction

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/cc/bvals_cc.hpp"

#ifdef MPI_PARALLEL
#include "./plimpton/remap_3d.h"
#include <mpi.h>
#endif // MPI_PARALLEL

class MeshBlock;

//----------------------------------------------------------------------------------------
//! \class RemapColumns
//! \brief information and functions for performing remaps and setting boundaries

// Important limitations:
//  - only works with MPI (no OpenMP, no single-threaded)
//  - only works with exactly one meshblock per process
//  - does not support mesh refinement

// Other notes:
//  - remaps are performed between four layouts indexed as follows:
//    0: same layout as meshblock (index in ijk for slow->fast)
//    1: global columns along x1 direction (ijk)
//    2: global columns along x2 direction (jki)
//    3: global columns along x3 direction (kij)
//  - this class is mainly used to derive classes for physics that require global solves
//    (e.g., radial ray-tracing, gravity)

class RemapColumns{
  public:
    void Initialize(MeshBlock * my_pmb);
    void Initialize(MeshBlock * my_pmb,
         int npj1, int npk1, int npi2, int npk2, int npi3, int npj3);
        // manually specify the remap layout
        // np(ijk)(123) sets the number of processes along axis ijk for layout 123
        // set to 0 to use default value
    void InitializeBoundary(AthenaArray<Real> * data);
        // set the data on which boundary communications will be performed
        // data can be a 3D or 4D array (the latter allows multiple variables)
    bool Initialized() {return initialized_;}
    bool InitializedBoundary() {return initialized_boundary_;}
    void SetBoundaries();
        // set block and non-user physical boundaries
        // Since the rest of this class alredy requires unrefined mesh with one block
        // per process, the communications are easier than standard athena++ boundary
        // communications and we do not require a tasklist.
        // This is particularly useful for adding ad-hoc physics in the problem generator.
    void Remap(const int layout_in, const int layout_out, Real * in, Real * out);
    bool Test(const int layout_in, const int layout_out, const int axis, const int print); // test remap
    void Test(); // perform all tests
    void Print(const int layout, Real * data); // print data in given layout (mainly used for debug)
  protected:
    MeshBlock * pmb;
    AthenaArray <Real> x[3]; // global mesh
    AthenaArray <Real> xf[3]; // global mesh
    int N[3]; // total number of cells along each direction
    int n[4][3]; // number of cells for each layout along each direction
    int np[4][3]={0}; // number of processes along each diection
    int is[4][3]; // start index on global mesh
    int ie[4][3]; // end index on global mesh
    int cnt; // total number of cells
  private:
    bool initialized_ = false;
    bool initialized_boundary_ = false;

    #ifdef MPI_PARALLEL
    MPI_Comm MPI_COMM_DECOMP;
    // plan_[i][j] = remap plan from layout i to layout j
    struct remap_plan_3d * plan_[4][4];
    #endif // MPI_PARALLEL

    Real *buf_; // buffer for remaps
    CellCenteredBoundaryVariable * bvar; // for boundary communication
    void InitializeInd();           // only called in Initialize
    void InitializeGrid();          // only called in Initialize
    void InitializePlans();         // only called in Initialize
    void ApplyPhysicalBoundaries(); // only called in SetBoundaries
};

#endif // FFT_REMAP_COLUMNS_HPP_