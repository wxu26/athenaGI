#ifndef GRAVITY_SPH_GRAVITY_HPP_
#define GRAVITY_SPH_GRAVITY_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file sph_gravity.hpp
//! \brief gravity solver for spherical-polar cooordinate based on a discrete spherical
//!        harmonics decomposition

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../fft/remap_columns.hpp"
#ifdef FFT
#include "fftw3.h"
#endif

class MeshBlock;
class ParameterInput;
class GravityBoundaryTaskList;
class SphGravity;

using UpdateStarFn = void (*)(SphGravity * grav, MeshBlock * pmb);

//----------------------------------------------------------------------------------------
//! \class SphGravity
//! \brief spherical-polar gravity solver for each block

class SphGravity: public RemapColumns {
  public:
    Real four_pi_G;
    Real M_star=0, x_star=0, y_star=0, z_star=0, r_smooth=0;
        // mass and location of point mass; smoothing length
        // these can be updated, for example, in UserWorkInLoop()
    void Initialize(MeshBlock * my_pmb, Real four_pi_G);
    void Solve(); // solve discrete Poisson equation under vacuum boundary conditions
    void SetBoundaries();
    void AddPointMass();

  private:
    
    Real dlnr; // radial spacing; we require log-uniform r grid

    // eigenvalues for theta and phi derivatives
    AthenaArray <Real> dsq_phi_eigen;
        // eigenvalues for d^2/dphi^2; -m^2 for continuous
    AthenaArray <Real> dsq_thphi_eigen;
        // eigenvalues for del^2(th,phi); -l(l+1) for continuous; indexed by (l,m)
    
    // fft plans in phi direction
    #ifdef FFT
    fftw_plan r2r_fft_fwd;
    fftw_plan r2r_fft_bck;
    #endif
    
    // matrices for mapping to and from theta basis (discrete version of P_lm)
    AthenaArray <Real> grid_to_basis;
    AthenaArray <Real> basis_to_grid;
    
    // LU decomposition for the linear problem in r
    AthenaArray <Real> LU;

    Real * s0, *s1; // scracth arrays

    void LoadSource(Real * src);
    void RFFT(Real * data, int dir);
    void GridToBasis(Real * data_in, Real * data_out, int dir);
    void SolveR(Real * src, Real * phi);
    void StorePhi(Real * phi);
    void RadialBoundaries();

    void InitializeR();
    void InitializeTh();
    void InitializePhi();
};


//----------------------------------------------------------------------------------------
//! \class SphGravityDriver
//! \brief driver for spherical-polar gravity solver

class SphGravityDriver {
  public:
    SphGravityDriver(Mesh *pm, ParameterInput *pin);
    ~SphGravityDriver();
    void Solve(int stage);
    void UpdateStar(Real M_star) {grav_.M_star = M_star;}
    void UpdateStar(Real M_star, Real x_star, Real y_star, Real z_star) {
      grav_.M_star = M_star;
      grav_.x_star = x_star; grav_.y_star = y_star; grav_.z_star = z_star;
    }
    void UpdateStar(Real M_star, Real x_star, Real y_star, Real z_star, Real r_smooth) {
      grav_.M_star = M_star;
      grav_.x_star = x_star; grav_.y_star = y_star; grav_.z_star = z_star;
      grav_.r_smooth = r_smooth;
    }
    void EnrollUpdateStarFn(UpdateStarFn my_update_star_fn) {
      update_star_fn_ = my_update_star_fn;
      use_update_star_fn_ = true;
    }

  private:
    Real four_pi_G_;
    //GravityBoundaryTaskList *gtlist_;
        // the boundaries are set with RemapColumns::SetBoundaries() insetad 
    Mesh *pmy_mesh_;
    SphGravity grav_;
    bool one_solve_per_cycle_ = false; // if true, only run solver at stage=0
    bool use_update_star_fn_ = false; // if ture, use a user-defined function to update 
                                      // stellar properties before each solve
    UpdateStarFn update_star_fn_;
};

#endif // GRAVITY_SPH_GRAVITY_HPP_
