#include <iostream>
#include <sstream>
#include <algorithm>    // std::min

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../inputs/hdf5_reader.hpp"  // HDF5ReadRealArray()
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"

#include "../gravity/gravity.hpp"
#include "../gravity/sph_gravity.hpp"

#include "../standalone_physics/optical_depth_rth.hpp"

#include "../nr_radiation/radiation.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

// unit system:
// Msun = au = G = 1 (so unit time = year/2pi)

//========================================================================================
// Parameters - read from input
//========================================================================================

// grid - used only for user-defined grid

Real nth_lo; // # of low/mid res cells (for half pi)
Real nth_hi; // # of high res cells (for half pi)
Real h_hi; // height of high res region (in rad)
Real dth_pole; // cell size at pole
int l0 = 1, m0 = 1;
  // the domain is 1/l0 of [0,pi] in theta, and 1/m0 of [0,2pi] in phi
  // these are computed automatically

// initial conditions

Real GM = 1;

Real R_in = 0;
Real R_out = 20; // inner/outer disk radii
Real Sigma_0 = 1;
Real Sigma_slope = -1;
Real T_0 = 0.01;
Real T_slope = -1;

// velocity damping

Real beta_damp = -1; // turn on velocity damping if beta_damp, beta_damp_r, or beta_damp_th>0
Real beta_damp_r = -1; // if no user input: default to beta_damp
Real beta_damp_th = -1; // if no user input: default to beta_damp

// cooling

Real beta_cool = -1; // turn on beta cooling if beta_cool>0
Real T_floor_0 = 0; // cool towards this floor
Real T_floor_slope = 0;

//========================================================================================
// Forward declarations
//========================================================================================

void MySource(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar);

void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void MidplaneHyd(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
Real MeshGen(Real x, RegionSize rs);

//========================================================================================
// Initialization
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  
  // grid
  l0 = std::round(PI/(pin->GetReal("mesh","x2max")-pin->GetReal("mesh","x2min")));
  m0 = std::round(2.*PI/(pin->GetReal("mesh","x3max")-pin->GetReal("mesh","x3min")));
  if (pin->GetOrAddReal("mesh","x2rat",1.0)<0.) { // turn on user-defined grid
    nth_lo   = pin->GetReal("mesh","nth_lo");
    nth_hi   = pin->GetReal("mesh","nth_hi");
    h_hi     = pin->GetReal("mesh","h_hi");
    dth_pole = pin->GetReal("mesh","dth_pole");
    EnrollUserMeshGenerator(X2DIR,MeshGen);
  }
  
  // initial conditions
  GM          = pin->GetOrAddReal("problem","GM",GM); // point source gravity is automatically turned on
  R_in        = pin->GetOrAddReal("problem","R_in",R_in);
  R_out       = pin->GetOrAddReal("problem","R_out",R_out);
  Sigma_0     = pin->GetOrAddReal("problem","Sigma_0",Sigma_0);
  Sigma_slope = pin->GetOrAddReal("problem","Sigma_slope",Sigma_slope);
  T_0         = pin->GetOrAddReal("problem","T_0",T_0);
  T_slope     = pin->GetOrAddReal("problem","T_slope",T_slope);

  // velocity damping
  beta_damp = pin->GetOrAddReal("problem","beta_damp",beta_damp);
  beta_damp_r = pin->GetOrAddReal("problem","beta_damp_r",beta_damp);
  beta_damp_th = pin->GetOrAddReal("problem","beta_damp_th",beta_damp);

  // cooling
  beta_cool     = pin->GetOrAddReal("problem","beta_cool",beta_cool);
  T_floor_0     = pin->GetOrAddReal("problem","T_floor_0",T_floor_0);
  T_floor_slope = pin->GetOrAddReal("problem","T_floor_slope",T_floor_slope);

  // source term
  EnrollUserExplicitSourceFunction(MySource);
  // boundary conitions
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, DiskInnerX1);
  }
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiskOuterX1);
  }
  if (mesh_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x2, MidplaneHyd);
  }
}

//========================================================================================
// Problem generator
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // TODO: sanity check for spherical polar
  // Sigma = power law between R_in and R_out, zero otherwise
  const Real dfloor = peos->GetDensityFloor();
  const Real gamma = peos->GetGamma();
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real R, z;

        Real r = pcoord->x1v(i);
        Real th = pcoord->x2v(j);
        R = r*std::sin(th);
        z = r*std::cos(th);

        Real Sigma = Sigma_0 * std::pow(R, Sigma_slope);
        if (R<R_in || R>R_out) Sigma = 0;

        Real T = T_0 * std::pow(R, T_slope);
        Real H = std::sqrt(T) / std::sqrt(GM/(R*R*R)); // computed using isothermal sound speed cs_iso = sqrt(T)
        Real rho_mid = Sigma/H/std::sqrt(2.*PI);
        phydro->u(IDN,k,j,i) = std::max(dfloor, rho_mid * std::exp(-.5*SQR(z/H))); // just a hard-coded density floor
        phydro->u(IM1,k,j,i) = 0.;
        phydro->u(IM2,k,j,i) = 0.;
        phydro->u(IM3,k,j,i) = phydro->u(IDN,k,j,i) * std::sqrt(R*R/(r*r*r));
        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) = phydro->u(IDN,k,j,i)*T/(gamma-1.)
           + .5*(SQR(phydro->u(IM1,k,j,i)) + SQR(phydro->u(IM2,k,j,i)) + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
        }
      }
    }
  }
}

//========================================================================================
// Source term
//========================================================================================

void MySource(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar) {
  const Real gm1 = pmb->peos->GetGamma()-1.;
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        Real r = pmb->pcoord->x1v(i);
        Real th = pmb->pcoord->x2v(j);
        Real R = r*std::sin(th);
        // beta cooling
        if (beta_cool>0.) {
          Real cooling_rate = std::sqrt(GM/R/R/R)/beta_cool;
          Real T_floor = T_floor_0 * std::pow(R, T_floor_slope);
          cons(IEN,k,j,i) -= (prim(IPR,k,j,i)-prim(IDN,k,j,i)*T_floor) / gm1 * (1-std::exp(-dt*cooling_rate));
        }
        // velocity damping
        if (beta_damp_r>0.) {
          Real damping_rate_r = std::sqrt(GM/R/R/R)/beta_damp_r;
          cons(IM1,k,j,i) *= std::exp(-dt*damping_rate_r);
        }
        if (beta_damp_th>0.) {
          Real damping_rate_th = std::sqrt(GM/R/R/R)/beta_damp_th;
          cons(IM2,k,j,i) *= std::exp(-dt*damping_rate_th);
        }
      }
    }
  }
}

//========================================================================================
// Boundary conditions
//========================================================================================

// outer: outflow
// surface density will sligtly change (~10%) near boundaries.
// I have tried many other options, none is substantially better.
// that's because we don't physically know what to expect for disk "outside" the boundary.
void DiskOuterX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        prim(IDN,k,j,iu+i) = prim(IDN,k,j,iu);
        prim(IVX,k,j,iu+i) = prim(IVX,k,j,iu);
        prim(IVY,k,j,iu+i) = prim(IVY,k,j,iu);
        prim(IVZ,k,j,iu+i) = prim(IVZ,k,j,iu);
        prim(IPR,k,j,iu+i) = prim(IPR,k,j,iu);
      }
    }
  }
}

// inner: outflow
// surface density will sligtly change (~10%) near boundaries.
// I have tried many other options, none is substantially better.
// that's because we don't physically know what to expect for disk "outside" the boundary.
void DiskInnerX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        prim(IDN,k,j,il-i) = prim(IDN,k,j,il);
        prim(IVX,k,j,il-i) = prim(IVX,k,j,il);
        prim(IVY,k,j,il-i) = prim(IVY,k,j,il);
        prim(IVZ,k,j,il-i) = prim(IVZ,k,j,il);
        prim(IPR,k,j,il-i) = prim(IPR,k,j,il);
      }
    }
  }
}

// midplane: reflecting
void MidplaneHyd(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=il; i<=iu; ++i) {
        prim(IDN,k,ju+j,i) = prim(IDN,k,ju-j+1,i);
        prim(IVX,k,ju+j,i) = prim(IVX,k,ju-j+1,i);
        prim(IVY,k,ju+j,i) = -prim(IVY,k,ju-j+1,i);
        prim(IVZ,k,ju+j,i) = prim(IVZ,k,ju-j+1,i);
        prim(IPR,k,ju+j,i) = prim(IPR,k,ju-j+1,i);
      }
    }
  }
}

//========================================================================================
// Mesh generator
//========================================================================================
Real MeshGen(Real x, RegionSize rs) {
  // variables set externally: nth_lo, nth_hi, h_hi, dth_pole

  // sanity check: is the resolution correct?
  bool reflect = (std::abs(rs.x2max-.5*PI)<1.e-8);
  if ((2-reflect) * (nth_lo + nth_hi) != rs.nx2) {
    std::stringstream msg;
    msg << "### FATAL ERROR in MeshGen" << std::endl
        << "input nth_lo and nth_hi do not agree with grid";
    ATHENA_ERROR(msg);
  }

  Real dth_mid = h_hi/nth_hi;
  Real A = PI*.5-h_hi;
  Real gamma = (dth_pole*std::log(dth_pole/dth_mid)+dth_mid-dth_pole) / (nth_lo*dth_pole-A);
  
  // remap to x=0 at midplane and 1 at pole
  Real sign = 1.;
  if (!reflect) {
    if (x>.5) sign = -1.;
    x = std::abs(x-.5) * 2.;
  } else {
    x = 1.-x;
  }
  // remap to x = nth_lo+nth_hi at pole
  x *= nth_lo+nth_hi;
  // y: 0 at midplane and pi/2 at pole
  Real y;
  if (x<=nth_hi) y = x*dth_mid;
  else if (x<=nth_hi+std::log(dth_pole/dth_mid)/gamma) y = nth_hi*dth_mid + 1./gamma*(std::exp((x-nth_hi)*gamma)-1.)*dth_mid;
  else y = PI*.5 - dth_pole*(nth_hi+nth_lo-x);
  // remap y
  y /= (PI*.5);
  if (!reflect) {
    y = .5-.5*y*sign;
  } else {
    y = 1.-y;
  }
  return y*rs.x2max + (1.-y)*rs.x2min;
}