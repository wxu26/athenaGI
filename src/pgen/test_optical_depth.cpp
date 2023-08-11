#include <iostream>
#include <sstream>

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

#include "../standalone_physics/optical_depth_rth.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

OpticalDepth OD;

Real high_res_height_rad;
Real high_res_cell_fraction;

Real MeshGen(Real x, RegionSize rs) {
  Real x0 = high_res_cell_fraction;
  Real C = high_res_height_rad/(.5*PI)/x0;
  Real x1 = .5+.5*x0;
  Real gamma = std::log((1.-C*x0)/(1.-x0)/C)/(x1-x0);
  const int n_iter = 8;
  for (int i=0; i<n_iter; ++i) {
    Real f = 1./gamma * (std::exp(gamma*(x1-x0))-1.) + std::exp(gamma*(x1-x0))*(1.-x1) - (1.-C*x0)/C;
    Real df = -1./SQR(gamma) * (std::exp(gamma*(x1-x0))-1.) + 1./gamma * (x1-x0)*std::exp(gamma*(x1-x0)) + (x1-x0)*std::exp(gamma*(x1-x0))*(1.-x1);
    gamma -= f/df;
  }
  bool reflect = (std::abs(rs.x2max-.5*PI)<1.e-8);
  Real y;
  // remap to x=0 at midplane and 1 at pole
  Real sign = 1.;
  if (!reflect) {
    if (x>.5) sign = -1.;
    x = std::abs(x-.5) * 2.;
  } else {
    x = 1.-x;
  }
  // get y
  if (x<x0) y = C*x;
  else if (x<x1) y = C*x0 + C/gamma*(std::exp(gamma*(x-x0))-1.);
  else y = 1.-C*std::exp(gamma*(x1-x0))*(1.-x);
  // map back to coord
  if (!reflect) {
    y = .5-.5*y*sign;
  } else {
    y = 1.-y;
  }
  return y*rs.x2max + (1.-y)*rs.x2min;
}

void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (pin->GetOrAddReal("mesh","x2rat",1.0)>0.) return;
  // below: customized mesh generator function for theta
  high_res_height_rad = pin->GetOrAddReal("problem","high_res_height_rad",0.5);
  high_res_cell_fraction = pin->GetOrAddReal("problem","high_res_cell_fraction",0.5);
  EnrollUserMeshGenerator(X2DIR,MeshGen);
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real x0 = pin->GetOrAddReal("problem","x0",0.);
  Real y0 = pin->GetOrAddReal("problem","y0",0.);
  Real z0 = pin->GetOrAddReal("problem","z0",0.);
  Real R0 = pin->GetOrAddReal("problem","R0",1.);
  Real tauc = pin->GetOrAddReal("problem","tauc",1.);
  Real dens = tauc/R0;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real x,y,z;
        if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
          Real r = pcoord->x1v(i);
          Real th = pcoord->x2v(j);
          Real phi = pcoord->x3v(k);
          x = r*std::sin(th)*std::cos(phi);
          y = r*std::sin(th)*std::sin(phi);
          z = r*std::cos(th);
        } else {
          std::stringstream msg;
          msg << "### FATAL ERROR in test_optical_depth.cpp ProblemGenerator" << std::endl
              << "Unsupported coordinate system " << COORDINATE_SYSTEM << std::endl;
          ATHENA_ERROR(msg);
        }
        Real d = std::sqrt(SQR(x-x0) + SQR(y-y0) + SQR(z-z0));
        phydro->u(IDN,k,j,i) = (d<=R0) ? dens : 0.;
        phydro->u(IM1,k,j,i) = 0.;
        phydro->u(IM2,k,j,i) = 0.;
        phydro->u(IM3,k,j,i) = 0.;
      }
    }
  }

  OD.keep_tau_cc = true;
  OD.Initialize(this);
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        OD.rho_kappa(k,j,i) = phydro->u(IDN,k,j,i);
      }
    }
  }
  OD.EstimateOpticalDepth();
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        phydro->u(IM1,k,j,i) = OD.weights_and_tau(0,k,j,i);
        phydro->u(IM2,k,j,i) = OD.weights_and_tau(1,k,j,i);
        phydro->u(IM3,k,j,i) = OD.weights_and_tau(2,k,j,i);
      }
    }
  }

  return;
}