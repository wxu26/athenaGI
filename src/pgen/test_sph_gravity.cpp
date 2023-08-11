#include <iostream>
#include <sstream>

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

#include "../gravity/gravity.hpp"
#include "../gravity/sph_gravity.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

const int IA = 0; // index for analytic solution in ruser_meshblock_data
const int IE = 1; // index for error in ruser_meshblock_data

Real high_res_height_rad;
Real high_res_cell_fraction;

Real MeshGen(Real x, RegionSize rs) {
  //Real high_res_height_rad = 0.3;//pin->GetOrAddReal("problem","high_res_height_rad",0.1);
  //Real high_res_cell_fraction = 0.5;//pin->GetOrAddReal("problem","high_res_cell_fraction",0.5);
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
  Real four_pi_G = pin->GetReal("problem","four_pi_G");
  SetFourPiG(four_pi_G);

  if (pin->GetOrAddReal("mesh","x2rat",1.0)>0.) return;
  // below: customized mesh generator function for theta
  high_res_height_rad = pin->GetOrAddReal("problem","high_res_height_rad",0.5);
  high_res_cell_fraction = pin->GetOrAddReal("problem","high_res_cell_fraction",0.5);
  EnrollUserMeshGenerator(X2DIR,MeshGen);
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  AllocateRealUserMeshBlockDataField(2);
  ruser_meshblock_data[0].NewAthenaArray(ncells3,ncells2,ncells1);
  ruser_meshblock_data[1].NewAthenaArray(ncells3,ncells2,ncells1);
  return;
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real x0 = pin->GetOrAddReal("problem","x0",0.);
  Real y0 = pin->GetOrAddReal("problem","y0",0.);
  Real z0 = pin->GetOrAddReal("problem","z0",0.);
  Real R0 = pin->GetOrAddReal("problem","R0",1.);
  Real dens = 1./(pmy_mesh->four_pi_G_/3*R0*R0*R0); // so GM = 1
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real x,y,z;
        if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
          x = pcoord->x1v(i);
          y = pcoord->x2v(j);
          z = pcoord->x3v(k);
        } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
          Real r = pcoord->x1v(i);
          Real th = pcoord->x2v(j);
          Real phi = pcoord->x3v(k);
          x = r*std::sin(th)*std::cos(phi);
          y = r*std::sin(th)*std::sin(phi);
          z = r*std::cos(th);
        } else {
          std::stringstream msg;
          msg << "### FATAL ERROR in test_poisson.cpp ProblemGenerator" << std::endl
              << "Unsupported coordinate system " << COORDINATE_SYSTEM << std::endl;
          ATHENA_ERROR(msg);
        }
        Real d = std::sqrt(SQR(x-x0) + SQR(y-y0) + SQR(z-z0));
        phydro->u(IDN,k,j,i) = (d<=R0) ? dens : 0.;
        phydro->u(IM1,k,j,i) = 0.;
        phydro->u(IM2,k,j,i) = 0.;
        phydro->u(IM3,k,j,i) = 0.;
        // analytic solution
        ruser_meshblock_data[IA](k,j,i) = (d>R0) ? -1./d : -1./R0 + .5*(SQR(d)-SQR(R0))/(R0*R0*R0);
      }
    }
  }
  //std::cout<<"finish pgen"<<std::endl;
  return;
}

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  //std::cout<<"start UWAP"<<std::endl;
  MeshBlock *pmb = my_blocks(0);
  int is = pmb->is, ie = pmb->ie;
  int js = pmb->js, je = pmb->je;
  int ks = pmb->ks, ke = pmb->ke;
  // compute phi & check performance
  int ncycle = pin->GetOrAddInteger("problem","ncycle",1);
  for (int n=0; n < ncycle; n++) {
    for (int b=0; b<nblocal; ++b) {
      pmb = my_blocks(b);
      pmb->pgrav->phi.ZeroClear();
    }
    psgrd->Solve(1);
  }
  // check error
  // step 1. compute total mass
  Real M = 0.;
  for (int b=0; b<nblocal; ++b) {
    pmb = my_blocks(b);
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          M += pmb->phydro->u(IDN,k,j,i) * pmb->pcoord->GetCellVolume(k,j,i);
        }
      }
    }
  }
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &M, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  Real GM = four_pi_G_/(4*PI)*M;
  // step 2. compute error
  //std::cout<<"start err"<<std::endl;
  Real err_tot = 0.;
  Real err_tot_vol = 0.;
  Real nzones = 0;
  Real vol = 0;
  Real max_phi = 0.;
  for (int b=0; b<nblocal; ++b) {
    pmb = my_blocks(b);
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          pmb->ruser_meshblock_data[IE](k,j,i) = pmb->pgrav->phi(k,j,i)/GM-pmb->ruser_meshblock_data[IA](k,j,i);
          err_tot += std::abs(pmb->ruser_meshblock_data[IE](k,j,i));
          err_tot_vol += std::abs(pmb->ruser_meshblock_data[IE](k,j,i))*pmb->pcoord->GetCellVolume(k,j,i);
          nzones += 1;
          vol += pmb->pcoord->GetCellVolume(k,j,i);
          max_phi = fmax(max_phi, std::abs(pmb->ruser_meshblock_data[IA](k,j,i)));
          // also save data to prim
          pmb->phydro->w(IVX,k,j,i) = pmb->ruser_meshblock_data[IA](k,j,i)*GM;
          pmb->phydro->w(IVY,k,j,i) = pmb->ruser_meshblock_data[IE](k,j,i);
        }
      }
    }
  }
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &err_tot, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &err_tot_vol, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &nzones, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &vol, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &max_phi, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  Real err_cell_mean = err_tot/nzones;
  Real err_vol_mean = err_tot_vol/vol;
  if (Globals::my_rank==0) {
    std::cout<<"cell avg L1 ="<<err_cell_mean<<std::endl;
    std::cout<<" vol avg L1 ="<<err_vol_mean <<std::endl;
    std::cout<<"max abs phi ="<<max_phi      <<std::endl;
  }
}