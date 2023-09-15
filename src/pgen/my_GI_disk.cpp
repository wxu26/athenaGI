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

#include "../gravity/gravity.hpp"
#include "../gravity/sph_gravity.hpp"

#include "../standalone_physics/optical_depth_rth.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

const Real Mtot = 1.; // keep this fixed
const Real G = 1.; // keep this fixed; do not read four_pi_G from input

Real Rd = 1.; // outer edge of disk; inned edge always at r->0
Real Rd_in = 0.4;
Real Md = 0.1; //0.1; // disk mass
Real Sigma_slope = -1;
Real Sigma_d = Md/(2.*PI*Rd*Rd)*(2+Sigma_slope); // Sigma at Rd
Real Qiso_d = 2/std::sqrt(1.6667); // 2/std::sqrt(1.6667); // Q at Rd, evaluated using isothermal sound speed
Real T_slope = -0.5;
Real T_d = SQR(PI*G*Sigma_d*Qiso_d/std::sqrt(G*Mtot/Rd/Rd/Rd));

bool read_from_2d;

Real tau_cool = -1; // linear cooling; <0 for no cooling
Real hypercool_density_threshold = 1.e-8; // <0 to turn off hypercooling at low density

bool inject_perturbation=false;
Real perturbation_level=1.e-4;
int perturbation_m_min=1;
int perturbation_m_max=6;
int random_seed = 2023;

bool relaxation = false;

Real dfloor; // density floor (to be read from input)

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

void GetStellarMassAndLocation(MeshBlock * pmb, Real Mtot, Real & M_star, Real & x_star, Real & y_star, Real & z_star);

// th grid function
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

// init mesh

void Mesh::InitUserMeshData(ParameterInput *pin) {
  //Real four_pi_G = pin->GetReal("problem","four_pi_G");
  SetFourPiG(4.*PI*G);

  EnrollUserExplicitSourceFunction(MySource);

  read_from_2d = pin->GetOrAddBoolean("problem","read_from_2d",read_from_2d);

  tau_cool = pin->GetOrAddReal("problem","tau_cool",tau_cool);

  inject_perturbation = pin->GetOrAddBoolean("problem","inject_perturbation",inject_perturbation);
  perturbation_level = pin->GetOrAddReal("problem","perturbation_level",perturbation_level);

  relaxation = pin->GetOrAddBoolean("problem","relaxation",relaxation);

  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, DiskInnerX1);
  }
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiskOuterX1);
  }

  if (pin->GetOrAddReal("mesh","x2rat",1.0)>0.) return;
  // below: customized mesh generator function for theta
  high_res_height_rad = pin->GetOrAddReal("problem","high_res_height_rad",0.5);
  high_res_cell_fraction = pin->GetOrAddReal("problem","high_res_cell_fraction",0.5);
  EnrollUserMeshGenerator(X2DIR,MeshGen);

  AllocateRealUserMeshDataField(1);
  ruser_mesh_data[0].NewAthenaArray(4); // stellar properties
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  // update stellar properties - this is only relevant for restarts
  Real M_star, x_star, y_star, z_star;
  M_star = pmy_mesh->ruser_mesh_data[0](0);
  x_star = pmy_mesh->ruser_mesh_data[0](1);
  y_star = pmy_mesh->ruser_mesh_data[0](2);
  z_star = pmy_mesh->ruser_mesh_data[0](3);
  pmy_mesh->psgrd->UpdateStar(M_star, x_star, y_star, z_star);
}

// initialize a disk

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real R, z;

        Real r = pcoord->x1v(i);
        Real th = pcoord->x2v(j);
        R = r*std::sin(th);
        z = r*std::cos(th);

        Real Sigma = (R<=Rd && R>Rd_in) ? Sigma_d * std::pow(R/Rd, Sigma_slope) : 0.;
        Real T = T_d * std::pow(R/Rd, T_slope);
        Real MR = Mtot; 
        if (R<Rd) MR = Mtot - 2.*PI * Sigma_d * (1.-std::pow(R/Rd, 2+Sigma_slope)) / (2+Sigma_slope); // M(<R)
        Real H = std::sqrt(T) / std::sqrt(G*MR/(R*R*R)); // computed using isothermal sound speed cs_iso = sqrt(T)
        Real rho_mid = Sigma/H/std::sqrt(2.*PI);
        phydro->u(IDN,k,j,i) = rho_mid * std::exp(-.5*SQR(z/H));
        phydro->u(IM1,k,j,i) = 0.;
        phydro->u(IM2,k,j,i) = 0.;
        phydro->u(IM3,k,j,i) = phydro->u(IDN,k,j,i) * std::sqrt(1./R);
        const Real gamma = peos->GetGamma();
        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) = phydro->u(IDN,k,j,i)*T/(gamma-1.)
           + .5*(SQR(phydro->u(IM1,k,j,i)) + SQR(phydro->u(IM2,k,j,i)) + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
        }
      }
    }
  }

  if (read_from_2d) {
    std::string input_filename = pin->GetString("problem", "input_filename");
    std::string dataset_cons = "cons";
    // make a global array
    AthenaArray<Real> u_global;
    int nx1_global = pmy_mesh->mesh_size.nx1 + 2*NGHOST;
    int nx2_global = pmy_mesh->mesh_size.nx2 + 2*NGHOST;
    int nx3_global = 1;
    u_global.NewAthenaArray(NHYDRO,nx3_global,nx2_global,nx1_global);
    // read data
    int start_cons_file[5];
    start_cons_file[1] = 0; // gid
    start_cons_file[2] = 0;
    start_cons_file[3] = 0;
    start_cons_file[4] = 0;
    int start_cons_indices[5];
    start_cons_indices[IDN] = 0;
    start_cons_indices[IM1] = 2;
    start_cons_indices[IM2] = 3;
    start_cons_indices[IM3] = 4;
    start_cons_indices[IEN] = 1;
    int count_cons_file[5];
    count_cons_file[0] = 1;
    count_cons_file[1] = 1;
    count_cons_file[2] = 1;
    count_cons_file[3] = pmy_mesh->mesh_size.nx2;
    count_cons_file[4] = pmy_mesh->mesh_size.nx1;
    int start_cons_mem[4];
    start_cons_mem[1] = 0;
    start_cons_mem[2] = js;
    start_cons_mem[3] = is;
    int count_cons_mem[4];
    count_cons_mem[0] = 1;
    count_cons_mem[1] = 1;
    count_cons_mem[2] = pmy_mesh->mesh_size.nx2;
    count_cons_mem[3] = pmy_mesh->mesh_size.nx1;
    for (int n = 0; n < NHYDRO; ++n) {
      start_cons_file[0] = start_cons_indices[n];
      start_cons_mem[0] = n;
      HDF5ReadRealArray(input_filename.c_str(), dataset_cons.c_str(), 5, start_cons_file,
                        count_cons_file, 4, start_cons_mem,
                        count_cons_mem, u_global, true);
      void HDF5ReadRealArray(const char *filename, const char *dataset_name, int rank_file,
                       const int *start_file, const int *count_file, int rank_mem,
                       const int *start_mem, const int *count_mem,
                       AthenaArray<Real> &array,
                       bool collective=false, bool noop=false);
    }
    // dump data into grid
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          int kg, jg, ig;
          kg = 0;
          jg = j + block_size.nx2*loc.lx2;
          ig = i + block_size.nx1*loc.lx1;
          for (int n = 0; n < NHYDRO; ++n)
            phydro->u(n,k,j,i) = u_global(n,kg,jg,ig);
        }
      }
    }
    // delete scratch array
    u_global.DeleteAthenaArray();
  }


  Real M_star, x_star, y_star, z_star;
  GetStellarMassAndLocation(this, Mtot, M_star, x_star, y_star, z_star);
  pmy_mesh->psgrd->UpdateStar(M_star, x_star, y_star, z_star);
  std::cout<<M_star<<" "<<Globals::my_rank;
}

// cooling: beta cooling

void MySource(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar) {
  const Real gm1 = pmb->peos->GetGamma()-1.;
  if (relaxation) {
    for (int k = pmb->ks; k <= pmb->ke; ++k) {
      for (int j = pmb->js; j <= pmb->je; ++j) {
        for (int i = pmb->is; i <= pmb->ie; ++i) {
          Real r = pmb->pcoord->x1v(i);
          Real th = pmb->pcoord->x2v(j);
          Real R = r*std::sin(th);
          Real p_goal = prim(IDN,k,j,i) * T_d * std::pow(R/Rd, T_slope);
          Real v = std::sqrt(SQR(prim(IVX,k,j,i))+SQR(prim(IVY,k,j,i)));
          Real vK = std::sqrt(G*Mtot/R);
          Real damping_rate = 10. * std::sqrt(G*Mtot/R/R/R) * std::max(1., v/(0.01*vK));
          cons(IEN,k,j,i) -= (prim(IPR,k,j,i)-p_goal) / gm1 * (1-std::exp(-dt*damping_rate));
          cons(IM1,k,j,i) -= prim(IDN,k,j,i) * prim(IVX,k,j,i) * (1.-std::exp(-dt*damping_rate));
          cons(IM2,k,j,i) -= prim(IDN,k,j,i) * prim(IVY,k,j,i) * (1.-std::exp(-dt*damping_rate));
        }
      }
    }
    return;
  }
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        Real r = pmb->pcoord->x1v(i);
        Real th = pmb->pcoord->x2v(j);
        Real R = r*std::sin(th);
        Real temp = prim(IPR,k,j,i) / prim(IDN,k,j,i);
        Real cooling_rate = 0.;
        if (tau_cool>0.) cooling_rate += 1/tau_cool; // beta cooling
        cooling_rate += hypercool_density_threshold/prim(IDN,k,j,i);
        cooling_rate *= std::sqrt(G*Mtot/R/R/R);
        cons(IEN,k,j,i) -= prim(IPR,k,j,i) / gm1 * (1-std::exp(-dt*cooling_rate));
      }
    }
  }
  return;
}

void MeshBlock::UserWorkInLoop() {
  // update stellar mass
  Real M_star, x_star, y_star, z_star;
  GetStellarMassAndLocation(this, Mtot, M_star, x_star, y_star, z_star);
  pmy_mesh->psgrd->UpdateStar(M_star, x_star, y_star, z_star);

  // inject perturbation
  if (inject_perturbation) {
    inject_perturbation = false;
    if (Globals::my_rank == 0) {
      std::cout<<"Injecting perturbation..."<<std::endl;
    }
    std::srand(random_seed); // fix the random seed on each process...
    int m_min = perturbation_m_min, m_max = perturbation_m_max;
    Real * amplitudes = new Real [m_max-m_min+1];
    Real * phases = new Real [m_max-m_min+1];
    for (int m=m_min; m<m_max; ++m) {
      amplitudes[m-m_min] = perturbation_level;
      phases[m-m_min] = ((Real) std::rand()/RAND_MAX) * 2.*PI;
    }
    for (int k=ks; k<=ke; ++k) {
      Real phi = pcoord->x3v(k);
      Real factor = 1.;
      for (int m=m_min; m<m_max; ++m) {
        factor += std::cos(m*(phi+phases[m-m_min])) * amplitudes[m-m_min];
      }
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          // leave dfloor fixed            
          if (phydro->u(IDN,k,j,i)>dfloor) {
            Real factor_d = (phydro->u(IDN,k,j,i)-dfloor)/phydro->u(IDN,k,j,i)*(factor-1.) + 1.;
            // perturb density, fix velocity and temperature
            phydro->u(IDN,k,j,i) *= factor_d;
            phydro->u(IM1,k,j,i) *= factor_d;
            phydro->u(IM2,k,j,i) *= factor_d;
            phydro->u(IM3,k,j,i) *= factor_d;
            if (NON_BAROTROPIC_EOS)
              phydro->u(IEN,k,j,i) *= factor_d;
          }
        }
      }
    }
    delete[] amplitudes;
    delete[] phases;
    if (Globals::my_rank == 0) {
      std::cout<<"Finished injecting perturbation."<<std::endl;
    }
  }
}

void DiskOuterX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        Real r1=pmb->pcoord->x1v(iu+i), r2=pmb->pcoord->x1v(iu-i+1);
        prim(IDN,k,j,iu+i) = prim(IDN,k,j,iu-i+1);
        prim(IVX,k,j,iu+i) = -prim(IVX,k,j,iu-i+1)*SQR(r2)/SQR(r1);
        prim(IVY,k,j,iu+i) = -prim(IVY,k,j,iu-i+1);
        prim(IVZ,k,j,iu+i) = prim(IVZ,k,j,iu-i+1)*r2/r1;
        prim(IPR,k,j,iu+i) = prim(IPR,k,j,iu-i+1);
      }
    }
  }
}
void DiskInnerX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        Real r1=pmb->pcoord->x1v(il-i), r2=pmb->pcoord->x1v(il+i-1);
        prim(IDN,k,j,il-i) = prim(IDN,k,j,il+i-1);
        prim(IVX,k,j,il-i) = -prim(IVX,k,j,il+i-1)*SQR(r2)/SQR(r1);
        prim(IVY,k,j,il-i) = -prim(IVY,k,j,il+i-1);
        prim(IVZ,k,j,il-i) = prim(IVZ,k,j,il+i-1)*r2/r1;
        prim(IPR,k,j,il-i) = prim(IPR,k,j,il+i-1);
      }
    }
  }
}

void GetStellarMassAndLocation(MeshBlock * pmb, Real Mtot, Real & M_star, Real & x_star, Real & y_star, Real & z_star) {
  Real M [4] = {0.,0.,0.,0.};
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        Real r = pmb->pcoord->x1v(i);
        Real th = pmb->pcoord->x2v(j);
        Real phi = pmb->pcoord->x3v(k);
        Real x = r*std::sin(th)*std::cos(phi);
        Real y = r*std::sin(th)*std::sin(phi);
        Real z = r*std::cos(th);
        Real vol = pmb->pcoord->GetCellVolume(k,j,i);
        Real den = pmb->phydro->u(IDN,k,j,i);
        M[0] += den*vol;
        M[1] += den*vol*x;
        M[2] += den*vol*y;
        M[3] += den*vol*z;
      }
    }
  }
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &M, 4, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  M_star = Mtot - M[0];
  x_star = -M[1]/M_star;
  y_star = -M[2]/M_star;
  z_star = -M[3]/M_star;

  // correction for 2d
  if (pmb->block_size.nx3==1) {
    x_star = 0.;
    y_star = 0.;
  }

  /*if (Globals::my_rank==0) {
    std::cout<<"disk-to-total mass ratio = "<<M[0]/Mtot<<std::endl;
    std::cout<<"x, y, z of star = "<<x_star<<" "<<y_star<<" "<<z_star<<std::endl;
  }*/
  // save results in ruser_mesh_data
  pmb->pmy_mesh->ruser_mesh_data[0](0) = M_star;
  pmb->pmy_mesh->ruser_mesh_data[0](1) = x_star;
  pmb->pmy_mesh->ruser_mesh_data[0](2) = y_star;
  pmb->pmy_mesh->ruser_mesh_data[0](3) = z_star;
}
