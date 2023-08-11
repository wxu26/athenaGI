//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file RemapColumns.cpp
//! \brief remap the domain to allow operations on global columns along each direction

#include <sstream>    // sstream
#include <chrono> // sleep
#include <thread> // sleep

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "../coordinates/coordinates.hpp"
#include "../globals.hpp"
#include "../bvals/cc/bvals_cc.hpp"

#ifdef MPI_PARALLEL
#include "./plimpton/remap_3d.h"
#include <mpi.h>
#endif // MPI_PARALLEL

#include "remap_columns.hpp"

//----------------------------------------------------------------------------------------
//! \fn void RemapColumns::Initialize(MeshBlock* my_pmb)
//! \brief initialize RemapColumns

// we do not use a constructor here because this needs to be called after the mesh and the
// MPI communication have been initialized

void RemapColumns::Initialize(MeshBlock* my_pmb) {
  #ifndef MPI_PARALLEL
  std::stringstream msg;
  msg << "### FATAL ERROR in RemapColumns" << std::endl
      << "RemapColumns requires MPI parallelization because"<< std::endl
      << "it uses Plimpton's MPI 3D remap functions"<< std::endl;
  ATHENA_ERROR(msg);
  #endif // MPI_PARALLEL
  if (initialized_) return;
  initialized_ = true;
  pmb = my_pmb;
  InitializeInd();
  InitializeGrid();
  InitializePlans();
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void RemapColumns::Initialize(MeshBlock * my_pmb,
//!          int npj1, int npk1, int npi2, int npk2, int npi3, int npj3);
//! \brief initialize RemapColumns while manually setting how the domain is decomposed
void RemapColumns::Initialize(MeshBlock * my_pmb,
     int npj1, int npk1, int npi2, int npk2, int npi3, int npj3) {
  int i=1,j=2,k=3;
  np[1][j] = npj1;
  np[1][k] = npk1;
  np[2][i] = npi2;
  np[2][k] = npk2;
  np[3][i] = npi3;
  np[3][j] = npj3;
  Initialize(my_pmb);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void bifactor(const int n1, const int n2, const int np, int * f1, int * f2)
//! \brief find most similar f1, f2 such that np = f1*f2, and f1,2 divides n1,2

void bifactor(const int n1, const int n2, const int np, int * f1, int * f2) {
  bool valid=false;
  int f1_current, f2_current, f1_best=0, f2_best=0;
  for (int f1_current=1; f1_current<=np; ++f1_current) {
    if (np%f1_current!=0) continue;
    f2_current = np/f1_current;
    if (n1%f1_current!=0 || n2%f2_current!=0) continue;
    valid = true;
    if (f1_current*f2_current > f1_best*f2_best) {
      f1_best = f1_current;
      f2_best = f2_current;
    }
  }
  if (!valid) {
    std::stringstream msg;
    msg << "### FATAL ERROR in RemapColumns" << std::endl
        << "Automatic partition failed; please set np's manually"<< std::endl;
    ATHENA_ERROR(msg);
  }
  *f1 = f1_best;
  *f2 = f2_best;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void RemapColumns::InitializeInd()
//! \brief initialize indices of each representation

void RemapColumns::InitializeInd(){
  // sanity check: one meshblock per rank
  if (pmb->pmy_mesh->nblocal!=1) {
    std::stringstream msg;
    msg << "### FATAL ERROR in RemapColumns" << std::endl
        << "Only one meshblock per process allowed"<< std::endl;
    ATHENA_ERROR(msg);
  }

  // mesh
  N[0] = pmb->pmy_mesh->mesh_size.nx1;
  N[1] = pmb->pmy_mesh->mesh_size.nx2;
  N[2] = pmb->pmy_mesh->mesh_size.nx3;
  
  // meshblock
  n[0][0] = pmb->block_size.nx1;
  n[0][1] = pmb->block_size.nx2;
  n[0][2] = pmb->block_size.nx3;
  for (int i=0;i<3;i++)
    np[0][i] = N[i]/n[0][i];
  cnt = n[0][0]*n[0][1]*n[0][2];
  
  // set other n and np
  for (int i=0;i<3;i++) {
    // the direction containing the whole axis
    np[i+1][i] = 1;
    int j,k;
    switch (i) {
      case 0: j=1; k=2; break;
      case 1: j=2; k=0; break;
      case 2: j=0; k=1; break;
    }
    if (np[i+1][j]==0 || np[i+1][k]==0) {
      int fj,fk;
      bifactor(n[0][j], n[0][k], np[0][i], &fj, &fk);
      np[i+1][j] = np[0][j] * fj;
      np[i+1][k] = np[0][k] * fk;
    }
    for (int l=0;l<3;l++)
      n[i+1][l] = N[l]/np[i+1][l];
  }
  
  // np sanity check
  bool np_correct=true;
  for (int i=0;i<4;i++) {
    // total number of processes is correct
    if (np[i][0]*np[i][1]*np[i][2]!=Globals::nranks) np_correct=false;
    // domain can be decomposed
    for (int j=0;j<3;j++) {
      if (N[j]%np[i][j]!=0) np_correct=false;
    }
  }
  if (!np_correct) {
    std::stringstream msg;
    msg << "### FATAL ERROR in RemapColumns" << std::endl
        << "Number of processes per dimension has to divide number of cells"<< std::endl
        << "current np is" << std::endl
        << np[0][0] << " " << np[0][1] << " " << np[0][2] << std::endl
        << np[1][0] << " " << np[1][1] << " " << np[1][2] << std::endl
        << np[2][0] << " " << np[2][1] << " " << np[2][2] << std::endl
        << np[3][0] << " " << np[3][1] << " " << np[3][2] << std::endl;
    ATHENA_ERROR(msg);
  }
  
  // set is, ie
  // TODO: optimize for performance here...
  int iloc = pmb->loc.lx1 + pmb->loc.lx2*np[0][0] + pmb->loc.lx3*np[0][0]*np[0][1];
  for (int i=0;i<4;i++) {
    int lx[3];
    lx[2] = iloc/(np[i][0]*np[i][1]);
    lx[1] = (iloc%(np[i][0]*np[i][1]))/np[i][0];
    lx[0] = iloc%np[i][0];
    for (int j=0;j<3;j++){
      is[i][j] = lx[j]*n[i][j];
      ie[i][j] = is[i][j] + n[i][j]-1;
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void RemapColumns::InitializeGrid()
//! \brief save a copy of the global mesh

void RemapColumns::InitializeGrid(){
  for (int i=0;i<3;i++){
    xf[i].NewAthenaArray(N[i]+1);
    x[i].NewAthenaArray(N[i]);
  }
  int Ni=N[0], Nj=N[1], Nk=N[2];
  for (int i=0; i<Ni+1; ++i) {
    Real rx = ComputeMeshGeneratorX(i,Ni,
      pmb->pmy_mesh->use_uniform_meshgen_fn_[X1DIR]);
    xf[0](i) = pmb->pmy_mesh->MeshGenerator_[X1DIR](rx, pmb->pmy_mesh->mesh_size);
  }
  for (int j=0; j<Nj+1; ++j) {
    Real rx = ComputeMeshGeneratorX(j,Nj,
      pmb->pmy_mesh->use_uniform_meshgen_fn_[X2DIR]);
    xf[1](j) = pmb->pmy_mesh->MeshGenerator_[X2DIR](rx, pmb->pmy_mesh->mesh_size);
  }
  for (int k=0; k<Nk+1; ++k) {
    Real rx = ComputeMeshGeneratorX(k,Nk,
      pmb->pmy_mesh->use_uniform_meshgen_fn_[X3DIR]);
    xf[2](k) = pmb->pmy_mesh->MeshGenerator_[X3DIR](rx, pmb->pmy_mesh->mesh_size);
  }
  if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
    for (int i=0; i<Ni; ++i) x[0](i) = .5*(xf[0](i)+xf[0](i+1));
    for (int j=0; j<Nj; ++j) x[1](j) = .5*(xf[1](j)+xf[1](j+1));
    for (int k=0; k<Nk; ++k) x[2](k) = .5*(xf[2](k)+xf[2](k+1));
  } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int i=0; i<Ni; ++i)
      x[0](i) = (TWO_3RD)*(std::pow(xf[0](i+1), 3) - std::pow(xf[0](i), 3)) /
              (std::pow(xf[0](i+1), 2) - std::pow(xf[0](i), 2));
    for (int j=0; j<Nj; ++j) x[1](j) = .5*(xf[1](j)+xf[1](j+1));
    for (int k=0; k<Nk; ++k) x[2](k) = .5*(xf[2](k)+xf[2](k+1));
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int i=0; i<Ni; ++i)
      x[0](i) = 0.75*(std::pow(xf[0](i+1), 4) - std::pow(xf[0](i), 4)) /
              (std::pow(xf[0](i+1), 3) - std::pow(xf[0](i), 3));
    for (int j=0; j<Nj; ++j)
      x[1](j) = ((std::sin(xf[1](j+1)) - xf[1](j+1)*std::cos(xf[1](j+1))) -
               (std::sin(xf[1](j  )) - xf[1](j  )*std::cos(xf[1](j  ))))/
               (std::cos(xf[1](j  )) - std::cos(xf[1](j+1)));
    for (int k=0; k<Nk; ++k) x[2](k) = .5*(xf[2](k)+xf[2](k+1));
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in RemapColumns" << std::endl
        << "Only cartesian, cylindrical, or spherical-polar coordinates are supported"
        << std::endl;
    ATHENA_ERROR(msg);
  }
  // sanity check
  bool agree = true;
  for (int i=0; i<n[0][0]; ++i) {
    if (pmb->pcoord->x1v(i+pmb->is) != x[0](i+is[0][0])) agree=false;
  }
  for (int j=0; j<n[0][1]; ++j) {
    if (pmb->pcoord->x2v(j+pmb->js) != x[1](j+is[0][1])) agree=false;
  }
  for (int k=0; k<n[0][2]; ++k) {
    if (pmb->pcoord->x3v(k+pmb->ks) != x[2](k+is[0][2])) agree=false;
  }
  if (!agree) {
    std::stringstream msg;
    msg << "### FATAL ERROR in RemapColumns" << std::endl
        << "grid is incorrect"
        << std::endl;
    ATHENA_ERROR(msg);
  }
  return;
}

#ifdef MPI_PARALLEL

//----------------------------------------------------------------------------------------
//! \fn void RemapColumns::InitializePlans()
//! \brief initialize remap_3d plans

void RemapColumns::InitializePlans(){
  // initialize buffer
  buf_ = new Real[cnt];
  // remap communicator
  MPI_Comm_dup(MPI_COMM_WORLD, &MPI_COMM_DECOMP);
  int precision = sizeof(Real)/4;
  int memory=0; // provide scratch at runtime; this saves memory consumption

  for (int i=0;i<4;i++) {
    for (int j=0;j<4;j++) {
      /*if (i==j) {
        plan_[i][j] = NULL;
        continue;
      }*/
      int il=i, jl=j;
      if (il==0) il+=1;
      if (jl==0) jl+=1;
      int f, m, s; // index for fast, mid, slow axis
      switch (il){
        case 1: f=0; m=1; s=2; break;
        case 2: f=1; m=2; s=0; break;
        case 3: f=2; m=0; s=1; break;
      }
      int permute = (jl-il+3)%3;
      int nqty = 1; // # of datums per element 
      plan_[i][j] = remap_3d_create_plan(
        MPI_COMM_DECOMP,
        is[i][f], ie[i][f], is[i][m], ie[i][m], is[i][s], ie[i][s],
        is[j][f], ie[j][f], is[j][m], ie[j][m], is[j][s], ie[j][s],
        nqty,permute,memory,precision);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void RemapColumns::Remap(const int rep_in, const int rep_out, Real *in, Real *out)
//! \brief perform a remap

void RemapColumns::Remap(const int i_in, const int i_out, Real * in, Real * out){
  if (plan_[i_in][i_out]==NULL)
    // std::swap(in, out); // swap can't be used here
    memcpy(out, in, cnt*sizeof(Real));
  else
    remap_3d(in, out, buf_, plan_[i_in][i_out]);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void RemapColumns::Print(const int layout, Real * data)
//! \brief print data in given layout

void RemapColumns::Print(const int layout, Real * data){
  int nfast, nmid, nslow;
  switch (layout) {
    case 0:
    case 1:
      nfast = n[layout][0];
      nmid  = n[layout][1];
      nslow = n[layout][2];
      break;
    case 2:
      nfast = n[layout][1];
      nmid  = n[layout][2];
      nslow = n[layout][0];
      break;
    case 3:
      nfast = n[layout][2];
      nmid  = n[layout][0];
      nslow = n[layout][1];
      break;
  }
  int output_now = 0;
  while (output_now<Globals::nranks) {
    if (Globals::my_rank==output_now) {
      int ind = 0;
      std::cout<<"rank "<<Globals::my_rank<<", index range="<<std::endl;
      std::cout<<is[layout][0]<<"-"<<ie[layout][0]<<" "
               <<is[layout][1]<<"-"<<ie[layout][1]<<" "
               <<is[layout][2]<<"-"<<ie[layout][2]<<std::endl;
      for (int k=0; k<nslow; k++) {
        std::cout<<"k="<<k<<std::endl;
        for (int j=0; j<nmid; j++) {
          for (int i=0; i<nfast; i++) {
            std::cout<<int(data[ind])<<" ";
            ind++;
          }
          std::cout<<std::endl;
        }
        std::cout<<std::endl;
      }
      output_now = Globals::my_rank+1;
    }
    MPI_Allreduce(MPI_IN_PLACE, &output_now, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

//----------------------------------------------------------------------------------------
//! \fn void RemapColumns::Test()
//! \brief test the remaps

void RemapColumns::Test(){
  bool agree=true;
  // loop through each pair
  for (int l1=0;l1<4;l1++) {
    for (int l2=0;l2<4;l2++) {
      for (int axis=0;axis<3;axis++) {
        if (!Test(l1,l2,axis,0)) agree=false;
      }
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &agree, 1, MPI_LOGICAL, MPI_LAND, MPI_COMM_WORLD);
  if (Globals::my_rank==0) {
    if (agree) {
      std::cout<<std::endl<<"### RemapColumns::Test ###"<<std::endl
               <<"test complete; all remaps are successful"<<std::endl;
    } else {
      std::cout<<std::endl<<"### RemapColumns::Test ###"<<std::endl
               <<"test complete; one or more remaps failed"<<std::endl;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn bool RemapColumns::Test(const int l1,const int l2,const int axis,const int print)
//! \brief test remap from l1->l2; cells are labeled with index along given axis

bool RemapColumns::Test(const int l1, const int l2, const int axis, const int print) {
  // print = 0: no output
  // print = 1: output whether the test is succesful
  // print = 2: output individual arrays
  // initialize buffer
  Real * data = new Real[cnt];
  Real * data_truth = new Real[cnt];
  // store data
  for (int k=0; k<n[l1][2]; k++) {
    for (int j=0; j<n[l1][1]; j++) {
      for (int i=0; i<n[l1][0]; i++) {
        int ind;
        if (l1==0 || l1==1) // ijk
          ind = i + n[l1][0]*j + n[l1][0]*n[l1][1]*k;
        else if (l1==2) // jki
          ind = j + n[l1][1]*k + n[l1][1]*n[l1][2]*i;
        else if (l1==3) // kij
          ind = k + n[l1][2]*i + n[l1][2]*n[l1][0]*j;
        if (axis==0)
          data[ind] = i + is[l1][0];
        else if (axis==1)
          data[ind] = j + is[l1][1];
        else if (axis==2)
          data[ind] = k + is[l1][2];
      }
    }
  }
  if (print>=2) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (Globals::my_rank==0)
      std::cout<<"original data"<<std::endl;
    Print(l1,data);
  }
  // remap
  std::swap(data, data_truth); // use data_truth as a scratch array for remap
  Remap(l1,l2,data_truth, data);
  // check data
  bool agree = true;
  for (int k=0; k<n[l2][2]; k++) {
    for (int j=0; j<n[l2][1]; j++) {
      for (int i=0; i<n[l2][0]; i++) {
        int ind;
        if (l2==0 || l2==1) // ijk
          ind = i + n[l2][0]*j + n[l2][0]*n[l2][1]*k;
        else if (l2==2) // jki
          ind = j + n[l2][1]*k + n[l2][1]*n[l2][2]*i;
        else if (l2==3) // kij
          ind = k + n[l2][2]*i + n[l2][2]*n[l2][0]*j;
        if (axis==0)
          data_truth[ind] = i + is[l2][0];
        else if (axis==1)
          data_truth[ind] = j + is[l2][1];
        else if (axis==2)
          data_truth[ind] = k + is[l2][2];
        if (data[ind]!=data_truth[ind]) agree=false;
      }
    }
  }
  if (print>=2) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (Globals::my_rank==0)
      std::cout<<"remapped data"<<std::endl;
    Print(l2,data);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (Globals::my_rank==0)
      std::cout<<"remapped data truth"<<std::endl;
    Print(l2,data_truth);
  }
  // output by rank
  if (print>=1) {
    int output_now = 0;
    while (output_now<Globals::nranks) {
      if (Globals::my_rank==output_now) {
        std::cout<<"test remap from "<<l1<<" to "<<l2<<" for axis "<<axis<<" on rank "
                 <<Globals::my_rank<<std::endl;
        if (agree)
          std::cout<<"success"<<std::endl;
        if (!agree)
          std::cout<<"failed"<<std::endl;
        output_now = Globals::my_rank+1;
      }
      MPI_Allreduce(MPI_IN_PLACE, &output_now, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  // this ensures that we have enough time to display the output from all ranks before
  // moving to the next iteration
  // clear buffer
  delete[] data;
  delete[] data_truth;
  return agree;
}

#endif // MPI_PARALLEL

//----------------------------------------------------------------------------------------
//! \fn void RemapColumns::InitializeBoundary(AthenaArray<Real> * data)
//! \brief initialize boundary communications for given data (optional)

void RemapColumns::InitializeBoundary(AthenaArray<Real> * data)
{
  if (!initialized_) {
    std::stringstream msg;
    msg << "### FATAL ERROR in RemapColumns" << std::endl
        << "Please run Initialize before running InitializeBoundary"<< std::endl;
    ATHENA_ERROR(msg);
  }
  initialized_boundary_ = true;
  AthenaArray<Real> empty_flux[3]
    = {AthenaArray<Real>(), AthenaArray<Real>(), AthenaArray<Real>()};
  bvar = new CellCenteredBoundaryVariable(pmb, data, NULL, empty_flux, false);
  bvar->bvar_index = pmb->pbval->bvars.size();
  pmb->pbval->bvars.push_back(bvar);
  bvar->SetupPersistentMPI();
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void RemapColumns::SetBoundaries()
//! \brief set block, pole, and non-user physical boundaties

void RemapColumns::SetBoundaries(){
  if (!initialized_boundary_) {
    std::stringstream msg;
    msg << "### FATAL ERROR in RemapColumns" << std::endl
        << "Please run InitializeBoundary before setting boundaries"<< std::endl;
    ATHENA_ERROR(msg);
  }
  // start
  bvar->StartReceiving(BoundaryCommSubset::all);
  // send
  bvar->SendBoundaryBuffers();
  // recv
  bool ret = false;
  while (!ret) ret = bvar->ReceiveBoundaryBuffers();
      // normally this requires a tasklist; but since we require one meshblock per
      // process for the remaps, this simple treatment works
  // set
  bvar->SetBoundaries();
  // physical bc
  ApplyPhysicalBoundaries();
  // clear
  bvar->ClearBoundary(BoundaryCommSubset::all);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void RemapColumns::ApplyPhysicalBoundaries(){
//! \brief apply non-user physical boundaties (reflect, outflow, polar-wedge)

void RemapColumns::ApplyPhysicalBoundaries(){
  // apply non-user physical boundaries (outflow, reflect, polar wedge)
  // user is treated as outflow here
  // block and polar have been handeled by boundary communications
  Real time=pmb->pmy_mesh->time, dt=pmb->pmy_mesh->dt;
  Real il=pmb->is, iu=pmb->ie, jl=pmb->js, ju=pmb->je, kl=pmb->ks, ku=pmb->ke;
  if (N[1]>1) {il-=NGHOST; iu+=NGHOST;}
  if (N[2]>1) {jl-=NGHOST; ju+=NGHOST;}
  // x[0]
  switch(pmb->pbval->block_bcs[BoundaryFace::inner_x1]) {
    case BoundaryFlag::reflect:
      bvar->ReflectInnerX1(time, dt, pmb->is, jl, ju, kl, ku, NGHOST); break;
    case BoundaryFlag::outflow:
      bvar->OutflowInnerX1(time, dt, pmb->is, jl, ju, kl, ku, NGHOST); break;
    case BoundaryFlag::user:
      bvar->OutflowInnerX1(time, dt, pmb->is, jl, ju, kl, ku, NGHOST); break;
    default:;
  }
  switch(pmb->pbval->block_bcs[BoundaryFace::outer_x1]) {
    case BoundaryFlag::reflect:
      bvar->ReflectOuterX1(time, dt, pmb->ie, jl, ju, kl, ku, NGHOST); break;
    case BoundaryFlag::outflow:
      bvar->OutflowOuterX1(time, dt, pmb->ie, jl, ju, kl, ku, NGHOST); break;
    case BoundaryFlag::user:
      bvar->OutflowOuterX1(time, dt, pmb->ie, jl, ju, kl, ku, NGHOST); break;
    default:;
  }
  // x[1]
  if (N[1]==1) return;
  switch(pmb->pbval->block_bcs[BoundaryFace::inner_x2]) {
    case BoundaryFlag::reflect:
      bvar->ReflectInnerX2(time, dt, il, iu, pmb->js, kl, ku, NGHOST); break;
    case BoundaryFlag::outflow:
      bvar->OutflowInnerX2(time, dt, il, iu, pmb->js, kl, ku, NGHOST); break;
    case BoundaryFlag::user:
      bvar->OutflowInnerX2(time, dt, il, iu, pmb->js, kl, ku, NGHOST); break;
    case BoundaryFlag::polar_wedge:
      bvar->PolarWedgeInnerX2(time, dt, il, iu, pmb->js, kl, ku, NGHOST); break;
    default:;
  }
  switch(pmb->pbval->block_bcs[BoundaryFace::outer_x2]) {
    case BoundaryFlag::reflect:
      bvar->ReflectOuterX2(time, dt, il, iu, pmb->je, kl, ku, NGHOST); break;
    case BoundaryFlag::outflow:
      bvar->OutflowOuterX2(time, dt, il, iu, pmb->je, kl, ku, NGHOST); break;
    case BoundaryFlag::user:
      bvar->OutflowOuterX2(time, dt, il, iu, pmb->je, kl, ku, NGHOST); break;
    case BoundaryFlag::polar_wedge:
      bvar->PolarWedgeOuterX2(time, dt, il, iu, pmb->je, kl, ku, NGHOST); break;
    default:;
  }
  // x[2]
  if (N[2]==1) return;
  switch(pmb->pbval->block_bcs[BoundaryFace::inner_x3]) {
    case BoundaryFlag::reflect:
      bvar->ReflectInnerX3(time, dt, il, iu, jl, ju, pmb->ks, NGHOST); break;
    case BoundaryFlag::outflow:
      bvar->OutflowInnerX3(time, dt, il, iu, jl, ju, pmb->ks, NGHOST); break;
    case BoundaryFlag::user:
      bvar->OutflowInnerX3(time, dt, il, iu, jl, ju, pmb->ks, NGHOST); break;
    default:;
  }
  switch(pmb->pbval->block_bcs[BoundaryFace::outer_x3]) {
    case BoundaryFlag::reflect:
      bvar->ReflectOuterX3(time, dt, il, iu, jl, ju, pmb->ke, NGHOST); break;
    case BoundaryFlag::outflow:
      bvar->OutflowOuterX3(time, dt, il, iu, jl, ju, pmb->ke, NGHOST); break;
    case BoundaryFlag::user:
      bvar->OutflowOuterX3(time, dt, il, iu, jl, ju, pmb->ke, NGHOST); break;
    default:;
  }
  return;
}