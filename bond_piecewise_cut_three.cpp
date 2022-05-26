/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */
/* ------------------------------------------------------------------------
 Piecewise/three bond style written by Mahdi Tavakol
 (mahditavakol90@gmail.com)
 bond style   piecewise/cut/three
 ------------------------------------------------------------------------ */

#include "bond_piecewise_cut_three.h"
#include <mpi.h>
#include <cmath>
#include <cstring>
#include "atom.h"
#include "neighbor.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"
#include "utils.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

BondPiecewiseCutThree::BondPiecewiseCutThree(LAMMPS *lmp) : Bond(lmp)
{
  reinitflag = 1;
}

/* ---------------------------------------------------------------------- */

BondPiecewiseCutThree::~BondPiecewiseCutThree()
{
  if (allocated && !copymode) {
    memory->destroy(setflag);
    memory->destroy(k0);
    memory->destroy(r0);
    memory->destroy(k1);
    memory->destroy(r1);
	memory->destroy(k2);
    memory->destroy(r2);
	memory->destroy(r3);
  }
}

/* ---------------------------------------------------------------------- */

void BondPiecewiseCutThree::compute(int eflag, int vflag)
{
  int i1,i2,n,type;
  double delx,dely,delz,ebond,fbond;
  double rsq,r,dr,rk,dr0,dr1,dr2;
  double dr0p, dr1p, dr2p, ebondp;

  ebond = 0.0;
  ev_init(eflag,vflag);

  double **x = atom->x;
  double **f = atom->f;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;

  for (n = 0; n < nbondlist; n++) {
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    type = bondlist[n][2];

    delx = x[i1][0] - x[i2][0];
    dely = x[i1][1] - x[i2][1];
    delz = x[i1][2] - x[i2][2];

    rsq = delx*delx + dely*dely + delz*delz;
    r = sqrt(rsq);
    dr = r - r0[type];
    rk = k0[type] * dr;

    // force & energy

	if (r > r3[type]) continue;

    if (r > r2[type]) {
      dr0 = r1[type] - r0[type];
      dr1 = r2[type] - r1[type];
	  dr2 = r - r2[type];
      rk = k0[type] * dr0 + k1[type] * dr1 + k2[type] * dr2;
      fbond = -2.0*rk/r;
    } else if (r > r1[type]) {
	  dr0 = r1[type] - r0[type];
	  dr1 = r - r1[type];
	  rk = k0[type] * dr0 + k1[type] * dr1;
      fbond = -2.0*rk/r;
	} else if (r > 0) {
	  fbond = -2.0*rk / r;
	} else fbond = 0.0;


    if (eflag) {
	dr2p = r3[type] - r2[type];
	dr1p = r2[type] - r1[type];
	dr0p = r1[type] - r0[type];
	ebondp = k0[type]*dr0p*dr0p + 2*k0[type]*dr0p*dr1p + k1[type]*dr1p*dr1p + 2*k0[type]*dr0p*dr2p + 2*k1[type]*dr1p*dr2p + k2[type]*dr2p*dr2p;
	  if (r > r2[type]) {
	ebond  = k0[type]*dr0*dr0 + 2*k0[type]*dr0*dr1 + k1[type]*dr1*dr1 + 2*k0[type]*dr0*dr2 + 2*k1[type]*dr1*dr2 + k2[type]*dr2*dr2 - ebondp;
	  } else if (r > r1[type]) {
	ebond = k0[type]*dr0*dr0 + 2*k0[type]*dr0*dr1 + k1[type]*dr1*dr1 - ebondp;
      } else {
	ebond = rk*dr - ebondp;
      }
    }

    // apply force to each of 2 atoms

    if (newton_bond || i1 < nlocal) {
      f[i1][0] += delx*fbond;
      f[i1][1] += dely*fbond;
      f[i1][2] += delz*fbond;
    }

    if (newton_bond || i2 < nlocal) {
      f[i2][0] -= delx*fbond;
      f[i2][1] -= dely*fbond;
      f[i2][2] -= delz*fbond;
    }

    if (evflag) ev_tally(i1,i2,nlocal,newton_bond,ebond,fbond,delx,dely,delz);
  }
}

/* ---------------------------------------------------------------------- */

void BondPiecewiseCutThree::allocate()
{
  allocated = 1;
  int n = atom->nbondtypes;

  memory->create(k0,n+1,"bond:k0");
  memory->create(r0,n+1,"bond:r0");
  memory->create(k1,n+1,"bond:k1");
  memory->create(r1,n+1,"bond:r1");
  memory->create(k2,n+1,"bond:k2");
  memory->create(r2,n+1,"bond:r2");
  memory->create(r3,n+1,"bond:r3");
  memory->create(setflag,n+1,"bond:setflag");
  for (int i = 1; i <= n; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more types
------------------------------------------------------------------------- */

void BondPiecewiseCutThree::coeff(int narg, char **arg)
{
  if (narg != 8) error->all(FLERR,"Incorrect args for bond coefficients");
  if (!allocated) allocate();

  if ((arg[2]>arg[4])||(arg[4]>arg[6])||(arg[6]>arg[7])) error->all(FLERR, "Incorrect args for bond coefficients");
  if (!allocated) allocate();

  int ilo,ihi;
  force->bounds(FLERR,arg[0],atom->nbondtypes,ilo,ihi);

  double k0_one = force->numeric(FLERR,arg[1]);
  double r0_one = force->numeric(FLERR,arg[2]);
  double k1_one = force->numeric(FLERR,arg[3]);
  double r1_one = force->numeric(FLERR,arg[4]);
  double k2_one = force->numeric(FLERR,arg[5]);
  double r2_one = force->numeric(FLERR,arg[6]);
  double r3_one = force->numeric(FLERR,arg[7]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    k0[i] = k0_one;
    r0[i] = r0_one;
    k1[i] = k1_one;
    r1[i] = r1_one;
	k2[i] = k2_one;
	r2[i] = r2_one;
	r3[i] = r3_one;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR,"Incorrect args for bond coefficients");
}

/* ----------------------------------------------------------------------
   return an equilbrium bond length
------------------------------------------------------------------------- */

double BondPiecewiseCutThree::equilibrium_distance(int i)
{
  return r0[i];
}

/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file
------------------------------------------------------------------------- */

void BondPiecewiseCutThree::write_restart(FILE *fp)
{
  fwrite(&k0[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&r0[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&k1[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&r1[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&k2[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&r2[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&r3[1],sizeof(double),atom->nbondtypes,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them
------------------------------------------------------------------------- */

void BondPiecewiseCutThree::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
	utils::sfread(FLERR,&k0[1],sizeof(double),atom->nbondtypes,fp,NULL,error);
    utils::sfread(FLERR,&r0[1],sizeof(double),atom->nbondtypes,fp,NULL,error);
    utils::sfread(FLERR,&k1[1],sizeof(double),atom->nbondtypes,fp,NULL,error);
    utils::sfread(FLERR,&r1[1],sizeof(double),atom->nbondtypes,fp,NULL,error);
	utils::sfread(FLERR,&k2[1],sizeof(double),atom->nbondtypes,fp,NULL,error);
    utils::sfread(FLERR,&r2[1],sizeof(double),atom->nbondtypes,fp,NULL,error);
	utils::sfread(FLERR,&r3[1],sizeof(double),atom->nbondtypes,fp,NULL,error);
  }
  MPI_Bcast(&k0[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&r0[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&k1[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&r1[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&k2[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&r2[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&r3[1],atom->nbondtypes,MPI_DOUBLE,0,world);

  for (int i = 1; i <= atom->nbondtypes; i++) setflag[i] = 1;
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void BondPiecewiseCutThree::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nbondtypes; i++)
    fprintf(fp,"%d %g %g %g %g %g %g %g\n",i,k0[i],r0[i],k1[i],r1[i],k2[i],r2[i],r3[i]);
}

/* ---------------------------------------------------------------------- */

double BondPiecewiseCutThree::single(int type, double rsq, int i, int j,
                        double &fforce)
{
  double r = sqrt(rsq);
  double dr = r - r0[type];
  double rk = k0[type] * dr;
  double dr2p = r3[type] - r2[type];
  double dr1p = r2[type] - r1[type];
  double dr0p = r1[type] - r0[type];
  double ebondp = k0[type]*dr0p*dr0p + 2*k0[type]*dr0p*dr1p + k1[type]*dr1p*dr1p + 2*k0[type]*dr0p*dr2p + 2*k1[type]*dr1p*dr2p + k2[type]*dr2p*dr2p;

    if (r > r3[type]) {
		fforce = 0.0;
		double energy = 0.0;
	} else if (r > r2[type]) {
		double dr0 = r1[type] - r0[type];
		double dr1 = r2[type] - r1[type];
		double dr2 = r - r2[type];
		rk = k0[type]*dr0 + k1[type]*dr1 + k2[type]*dr2;
		fforce = -2.0*rk / r;
		double energy = k0[type]*dr0*dr0 + 2*k0[type]*dr0*dr1 + k1[type]*dr1*dr1 + 2*k0[type]*dr0*dr2 + 2*k1[type]*dr1*dr2 + k2[type]*dr2*dr2 - ebondp;
	} else if (r > r1[type]) {
      double dr0 = r1[type] - r0[type];
      double dr1 = r - r1[type];
      rk = k0[type] * dr0 + k1[type] * dr1;
      fforce = -2.0*rk/r;
      double energy = k0[type]*dr0*dr0 + 2*k0[type]*dr0*dr1 + k1[type]*dr1*dr1 - ebondp;
    } else if (r > 0.0) {
      fforce = -2.0*rk/r;
      double energy = rk*dr - ebondp;
    } else {
      fforce = 0.0;
      double energy = rk*dr - ebondp;
    }

  return energy;
}

/* ----------------------------------------------------------------------
    Return ptr to internal members upon request.
------------------------------------------------------------------------ */
void *BondPiecewiseCutThree::extract( char *str, int &dim )
{
  dim = 1;
  if (strcmp(str,"kappa0")==0) return (void*) k0;
  if (strcmp(str,"r0")==0)     return (void*) r0;
  if (strcmp(str,"kappa1")==0) return (void*) k1;
  if (strcmp(str,"r1")==0)     return (void*) r1;
  if (strcmp(str,"kappa2")==0) return (void*) k2;
  if (strcmp(str,"r2")==0)     return (void*) r2;
  if (strcmp(str,"r3")==0)     return (void*) r3;
  return NULL;
}
