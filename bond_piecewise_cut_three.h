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

#ifdef BOND_CLASS

BondStyle(piecewise/cut/three,BondPiecewiseCutThree)

#else

#ifndef LMP_BOND_PIECEWISE_CUT_THREE_H
#define LMP_BOND_PIECEWISE_CUT_THREE_H

#include "bond.h"

namespace LAMMPS_NS {

class BondPiecewiseCutThree : public Bond {
 public:
  BondPiecewiseCutThree(class LAMMPS *);
  virtual ~BondPiecewiseCutThree();
  virtual void compute(int, int);
  virtual void coeff(int, char **);
  double equilibrium_distance(int);
  void write_restart(FILE *);
  virtual void read_restart(FILE *);
  void write_data(FILE *);
  double single(int, double, int, int, double &);
  virtual void *extract(char *, int &);

 protected:
  double *k0,*r0,*k1,*r1, *k2, *r2, *r3;

  virtual void allocate();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Incorrect args for bond coefficients

Self-explanatory.  Check the input script or data file.

*/
