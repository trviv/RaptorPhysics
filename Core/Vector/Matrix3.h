#ifndef RX_MATRIX3
#define RX_MATRIX3

#include "Real3.h"

/// Class to represent rotational and scaling componets of matrix
class Matrix3
{
protected:
  Real3 r0, r1, r2;

public:

  /// Default constructor
  Matrix3(){}

  /// Copy constructor
  Matrix3(const Matrix3& mat)
  {
    *this = mat;
  }

  /// Constructor using different vectors
  Matrix3(const Real3& r0, const Real3& r1,
    const Real3& r2)
  {
    set(r0, r1, r2);
  }

  /// Constructor usign different values
  Matrix3(
    const real& xx, const real& xy, const real& xz,
    const real& yx, const real& yy, const real& yz,
    const real& zx, const real& zy, const real& zz)
  {
    set(xx, xy, xz, yx, yy, yz, zx, zy, zz);
  }

  /// Assignment operator overloading
  Matrix3&  operator=(const Matrix3& m)
  {
    r0 = m.r0;  r1 = m.r1;  r2 = m.r2;  return *this;
  }

  /// Multiply operator overloading
  void  operator*=(const Matrix3& m)
  {
    const Real3 c0(m.getColumn(0));
    const Real3 c1(m.getColumn(1));
    const Real3 c2(m.getColumn(2));
    r0.set(r0.dot(c0), r0.dot(c1), r0.dot(c2));
    r1.set(r1.dot(c0), r1.dot(c1), r1.dot(c2));
    r2.set(r2.dot(c0), r2.dot(c1), r2.dot(c2));
  }

  /// Get column value at index
  const Real3 getColumn(const int& index)const
  {
    return  Real3(r0[index], r1[index], r2[index]);
  }

  /// Get row at index
  const Real3&  getRow(const int& index)const
  {
    return  *(&r0 + index);
  }

  /// Get identity matrix
  static const Matrix3 getIdentity()
  {
    return  Matrix3(Real3(1, 0, 0), Real3(0, 1, 0), Real3(0, 0, 1));
  }

  /// Transform direction using this matrix and store in variable
  void  transformDir(Real3& var,
    const Real3& dir)const
  {
    var.set(r0.dot(dir), r1.dot(dir), r2.dot(dir));
  }

  /// Transform direction using matrix
  void  transformDir(Real3& dir)const
  {
    dir.set(r0.dot(dir), r1.dot(dir), r2.dot(dir));
  }

  /// Get transformed direction using matrix
  Real3 transformDir(const Real3& dir)const
  {
    return  Real3(r0.dot(dir), r1.dot(dir), r2.dot(dir));
  }

  /// Set matrix by vectors
  void  set(
    const Real3& rw0, const Real3& rw1, const Real3& rw2)
  {
    r0 = rw0, r1 = rw1, r2 = rw2;
  }

  /// Set matrix by values
  void  set(
    const real& xx, const real& xy, const real& xz,
    const real& yx, const real& yy, const real& yz,
    const real& zx, const real& zy, const real& zz)
  {
    r0.set(xx, xy, xz), r1.set(yx, yy, yz), r2.set(zx, zy, zz);
  }

  /// Set identity matrix
  void  setIdentity()
  {
    r0.set(1, 0, 0), r1.set(0, 1, 0), r2.set(0, 0, 1);
  }
};

#endif
