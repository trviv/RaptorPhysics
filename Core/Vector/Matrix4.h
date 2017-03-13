#ifndef RX_MATRIX4
#define RX_MATRIX4

#include "Matrix3.h"

/// Class to represent all matrix components
/// ie translation, scaling, rotation
CDEF class Matrix4 : public Matrix3
{
protected:
  Real3 pos;
  friend class Matrix;

public:
  /// Default constructor ( unused )
  FORCE_INLINE  CU_DEV_HOST Matrix4()
  {}

  /// Construct using a Matrix 3 and position
  FORCE_INLINE  CU_DEV_HOST Matrix4(const Matrix3& mat,
    const Real3& position)
    :Matrix3(mat), pos(position)
  {}

  /// Copy constructor
  FORCE_INLINE  CU_DEV_HOST Matrix4(const Matrix4& mat)
  {
    *this = mat;
  }

  /// Assignment operator
  FORCE_INLINE  CU_DEV_HOST void  operator=(const Matrix4& mat)
  {
    Matrix3::operator=(mat);  pos = mat.pos;
  }

  /// Multiplication operator
  FORCE_INLINE  CU_DEV_HOST void  operator*=(const Matrix4& mat)
  {
    pos.set(r0.dot(mat.pos) + pos[X],
      r1.dot(mat.pos) + pos[Y],
      r2.dot(mat.pos) + pos[Z]);
    Matrix3& mat3 = *this;
    mat3 *= mat;
  }

  /// Transform position using matrix and store in variable
  FORCE_INLINE  CU_DEV_HOST void  transformPos(Real3& var,
    const Real3& pos)const
  {
    var.set(r0.dot(pos) + this->pos[X],
      r1.dot(pos) + this->pos[Y],
      r2.dot(pos) + this->pos[Z]);
  }

  /// Transform position using matrix
  FORCE_INLINE  CU_DEV_HOST void  transformPos(Real3& pos)const
  {
    pos.set(r0.dot(pos) + this->pos[X],
      r1.dot(pos) + this->pos[Y],
      r2.dot(pos) + this->pos[Z]);
  }

  /// Get position component
  FORCE_INLINE  CU_DEV_HOST const Real3&  getPos()const
  {
    return pos;
  }

  /// Set matrix using values
  FORCE_INLINE  CU_DEV_HOST void  set(
    const real& xx, const real& xy, const real& xz, const real& xa,
    const real& yx, const real& yy, const real& yz, const real& ya,
    const real& zx, const real& zy, const real& zz, const real& za)
  {
    r0.set(xx, xy, xz);
    r1.set(yx, yy, yz);
    r2.set(zx, zy, zz);
    pos.set(xa, ya, za);
  }

  /// Set using matrix and position
  FORCE_INLINE  CU_DEV_HOST void  set(const Matrix3& mat,
    const Real3& pos)
  {
    Matrix3::operator=(mat);  this->pos = pos;
  }

  /// Set matrix to identity matrix
  FORCE_INLINE  CU_DEV_HOST void  setIdentity()
  {
    Matrix3::setIdentity(); pos.set(0, 0, 0);
  }

  /// Set matrix to multiplication of given matrix ( = a x b )
  FORCE_INLINE  CU_DEV_HOST void  setMul(const Matrix4& a,
    const Matrix4& b)
  {
    Real3 c0(b.getColumn(0)),
      c1(b.getColumn(1)),
      c2(b.getColumn(2));
    const Real3& a0 = a.r0, &a1 = a.r1, &a2 = a.r2;
    pos.set(a0.dot(b.pos) + a.pos[X],
      a1.dot(b.pos) + a.pos[Y],
      a2.dot(b.pos) + a.pos[Z]);
    r0.set(a0.dot(c0), a0.dot(c1), a0.dot(c2));
    r1.set(a1.dot(c0), a1.dot(c1), a1.dot(c2));
    r2.set(a2.dot(c0), a2.dot(c1), a2.dot(c2));
  }

  /// Multiply matrix in preceding order ( = mat x this )
  FORCE_INLINE  CU_DEV_HOST void  preMul(const Matrix4& mat)
  {
    Real3 c0(getColumn(0)),
      c1(getColumn(1)),
      c2(getColumn(2));
    const Real3& m0 = mat.r0, &m1 = mat.r1, &m2 = mat.r2;
    pos.set(m0.dot(pos) + mat.pos[X],
      m1.dot(pos) + mat.pos[Y],
      m2.dot(pos) + mat.pos[Z]);
    r0.set(m0.dot(c0), m0.dot(c1), m0.dot(c2));
    r1.set(m1.dot(c0), m1.dot(c1), m1.dot(c2));
    r2.set(m2.dot(c0), m2.dot(c1), m2.dot(c2));
  }
};

#endif