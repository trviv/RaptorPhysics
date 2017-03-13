#ifndef RX_MATRIX
#define RX_MATRIX

#include "Matrix4.h"

/// Enumeration of affine matrix components
enum MAT_TRANS
{
  TRANS,
  INV_TRANS
};

/// Class representing Affine matrix for an object
CDEF class Matrix
{
protected:
  Matrix4 trans, itrans;
public:

  /// Default constructor ( unused )
  FORCE_INLINE  CU_DEV_HOST Matrix()
  {}

  /// Copy constructor
  FORCE_INLINE  CU_DEV_HOST Matrix(const Matrix& mat)
  {
    trans = mat.trans;
    itrans = mat.itrans;
  }

  /// Multiplication operator
  FORCE_INLINE  CU_DEV_HOST void  operator*=(const Matrix& mat)
  {
    trans *= mat.trans;
    itrans.preMul(mat.itrans);
  }

  /// Access operator
  FORCE_INLINE  CU_DEV_HOST const Matrix4&  operator[](
    const Counter& index)const
  {
    return *(&trans + index);
  }

  /// Set identity matrix
  FORCE_INLINE  CU_DEV_HOST void  setIdentity()
  {
    trans.setIdentity();
    itrans.setIdentity();
  }

  /// Apply rotation according to value
  FORCE_INLINE  CU_DEV_HOST void  rotate(const Real3& val)
  {
    real cost, sint;
    Real3 rv = val * M_PI_180;

    Matrix tm, ta, &mat = *this;

    cost = mCos(rv[X]);
    sint = mSin(rv[X]);

    tm.setIdentity();
    tm.trans.r1[Y] = tm.itrans.r1[Y] = tm.trans.r2[Z] = tm.itrans.r2[Z]
      = cost;
    tm.trans.r2[Y] = tm.itrans.r1[Z] = sint;
    tm.trans.r1[Z] = tm.itrans.r2[Y] = -sint;
    ta.trans.setMul(mat.trans, tm.trans);
    ta.itrans.setMul(tm.itrans, mat.itrans);

    cost = mCos(rv[Y]);
    sint = mSin(rv[Y]);

    mat.setIdentity();
    mat.trans.r0[X] = mat.itrans.r0[X] = mat.trans.r2[Z] =
      mat.itrans.r2[Z] = cost;
    mat.trans.r0[Z] = mat.itrans.r2[X] = sint;
    mat.trans.r2[X] = mat.itrans.r0[Z] = -sint;
    tm.trans.setMul(ta.trans, mat.trans);
    tm.itrans.setMul(mat.itrans, ta.itrans);

    cost = mCos(rv[Z]);
    sint = mSin(rv[Z]);

    ta.setIdentity();
    ta.trans.r0[X] = ta.itrans.r0[X] = ta.trans.r1[Y] = ta.itrans.r1[Y]
      = cost;
    ta.trans.r1[X] = ta.itrans.r0[Y] = sint;
    ta.trans.r0[Y] = ta.itrans.r1[X] = -sint;
    mat.trans.setMul(tm.trans, ta.trans);
    mat.itrans.setMul(ta.itrans, tm.itrans);
  }

  /// Apply scale according to value
  FORCE_INLINE  CU_DEV_HOST void  scale(const Real3& val)
  {
    Matrix sm;
    sm.setIdentity();
    sm.trans.r0[X] = val[X];
    sm.trans.r1[Y] = val[Y];
    sm.trans.r2[Z] = val[Z];

    sm.itrans.r0[X] /= val[X];
    sm.itrans.r1[Y] /= val[Y];
    sm.itrans.r2[Z] /= val[Z];

    *this *= sm;
  }

  /// Apply translation according to value
  FORCE_INLINE  CU_DEV_HOST void  translate(const Real3& val)
  {
    Matrix t;
    t.trans.set(Matrix3::getIdentity(), val);
    t.itrans.set(Matrix3::getIdentity(), val.additiveInv());

    *this *= t;
  }
};

#endif