#ifndef RX_MATRIX4
#define RX_MATRIX4

#include "Matrix3.h"

/// Class to represent all matrix components
/// ie translation, scaling, rotation
class Matrix4 : public Matrix3
{
protected:
  Real3 pos;
  friend class Matrix;

public:
  /// Default constructor ( unused )
  Matrix4()
  {}

  /// Construct using a Matrix 3 and position
  Matrix4(const Matrix3& mat,
    const Real3& position)
    :Matrix3(mat), pos(position)
  {}

  /// Copy constructor
  Matrix4(const Matrix4& mat)
  {
    *this = mat;
  }

  /// Assignment operator
  void  operator=(const Matrix4& mat)
  {
    Matrix3::operator=(mat);  pos = mat.pos;
  }

  /// Multiplication operator
  void  operator*=(const Matrix4& mat)
  {
    pos.set(r0.dot(mat.pos) + pos[X],
      r1.dot(mat.pos) + pos[Y],
      r2.dot(mat.pos) + pos[Z]);
    Matrix3& mat3 = *this;
    mat3 *= mat;
  }

  /// Transform position using matrix and store in variable
  void  transformPos(Real3& var,
    const Real3& pos)const
  {
    var.set(r0.dot(pos) + this->pos[X],
      r1.dot(pos) + this->pos[Y],
      r2.dot(pos) + this->pos[Z]);
  }

  /// Transform position using matrix
  void  transformPos(Real3& pos)const
  {
    pos.set(r0.dot(pos) + this->pos[X],
      r1.dot(pos) + this->pos[Y],
      r2.dot(pos) + this->pos[Z]);
  }

  /// Get position component
  const Real3&  getPos()const
  {
    return pos;
  }

  /// Set matrix using values
  void  set(
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
  void  set(const Matrix3& mat,
    const Real3& pos)
  {
    Matrix3::operator=(mat);  this->pos = pos;
  }

  /// Set matrix to identity matrix
  void  setIdentity()
  {
    Matrix3::setIdentity(); pos.set(0, 0, 0);
  }

  /// Set matrix to multiplication of given matrix ( = a x b )
  void  setMul(const Matrix4& a,
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
  void  preMul(const Matrix4& mat)
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

  // multiply two matrix in the form of array
  static void multiply(real result[16], const real matA[16], const real matB[16])
  {
    uint i, j, k;
    for (i = 0; i < 4; i++)
    {
      for (j = 0; j < 4; j++)
      {
        result[i * 4 + j] = 0.f;
        for (k = 0; k < 4; k++)
        {
          result[i * 4 + j] += matA[i * 4 + k] * matB[k * 4 + j];
        }
      }
    }
  }

  // invert a matrix
  static bool invert(real invMatrix[16], const real matrix[16])
  {
    real inv[16], det;

    inv[0] = matrix[5]  * matrix[10] * matrix[15] -
             matrix[5]  * matrix[11] * matrix[14] -
             matrix[9]  * matrix[6]  * matrix[15] +
             matrix[9]  * matrix[7]  * matrix[14] +
             matrix[13] * matrix[6]  * matrix[11] -
             matrix[13] * matrix[7]  * matrix[10];

    inv[4] = -matrix[4]  * matrix[10] * matrix[15] +
              matrix[4]  * matrix[11] * matrix[14] +
              matrix[8]  * matrix[6]  * matrix[15] -
              matrix[8]  * matrix[7]  * matrix[14] -
              matrix[12] * matrix[6]  * matrix[11] +
              matrix[12] * matrix[7]  * matrix[10];

    inv[8] = matrix[4]  * matrix[9] * matrix[15] -
             matrix[4]  * matrix[11] * matrix[13] -
             matrix[8]  * matrix[5] * matrix[15] +
             matrix[8]  * matrix[7] * matrix[13] +
             matrix[12] * matrix[5] * matrix[11] -
             matrix[12] * matrix[7] * matrix[9];

    inv[12] = -matrix[4]  * matrix[9] * matrix[14] +
               matrix[4]  * matrix[10] * matrix[13] +
               matrix[8]  * matrix[5] * matrix[14] -
               matrix[8]  * matrix[6] * matrix[13] -
               matrix[12] * matrix[5] * matrix[10] +
               matrix[12] * matrix[6] * matrix[9];

    inv[1] = -matrix[1]  * matrix[10] * matrix[15] +
              matrix[1]  * matrix[11] * matrix[14] +
              matrix[9]  * matrix[2] * matrix[15] -
              matrix[9]  * matrix[3] * matrix[14] -
              matrix[13] * matrix[2] * matrix[11] +
              matrix[13] * matrix[3] * matrix[10];

    inv[5] = matrix[0]  * matrix[10] * matrix[15] -
             matrix[0]  * matrix[11] * matrix[14] -
             matrix[8]  * matrix[2] * matrix[15] +
             matrix[8]  * matrix[3] * matrix[14] +
             matrix[12] * matrix[2] * matrix[11] -
             matrix[12] * matrix[3] * matrix[10];

    inv[9] = -matrix[0]  * matrix[9] * matrix[15] +
              matrix[0]  * matrix[11] * matrix[13] +
              matrix[8]  * matrix[1] * matrix[15] -
              matrix[8]  * matrix[3] * matrix[13] -
              matrix[12] * matrix[1] * matrix[11] +
              matrix[12] * matrix[3] * matrix[9];

    inv[13] = matrix[0]  * matrix[9] * matrix[14] -
              matrix[0]  * matrix[10] * matrix[13] -
              matrix[8]  * matrix[1] * matrix[14] +
              matrix[8]  * matrix[2] * matrix[13] +
              matrix[12] * matrix[1] * matrix[10] -
              matrix[12] * matrix[2] * matrix[9];

    inv[2] = matrix[1]  * matrix[6] * matrix[15] -
             matrix[1]  * matrix[7] * matrix[14] -
             matrix[5]  * matrix[2] * matrix[15] +
             matrix[5]  * matrix[3] * matrix[14] +
             matrix[13] * matrix[2] * matrix[7] -
             matrix[13] * matrix[3] * matrix[6];

    inv[6] = -matrix[0]  * matrix[6] * matrix[15] +
              matrix[0]  * matrix[7] * matrix[14] +
              matrix[4]  * matrix[2] * matrix[15] -
              matrix[4]  * matrix[3] * matrix[14] -
              matrix[12] * matrix[2] * matrix[7] +
              matrix[12] * matrix[3] * matrix[6];

    inv[10] = matrix[0]  * matrix[5] * matrix[15] -
              matrix[0]  * matrix[7] * matrix[13] -
              matrix[4]  * matrix[1] * matrix[15] +
              matrix[4]  * matrix[3] * matrix[13] +
              matrix[12] * matrix[1] * matrix[7] -
              matrix[12] * matrix[3] * matrix[5];

    inv[14] = -matrix[0]  * matrix[5] * matrix[14] +
               matrix[0]  * matrix[6] * matrix[13] +
               matrix[4]  * matrix[1] * matrix[14] -
               matrix[4]  * matrix[2] * matrix[13] -
               matrix[12] * matrix[1] * matrix[6] +
               matrix[12] * matrix[2] * matrix[5];

    inv[3] = -matrix[1] * matrix[6] * matrix[11] +
              matrix[1] * matrix[7] * matrix[10] +
              matrix[5] * matrix[2] * matrix[11] -
              matrix[5] * matrix[3] * matrix[10] -
              matrix[9] * matrix[2] * matrix[7] +
              matrix[9] * matrix[3] * matrix[6];

    inv[7] = matrix[0] * matrix[6] * matrix[11] -
             matrix[0] * matrix[7] * matrix[10] -
             matrix[4] * matrix[2] * matrix[11] +
             matrix[4] * matrix[3] * matrix[10] +
             matrix[8] * matrix[2] * matrix[7] -
             matrix[8] * matrix[3] * matrix[6];

    inv[11] = -matrix[0] * matrix[5] * matrix[11] +
               matrix[0] * matrix[7] * matrix[9] +
               matrix[4] * matrix[1] * matrix[11] -
               matrix[4] * matrix[3] * matrix[9] -
               matrix[8] * matrix[1] * matrix[7] +
               matrix[8] * matrix[3] * matrix[5];

    inv[15] = matrix[0] * matrix[5] * matrix[10] -
              matrix[0] * matrix[6] * matrix[9] -
              matrix[4] * matrix[1] * matrix[10] +
              matrix[4] * matrix[2] * matrix[9] +
              matrix[8] * matrix[1] * matrix[6] -
              matrix[8] * matrix[2] * matrix[5];

    det = matrix[0] * inv[0] + matrix[1] * inv[4] + matrix[2] * inv[8] + matrix[3] * inv[12];

    if (det == 0)
    {
      return false;
    }

    det = 1.0 / det;

    for (int i = 0; i < 16; i++)
    {
      invMatrix[i] = inv[i] * det;
    }

    invMatrix[3] = inv[12];
    invMatrix[7] = inv[13];
    invMatrix[11] = inv[14];

    invMatrix[12] = inv[3] * det;
    invMatrix[13] = inv[7] * det;
    invMatrix[14] = inv[11] * det;

    return true;
  }

  // transform a 4 element vector and return a Real3 value
  static Real3 transformVec(const real matrix[16], const real vector[4])
  {
    real ret[4];
    for (int i = 0; i < 4; i++)
    {
      ret[i] = 0;
      for (int j = 0; j < 4; j++)
      {
        ret[i] += matrix[i * 4 + j] * vector[j];
      }
    }
    return Real3(ret[0], ret[1], ret[2]);
  }

  // transform a 4 element vector and modify a vector
  static void transformVec(real result[4], const real matrix[16], const real vector[4])
  {
    real ret[4];
    for (int i = 0; i < 4; i++)
    {
      ret[i] = 0;
      for (int j = 0; j < 4; j++)
      {
        ret[i] += matrix[i * 4 + j] * vector[j];
      }
    }
    for (int i = 0; i < 4; i++)
    {
      result[i] = ret[i];
    }
  }
};

#endif
