#include "Cloth.h"
#include "PhysicsSystem.h"

Cloth::Cloth()
{
  solver = SOLVER_CLOTH;
}

void Cloth::initXY(const real dimensions[], const real particleRadius, const real mass)
{
  uint subdivision[3] = { 1, 1, 1 };
  for (int i = 0; i < 2; i++) subdivision[i] = uint(dimensions[i] / particleRadius);
  (*entityParticleSharedData.host())[0].sharedRadius = particleRadius;
  initXY(dimensions, subdivision, mass);
}

void Cloth::initXY(const real dimensions[], const uint subdivision[], const real mass)
{
  vector<uint32_t> connectionElements;

  Real3 del_x(0);
  if (subdivision[0] > 1)
  {
    del_x[0] = dimensions[0] / (subdivision[0] - 1);
  }
  Real3 del_y(0);
  if (subdivision[1] > 1)
  {
    del_y[1] = -dimensions[1] / (subdivision[1] - 1);
  }
  Real3 top_left(-dimensions[0] / 2, dimensions[1] / 2, 0);

  const real x_len = del_x.length();
  const real y_len = del_y.length();
  const real diag_len = mSqrt(mSqr(x_len) + mSqr(y_len));
  vector<Real3> pointPosition;

  const real perParticleInvMass = real(mass) / real(subdivision[0] * subdivision[1]);

  for (uint y = 0; y < subdivision[1]; y++)
  {
    uint index = y*subdivision[0];
    for (uint x = 0; x < subdivision[0]; x++)
    {
      addConnection(index, index, y ? perParticleInvMass : 0);
      ParticleAuxData auxData;
      auxData.invMass = rawConstrainCoefficients[index][0];
      auxData.radius = 0;
      particleAuxData.host()->push_back(auxData);
      index++;
    }
  }

  for (uint y = 0; y < subdivision[1]; y++)
  {
    Real3 pos = top_left + del_y*(float)y;
    uint index = y*subdivision[0];
    for (uint x = 0; x < subdivision[0]; x++)
    {
      Real3 newPosition = pos + Real3(0, 0, (((subdivision[1] - y) == 1 && (subdivision[0] - x) == 1) ? .5f : 0));
      setConstant(index, newPosition);
      pointPosition.push_back(newPosition);

      // add twice because constrain is solved only once
      if (x + 1 < subdivision[0])
      {
        addConnection(index, index + 1, x_len);
        addConnection(index + 1, index, x_len);
        connectionElements.push_back(index);
        connectionElements.push_back(index + 1);
      }
      if (y + 1 < subdivision[1])
      {
        addConnection(index, index + subdivision[0], y_len);
        addConnection(index + subdivision[0], index, y_len);
        connectionElements.push_back(index);
        connectionElements.push_back(index + subdivision[0]);
      }
      if (x + 1 < subdivision[0] && y + 1 < subdivision[1])
      {
        addConnection(index, index + subdivision[0] + 1, diag_len);
        addConnection(index + subdivision[0] + 1, index, diag_len);
        connectionElements.push_back(index);
        connectionElements.push_back(index + subdivision[0] + 1);
      }
      if (x >= 1 && y + 1 < subdivision[1])
      {
        addConnection(index, index + subdivision[0] - 1, diag_len);
        addConnection(index + subdivision[0] - 1, index, diag_len);
        connectionElements.push_back(index);
        connectionElements.push_back(index + subdivision[0] - 1);
      }

      index++;
      pos += del_x;
    }
  }

#ifdef ENABLE_RENDERING
  displayVertex.gen();
  displayVertex.copyData(&pointPosition[0][0], subdivision[0] * subdivision[1], 0, sizeof(Real3));

  displayElements.gen();
  displayElements.copyData((GLuint*)&connectionElements[0], connectionElements.size());

  displayShader.init("display_vert.glsl", "display_frag.glsl");
#endif
}

#ifdef ENABLE_RENDERING

void Cloth::render(ParticleStruct* particles)
{
  GLfloat model_mat[16], proj_mat[16];
  glGetFloatv(GL_PROJECTION_MATRIX, proj_mat);
  glGetFloatv(GL_MODELVIEW_MATRIX, model_mat);

  glDisable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);
  glDisable(GL_LIGHTING);
  glDisable(GL_POINT_SMOOTH);
  glDisable(GL_CULL_FACE);

  glPushMatrix();
  displayShader.bind();
  displayShader.set("modelViewMatrix", model_mat);
  displayShader.set("projectionMatrix", proj_mat);

  uint instances = 1;
  for (uint i = 0; i < instances; i++)
  {
    //displayVertex.bind();
    GL_CHECK(glEnableVertexAttribArray(0));
    GL_CHECK(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(ParticleStruct), particles));
    displayElements.bind();
    GL_CHECK(glDrawElementsInstanced(GL_LINES, displayElements.count(), GL_UNSIGNED_INT, NULL, 1));
    displayElements.unbind();
    GL_CHECK(glDisableVertexAttribArray(0));
    //displayVertex.unbind();
  }
  displayShader.unbind();
  glPopMatrix();
}

#endif