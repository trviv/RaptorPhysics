#include "Cloth.h"
//#include "PhysicsSystem.h"

void Cloth::init(const Matrix4& transform, const float dim[],
  const Counter subdivision[])
{
  Real3 del_x(0); del_x[0] = 2.*dim[0] / (subdivision[0] - 1);
  Real3 del_y(0); del_y[1] = -2.*dim[1] / (subdivision[1] - 1);
  Real3 top_left(-dim[0], dim[1], 0);

  float x_len = del_x.length();
  float y_len = del_y.length();
  float diag_len = mSqrt(mSqr(x_len) + mSqr(y_len));
  std::vector<Real3> point_pos;

  for (Counter y = 0; y < subdivision[1]; y++)
  {
    Counter index = y*subdivision[0];
    for (Counter x = 0; x < subdivision[0]; x++)
    {
      physics_system->distanceConstrain()->add(index, index, 0, 0, y ? 1 : 0);
      index++;
    }
  }

  for (int y = 0; y < subdivision[1]; y++)
  {
    Real3 pos = top_left + del_y*y;
    int index = y*subdivision[0];
    for (int x = 0; x < subdivision[0]; x++)
    {
      physics_system->distanceConstrain()->addValue(index,
        pos + Real3((((subdivision[1] - y) == 1) ? .2 : 0), 0, 0));
      point_pos.push_back(pos + Real3((((subdivision[1] - y) == 1) ? .2 : 0),
        0, 0));
      //constrain.add(index, index, 0, 0, y ? 1 : 0);
      if (x + 1 < subdivision[0])
      {
        physics_system->distanceConstrain()->add(index, index + 1, 1, x_len);
        physics_system->distanceConstrain()->add(index + 1, index, 1, x_len);
        connection_elements.push_back(index);
        connection_elements.push_back(index + 1);
      }
      if (y + 1 < subdivision[1])
      {
        physics_system->distanceConstrain()->add(index, index + subdivision[0],
          1, y_len);
        physics_system->distanceConstrain()->add(index + subdivision[0], index,
          1, y_len);
        connection_elements.push_back(index);
        connection_elements.push_back(index + subdivision[0]);
      }
      if (x + 1 < subdivision[0] && y + 1 < subdivision[1])
      {
        physics_system->distanceConstrain()->add(index,
          index + subdivision[0] + 1, 1, diag_len);
        physics_system->distanceConstrain()->add(index + subdivision[0] + 1,
          index, 1, diag_len);
        connection_elements.push_back(index);
        connection_elements.push_back(index + subdivision[0] + 1);
      }
      if (x >= 1 && y + 1 < subdivision[1])
      {
        physics_system->distanceConstrain()->add(index,
          index + subdivision[0] - 1, 1, diag_len);
        physics_system->distanceConstrain()->add(index + subdivision[0] - 1,
          index, 1, diag_len);
        connection_elements.push_back(index);
        connection_elements.push_back(index + subdivision[0] - 1);
      }
      index++;
      pos += del_x;
    }
  }
  disp_vertex.gen();
  disp_vertex.copyData(&point_pos[0][0], subdivision[0] * subdivision[1], 0,
    sizeof(Real3));
  disp_elements.gen();
  disp_elements.copyData((GLuint*)&connection_elements[0],
    connection_elements.size());
  connection_elements.clear();
  disp_shader.init("../../ParticlePhysics/display_vert.glsl",
    "../../ParticlePhysics/display_frag.glsl");
  plug.setGLResource(disp_vertex);
  physics_system->distanceConstrain()->exportToDevice();
  //physics_system->distanceConstrain()->show();
}

void Cloth::step()
{
  physics_system->distanceConstrain()->solve();
  CU_PROMPT;
  void* vertex_array = this->plug.map();

  dim3 threads;
  dim3 blocks;
  physics_system->distanceConstrain()->configureGrid(threads, blocks);
  /*
  ConstrainBuffer buf = ConstrainBuffer(1 - (constrain.getIterations() & 1));
  DeviceEntity<Real3>::copy((Real3*)vertex_array,
  (Real3*)constrain.getValueBuffer(buf), constrain.getNodeCount());
  cudaDeviceSynchronize();
  CU_PROMPT;
  */
  DeviceEntity<Real3>::copy((Real3*)vertex_array,
    (Real3*)physics_system->distanceConstrain()->getPosition(),
    physics_system->distanceConstrain()->getNodeCount());
  CU_PROMPT;
  this->plug.unmap();
}

void Cloth::render()
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
  disp_shader.bind();
  disp_shader.set("modelViewMatrix", model_mat);
  disp_shader.set("projectionMatrix", proj_mat);
  disp_vertex.bind();
  GL_CHECK(glEnableVertexAttribArray(0));
  GL_CHECK(glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(Real3),
    (GLvoid*)0));
  //GL_CHECK(glEnableVertexAttribArray(1));
  //GL_CHECK(glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (GLvoid*)0));
  //GL_CHECK(glDrawArrays(GL_POINTS, 0, disp_vertex.count()));
  //GL_CHECK(glDrawElementsInstanced(GL_TRIANGLES, grid_mesh_face.count(), GL_UNSIGNED_INT, 0, pow(simulation_dim[0] - 1, 2)));
  disp_elements.bind();
  GL_CHECK(glDrawElements(GL_LINES, disp_elements.count(), GL_UNSIGNED_INT,
    NULL));
  disp_elements.unbind();
  //GL_CHECK(glDisableVertexAttribArray(1));
  GL_CHECK(glDisableVertexAttribArray(0));
  disp_vertex.unbind();
  disp_shader.unbind();
  glPopMatrix();
}