#include "GLClass.h"
#include <iostream>
#include <fstream>

GLObject::GLObject()
{
  index = -1;
}

GLObject::~GLObject()
{
  free();
}

void GLObject::free()
{}

void GLObject::bindTex(GLuint tex)const
{
  GL_CHECK(glBindTexture(GL_TEXTURE_2D, tex));
}

void GLObject::bindFBO(GLuint fbo)const
{
  GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, fbo));
}

void GLObject::bindRBO(GLuint rbo)const
{
  GL_CHECK(glBindRenderbuffer(GL_RENDERBUFFER, rbo));
}

void GLObject::bindBuf(GLuint buf)const
{
  GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, buf));
}

void GLObject::bindElemBuf(GLuint buf)const
{
  GL_CHECK(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buf));
}

GLuint GLObject::get()const
{
  return index;
}

Vertex::Vertex()
{}

void Vertex::bind()const
{
  GL_CHECK(glBindVertexArray(vertex_index));
  GL_CHECK(bindBuf(index));
}

void Vertex::unbind()const
{
  GL_CHECK(bindBuf(0));
  GL_CHECK(glBindVertexArray(0));
}

void Vertex::free()
{
  if (index != -1)
  {
    GL_CHECK(glDeleteBuffers(1, &index));
    GL_CHECK(glDeleteBuffers(1, &vertex_index));
    index = -1;
    vertex_index = -1;
  }
}

void Vertex::make(float w, float h)
{
  float vertex[] = {
    -w, -h, 0, 0, 0, w, -h, 0, 1, 0, w, h, 0, 1, 1,
    -w, -h, 0, 0, 0, -w, h, 0, 0, 1, w, h, 0, 1, 1
  };

  copyData(vertex, 6, 0, 5 * SIZEOF_FLOAT);
}

void Vertex::copyData(float vertex[], GLsizei vertex_count, int vertex_width, int vertex_stride)
{
  this->vertex_stride = vertex_stride;
  this->vertex_width = vertex_width;
  this->vertex_count = vertex_count;
  bind();
  GL_CHECK(glBufferData(GL_ARRAY_BUFFER, vertex_count*vertex_stride, vertex, GL_STATIC_DRAW));
  unbind();
}

void Vertex::gen()
{
  GL_CHECK(glGenVertexArrays(1, &vertex_index));
  GL_CHECK(glBindVertexArray(vertex_index));
  GL_CHECK(glGenBuffers(1, &index));
}

void Vertex::render(int mode)
{
  bind();

  unbind();
}

int Vertex::count()const
{
  return vertex_count;
}

void Vertex::attrib(int var_location)
{
  //gl.glVertexAttribPointer(var_location, vertex_width, GL.GL_FLOAT, false, 0, 0L);
}

Face::Face()
{}

void Face::bind()const
{
  bindElemBuf(index);
}

void Face::unbind()const
{
  bindElemBuf(0);
}

void Face::free()
{
  if (index != -1)
  {
    GL_CHECK(glDeleteBuffers(1, &index));
    index = -1;
  }
}

void Face::copyData(GLuint indices[], GLsizei count)
{
  index_count = count;
  bind();
  GL_CHECK(glBufferData(GL_ELEMENT_ARRAY_BUFFER, count*sizeof(GLuint), indices, GL_STATIC_DRAW));
  unbind();
}

void Face::gen()
{
  GL_CHECK(glGenBuffers(1, &index));
}

GLsizei Face::count()const
{
  return index_count;
}

Texture::Texture()
{
}

Texture::Texture(int w, int h)
{
  init(w, h);
}

void Texture::init(int w, int h)
{
  this->w = w;
  this->h = h;
}

void Texture::bind()const
{
  bindTex(index);
}

void Texture::unbind()const
{
  bindTex(0);
}

void Texture::free()
{
  if (index != -1)
  {
    GL_CHECK(glDeleteTextures(1, &index));
    index = -1;
  }
}

void Texture::gen(float buffer[], int type)
{
  if (index == -1)
  {
    GL_CHECK(glGenTextures(1, &index));
  }

  bind();
  if (type == COLOR_BUFFER)
  {
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, buffer));
    //gl.glTexParameterf(GL.GL_TEXTURE_2D, GL2.GL_GENERATE_MIPMAP, GL.GL_TRUE);
  }
  else if (type == 1)
  {
    //      gl.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR);
    //      gl.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR);
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    GL_CHECK(glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    GL_CHECK(glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, buffer));
    //gl.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT);
    //gl.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT);
    //gl.glTexParameterf(GL.GL_TEXTURE_2D, GL2.GL_GENERATE_MIPMAP, GL.GL_TRUE);
  }
  else if (type == DEPTH_BUFFER)
  {
    //      gl.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR);
    //      gl.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR);
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    GL_CHECK(glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    GL_CHECK(glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, w, h, 0, GL_DEPTH_COMPONENT, GL_FLOAT, buffer));
    //gl.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT);
    //gl.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT);
    //gl.glTexParameterf(GL.GL_TEXTURE_2D, GL2.GL_GENERATE_MIPMAP, GL.GL_TRUE);
  }

  //GL.GL_RGBA32F
  unbind();
}

void Texture::gen()
{
  gen(NULL, COLOR_BUFFER);
}

void Texture::copy(float image[])const
{
  bind();
  GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, image));
  unbind();
}

void Texture::copy(float image[], GLint x_off, GLint y_off, GLsizei width, GLsizei height)const
{
  bind();
  GL_CHECK(glTexSubImage2D(GL_TEXTURE_2D, 0, x_off, y_off, width, height, GL_RGBA, GL_FLOAT, image));
  unbind();
}

void Texture::copy(float image[], GLint x_off, GLint y_off, GLsizei length)const
{
  bind();
  if (length > width())
  {
    GL_CHECK(glTexSubImage2D(GL_TEXTURE_2D, 0, x_off, y_off, this->width(), length / this->width(), GL_RGBA, GL_FLOAT, image));
    if (length & (this->width() - 1))
    {
      GL_CHECK(glTexSubImage2D(GL_TEXTURE_2D, 0, x_off, y_off + length / this->width(), length & (this->width() - 1), 1, GL_RGBA, GL_FLOAT, image + 4 * this->width() * (length / this->width()) ));
    }
  }
  else
  {
    GL_CHECK(glTexSubImage2D(GL_TEXTURE_2D, 0, x_off, y_off, length, 1, GL_RGBA, GL_FLOAT, image));
  }
  unbind();
}

void Texture::copy()const
{
  bind();
  GL_CHECK(glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 0, 0, w, h, 0));
  unbind();
}

GLuint Texture::get()const
{
  return GLObject::get();
}

void Texture::get(float target[])const
{
  bind();
  GL_CHECK(glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, target));
  unbind();
}

Render::Render()
{}

void Render::bind()const
{
  bindRBO(index);
}

void Render::unbind()const
{
  bindRBO(0);
}

void Render::gen()
{
  GL_CHECK(glGenRenderbuffers(1, &index));
}

void Render::free()
{
  if (index != -1)
  {
    GL_CHECK(glDeleteRenderbuffers(1, &index));
    index = -1;
  }
}

Frame::Frame()
{}

void Frame::bind()const
{
  bindFBO(index);
}

void Frame::unbind()const
{
  bindFBO(default_frame);
}

void Frame::gen()
{
  GL_CHECK(glGenFramebuffers(1, &index));
}

void Frame::free()
{
  if (index != -1)
  {
    GL_CHECK(glDeleteFramebuffers(1, &index));
    index = -1;
  }
}

Renderer::Renderer()
{
  /*
  GL_CHECK(glEnable(GL_DEPTH_TEST));
  GL_CHECK(glEnable(GL_LIGHTING));
  GL_CHECK(glEnable(GL_TEXTURE_2D));
  GL_CHECK(glDisable(GL_CULL_FACE));
  GL_CHECK(glEnable(GL_BLEND));
  GL_CHECK(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
  */
}

void Renderer::bind()const
{
  fbo.bind();
  GL_CHECK(glViewport(0, 0, w, h));
  GL_CHECK(glClearColor(0, 0, 0, 0));
  GL_CHECK(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));
}

void Renderer::unbind()const
{
  fbo.unbind();
}

void Renderer::init(GLsizei w, GLsizei h)
{
  this->w = w;
  this->h = h;
  tex.init(w, h);
  tex.gen();
  //depth.init(w,h);
  //depth.gen(NULL,Texture::DEPTH_BUFFER);
  fbo.gen();
  rbo.gen();

  //then create a render buffer
  rbo.bind();
  //ask for a depth buffer and sets the size
  GL_CHECK(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, w, h));
  rbo.unbind();

  fbo.bind();
  GL_CHECK(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex.get(), 0));
  //GL_CHECK(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth.get(), 0));
  GL_CHECK(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo.get()));
  //check for completeness
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    std::cerr << "Init FBO: rendering to texture could not be initialised." << std::endl;
  fbo.unbind();
}

void Renderer::free()
{
  tex.free();
  fbo.free();
  rbo.free();
}

Renderer::~Renderer()
{
  free();
}

unsigned long getFileLength(std::ifstream& file)
{
  if (!file.good()) return 0;

  unsigned long pos = (unsigned long)file.tellg();
  file.seekg(0, std::ios::end);
  unsigned long len = (unsigned long)file.tellg();
  file.seekg(std::ios::beg);

  return len;
}

int loadShader(const char* filename, GLchar** shader_source, GLint* len)
{
  std::string data = readFile(filename);
  *shader_source = new GLchar[data.size()];
  *len = data.size();
  memcpy(*shader_source, data.c_str(), *len);

  return 0; // No Error
}

void unloadShader(GLchar** ShaderSource)
{
  if (*ShaderSource != NULL) delete[] * ShaderSource;
  *ShaderSource = NULL;
}

bool checkShader(GLuint shader, const char* file)
{
  GLint compiled;
  GLsizei len;
  GL_CHECK(glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled));
  GLchar log[512];
  log[0] = 0;
  if (!compiled)
  {
    glGetShaderInfoLog(shader, 512, &len, log);
    std::cerr << "Shader: " << file << std::endl <<
      "Compilation issue: " << std::endl << log << std::endl;
    return false;
  }
  std::cout << "Shader: " << file << std::endl <<
    "Compilation log: " << std::endl << log << std::endl;
  return true;
}

Shader::Shader()
{
  program = -1;
  vertex_shader = -1;
  fragment_shader = -1;
}

Shader::~Shader()
{
  if (program != -1)
  {
    GL_CHECK(glDeleteProgram(program));
    program = -1;
  }
  if (vertex_shader != -1)
  {
    GL_CHECK(glDeleteShader(vertex_shader));
    vertex_shader = -1;
  }
  if (fragment_shader != -1)
  {
    GL_CHECK(glDeleteShader(fragment_shader));
    fragment_shader = -1;
  }
}

Shader::Shader(const char* vert, const char* frag)
{
  init(vert, frag);
}

void Shader::init(const char* vert, const char* frag)
{
  GLchar* vertex_program = NULL, *fragment_program = NULL;
  GLint vertex_len, fragment_len;
  if (loadShader(vert, &vertex_program, &vertex_len) == 0 &&
    loadShader(frag, &fragment_program, &fragment_len) == 0)
  {
    vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);

    GL_CHECK(glShaderSource(vertex_shader, 1, &vertex_program, &vertex_len));
    GL_CHECK(glShaderSource(fragment_shader, 1, &fragment_program, &fragment_len));

    GL_CHECK(glCompileShader(vertex_shader));
    GL_CHECK(glCompileShader(fragment_shader));

    if (checkShader(vertex_shader, vert) && checkShader(fragment_shader, frag))
    {
      program = glCreateProgram();
      GL_CHECK(glAttachShader(program, vertex_shader));
      GL_CHECK(glAttachShader(program, fragment_shader));
    }
  }
  unloadShader(&vertex_program);
  unloadShader(&fragment_program);
}

void Shader::bindLocation(GLuint index, const GLchar *name)
{
  GL_CHECK(glBindAttribLocation(program, index, name));
  GL_CHECK(glEnableVertexAttribArray(index));
}

void Shader::linkPrograms()
{
  GL_CHECK(glLinkProgram(program));
  GLint linked;
  GL_CHECK(glGetProgramiv(program, GL_LINK_STATUS, &linked));
  if (!linked)
  {
    GLchar log[512];
    GLsizei len;
    log[0] = 0;
    GL_CHECK(glGetProgramInfoLog(program, 512, &len, log));
    std::cerr << "Cannot link program for shaders..." << std::endl;
    std::cerr << log << std::endl;
  }
  else
  {
    GLchar log[512];
    GLsizei len;
    log[0] = 0;
    GL_CHECK(glGetProgramInfoLog(program, 512, &len, log));
    std::cerr << log << std::endl;
  }
}

void Shader::bind()const
{
  GL_CHECK(glUseProgram(program));
}

void Shader::unbind()const
{
  GL_CHECK(glUseProgram(0));
}

void Shader::set(const char* uniform_name, float v0)const
{
  GLint loc = glGetUniformLocation(program, uniform_name);
  GL_CHECK(glUniform1f(loc, v0));
}

void Shader::set(const char* uniform_name, float v0, float v1)const
{
  GLint loc = glGetUniformLocation(program, uniform_name);
  GL_CHECK(glUniform2f(loc, v0, v1));
}

void Shader::set(const char* uniform_name, float v0, float v1, float v2, float v3)const
{
  GLint loc = glGetUniformLocation(program, uniform_name);
  GL_CHECK(glUniform4f(loc, v0, v1, v2, v3));
}

void Shader::set(const char* uniform_name, int v0)const
{
  GLint loc = glGetUniformLocation(program, uniform_name);
  GL_CHECK(glUniform1i(loc, v0));
}

void Shader::set(const char* uniform_name, int v0, int v1)const
{
  GLint loc = glGetUniformLocation(program, uniform_name);
  GL_CHECK(glUniform2i(loc, v0, v1));
}

void Shader::set(const char* uniform_name, int v0, int v1, int v2)const
{
  GLint loc = glGetUniformLocation(program, uniform_name);
  GL_CHECK(glUniform3i(loc, v0, v1, v2));
}

void Shader::set(const char* uniform_name, int v0, int v1, int v2, int v3)const
{
  GLint loc = glGetUniformLocation(program, uniform_name);
  GL_CHECK(glUniform4i(loc, v0, v1, v2, v3));
}

void Shader::activateTexture(const char* uniform_name, GLint index, const Texture& tex)const
{
  GLint loc = glGetUniformLocation(program, uniform_name);
  GL_CHECK(glActiveTexture(GL_TEXTURE0 + index));
  tex.bind();
  GL_CHECK(glUniform1i(loc, index));
}

void Shader::set(const GLchar* uniform_name, const float matrix[])const
{
  GLint loc = glGetUniformLocation(program, uniform_name);
  GL_CHECK(glUniformMatrix4fv(loc, 1, GL_FALSE, matrix));
}
