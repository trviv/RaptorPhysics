#ifndef GL_CLASS
#define GL_CLASS

#include "ComputeInterface.h"
#include "ParameterReader.h"

const int SIZEOF_INT = sizeof(int);
const int SIZEOF_FLOAT = sizeof(float);
const int SIZEOF_BYTE = sizeof(unsigned char);

static void CheckOpenGLError(const char* stmt, const char* fname, int line)
{
  GLenum err = glGetError();
  if (err != GL_NO_ERROR)
  {
    printf("OpenGL error %08x, at %s:%i - for %s\n", err, fname, line, stmt);
    abort();
  }
}

#define GL_CHECK(stmt) do { \
  stmt; \
  CheckOpenGLError(" ", __FILE__, __LINE__); \
} while (0)

class Texture;

class Shader
{

  GLuint program;
  GLuint vertex_shader;
  GLuint fragment_shader;

public:
  Shader();

  Shader(const char* vert, const char* frag);

  void init(const char* vert, const char* frag);

  void bindLocation(GLuint index, const GLchar* name);

  void linkPrograms();

  ~Shader();

  void bind()const;

  void unbind()const;

  void set(const GLchar* uniform_name, float v0)const;

  void set(const GLchar* uniform_name, float v0, float v1)const;

  void set(const GLchar* uniform_name, float v0, float v1, float v2, float v3)const;

  void set(const GLchar* uniform_name, int v0)const;

  void set(const GLchar* uniform_name, int v0, int v1)const;

  void set(const GLchar* uniform_name, int v0, int v1, int v2)const;

  void set(const GLchar* uniform_name, int v0, int v1, int v2, int v3)const;

  void set(const GLchar* uniform_name, const float matrix[])const;

  void activateTexture(const GLchar* uniform_name, GLint index, const Texture& tex)const;
};

class GLObject
{

protected:
  GLuint index;

  void bindTex(GLuint tex)const;

  void bindFBO(GLuint fbo)const;

  void bindRBO(GLuint rbo)const;

  void bindBuf(GLuint buf)const;

  void bindElemBuf(GLuint buf)const;

public:
  GLObject();

  virtual ~GLObject();

  GLuint get()const;

  virtual void bind()const = 0;

  virtual void unbind()const = 0;

  virtual void gen() = 0;

  virtual void free();
};

class Frame : public GLObject
{
  static const GLuint default_frame = 0;

public:
  Frame();

  void bind()const;

  void unbind()const;

  void gen();

  void free();
};

class Render :public GLObject
{
public:
  Render();

  void bind()const;

  void unbind()const;

  void gen();

  void free();
};

enum TextureFormat
{
  TEXTURE_FORMAT_FLOAT = GL_FLOAT,
  TEXTURE_FORMAT_INT = GL_INT,
  TEXTURE_FORMAT_UBYTE = GL_UNSIGNED_BYTE
};

class Texture : public GLObject
{
  GLsizei w, h;
  TextureFormat format;

public:
  enum TextureGen
  {
    COLOR_BUFFER,
    COLOR_BUFFER_FLOAT,
    DEPTH_BUFFER,
    COLOR_BUFFER_UCHAR
  };

public:
  Texture(TextureFormat format = TEXTURE_FORMAT_FLOAT);

  Texture(GLsizei w, GLsizei h);

  void init(GLsizei w, GLsizei h);

  void bind()const;

  void unbind()const;

  void free();

  void gen(float buffer[], TextureGen type);

  void gen();

  void copy(float image[])const;

  void copy(float image[], GLint x_off, GLint y_off, GLsizei width, GLsizei height)const;

  void copy(float image[], GLint x_off, GLint y_off, GLsizei length)const;

  void copy()const;

  GLuint get()const;

  void get(float target[])const;

  GLsizei width()       { return w; }
  GLsizei width()const  { return w; }

  GLsizei height()      { return h; }
  GLsizei height()const { return h; }
};

class Renderer
{
  GLsizei w, h;

public:
  Texture tex;
  Frame   fbo;
  Render  rbo;
  Texture depth;

  Renderer();

  ~Renderer();

  void bind()const;

  void unbind()const;

  void init(GLsizei w, GLsizei h);

  void free();
};

class Vertex :public GLObject
{
  GLuint vertex_index;
  GLsizei vertex_stride;
  int vertex_width;
  GLsizei vertex_count;

public:
  Vertex();

  void bind()const;

  void unbind()const;

  void make(float w, float h);

  void copyData(float vertex[], GLsizei vertex_count, int vertex_width, GLsizei vertex_stride);

  void gen();

  void free();

  void render(int mode);

  void attrib(int var_location);

  int count()const;
};

class Face :public GLObject
{
  GLsizei index_count;

public:
  Face();

  void bind()const;

  void unbind()const;

  void copyData(GLuint indices[], GLsizei count);

  void gen();

  void free();

  GLsizei count()const;
};

#endif
