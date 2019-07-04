#ifndef GL_WINDOW
#define GL_WINDOW

#include "GLClass.h"
#include "../Vector/Matrix.h"

class Window
{
  int win_width;
  int win_height;

  int intial_mouse_x;
  int intial_mouse_y;

  float yaw, pitch;

  Real3 cameraUp;
  Real3 cameraFront;
  Real3 cameraPosition;
  float cameraUpSpeed;
  float cameraSideSpeed;
  float cameraForwardSpeed;

protected:

  float clearColor[4];

  float modelMatrix[16];
  float projectionMatrix[16];

public:

  virtual ~Window();

  static int del_time;

  int width()
  {
    return win_width;
  }

  int height()
  {
    return win_height;
  }

  virtual void init(int argc, char** argv, int width = 512, int height = 512,
    const char* name = "GL Window");

  bool keyboard(unsigned char key, int x, int y);
  void reshape(int width, int height);
  void mouse(int button, int dir, int x, int y);
  void mouseDrag(int x, int y);
  void mouseWheel(int button, int dir, int x, int y);

  void start();
  void display();

  virtual void render() {};
  virtual void step() {};
};

extern Window *main_window;

#endif
