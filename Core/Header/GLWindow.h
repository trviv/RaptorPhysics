#ifndef GL_WINDOW
#define GL_WINDOW

#include "GLClass.h"

extern void glwRefreshTimer(int value);
extern void glwDisplay();
extern void glwKeyboard(unsigned char key, int x, int y);
extern void glwReshape(int width, int height);
extern void glwMouseWheel(int button, int dir, int x, int y);
extern void glwMouse(int button, int dir, int x, int y);
extern void glwMouseDrag(int x, int y);

class Window
{
protected:
  int index;

  int win_width;
  int win_height;

  int intial_mouse_x;
  int intial_mouse_y;

  float rx, ry, dx, dy, dz;

public:
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

  void display();
  bool keyboard(unsigned char key, int x, int y);
  void reshape(int width, int height);
  void mouse(int button, int dir, int x, int y);
  void mouseDrag(int x, int y);
  void mouseWheel(int button, int dir, int x, int y);
  void start();

  virtual void render() {};
  virtual void step() {};
};

extern Window *main_window;

#endif