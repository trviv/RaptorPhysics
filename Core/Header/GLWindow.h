#ifndef GL_WINDOW
#define GL_WINDOW

#include "GLClass.h"
#include "../Vector/Matrix.h"

enum WindowOptionType
{
  WINDOW_OPTION_BOOL,
  WINDOW_OPTION_STRING
};

struct WindowOption
{
  WindowOptionType  type;
  string            name;
  bool              boolValue;
  string            stringValue;

  WindowOption(const string& name, const bool value)
  {
    type = WINDOW_OPTION_BOOL;
    this->name = name;
    boolValue = value;
  }

  WindowOption(const string& name, const string& value)
  {
    type = WINDOW_OPTION_STRING;
    this->name = name;
    stringValue = value;
  }
};

class Window : public ParameterReader
{
  int win_width;
  int win_height;

  int intial_mouse_x;
  int intial_mouse_y;

  float scroll_prev_x;
  float scroll_prev_y;

  float yaw, pitch;

  Real3 cameraUp;
  Real3 cameraFront;
  Real3 cameraPosition;
  float cameraUpSpeed;
  float cameraSideSpeed;
  float cameraForwardSpeed;

protected:

  Real3 down;

  // GUI Frame info relates variables
  Real3   frameTextSize;
  string  frameText;

  Real3   frameOptionSize;
  vector<WindowOption>  frameOptionList;
  map<string, uint>     frameOptionIndex;

  float clearColor[4];

  float modelMatrix[16];
  float projectionMatrix[16];

  void addFrameOption(const WindowOption& option);

  WindowOption& getFrameOption(const string& name);

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

  void scroll(float x, float y);

  void pinch(float d);

  void start();
  void display();

  virtual void render() {};
  virtual void step() {};
};

extern Window *main_window;

#endif
