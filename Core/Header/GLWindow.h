#ifndef GL_WINDOW
#define GL_WINDOW

#include "../Vector/Matrix.h"
#include "UIElements.h"
#include "Animator.h"
#include "ParameterReader.h"

class Window : public ParameterReader
{
  int win_width;
  int win_height;

  int intial_mouse_x;
  int intial_mouse_y;

  float scroll_prev_x;
  float scroll_prev_y;
  float scroll_prev_z;

  float yaw, pitch;

  AnimationElement<Real3> cameraUp;
  AnimationElement<Real3> cameraFront;
  AnimationElement<Real3> cameraPosition;

  float cameraUpSpeed;
  float cameraSideSpeed;
  float cameraForwardSpeed;

protected:

  Real3 down;

  // GUI Frame info related variables
  Real3   frameTextSize;
  string  frameText;

  // Dynamic GUI options
  Real3   frameOptionSize;
  vector<UIElement>           frameOptionList;
  unordered_map<string, uint> frameOptionIndex;

  float clearColor[4];

  float modelMatrix[16];
  float projectionMatrix[16];

  void addFrameOption(const UIElement& option);

  UIElement& getFrameOption(const string& name);

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
