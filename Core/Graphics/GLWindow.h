#ifndef GL_WINDOW
#define GL_WINDOW

#include <Vector/Matrix.h>
#include <GUI/UIFrame.h>
#include <Utils/Animator.h>
#include <Compute/ComputeGraphicsSharedTexture.h>

class Window
{
  friend class ReaderScene;
  
  int win_width;
  int win_height;

  int intial_mouse_x;
  int intial_mouse_y;

  float scroll_prev_x;
  float scroll_prev_y;
  float scroll_prev_z;

  float initialYaw, initialPitch;
  float yaw, pitch;

  AnimationElement<Real3> cameraUp;
  AnimationElement<Real3> cameraFront;
  AnimationElement<Real3> cameraPosition;

  float cameraUpSpeed;
  float cameraSideSpeed;
  float cameraForwardSpeed;

protected:
  float controlWindowHeight, controlWindowSidePos, controlWindowBottomPos;
  Texture moveControlImageBack, moveControlImage;
  Real3 down;

  UIList  uiFrames;
  int     forceRefreshUICount;

  float clearColor[4];

  float modelMatrix[16];
  float projectionMatrix[16];

  void processOnScreenController(void* eventData, bool end);

public:
  UIFrame* statFrame, *optionFrame, *timeSliderFrame;

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

  ComputeGraphicsSharedTexture createSharedTexture(ComputeInterface* compute, uint textureSize[2], SharedTextureFormat textureFormat);
};

extern Window *main_window;

#endif
