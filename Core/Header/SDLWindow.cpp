#include "GLWindow.h"

#if ENV_APPLE

#include <SDL2/SDL.h>

int Window::del_time = 5;
Window        *main_window = NULL;
SDL_Window    *sdl_window = NULL;
SDL_GLContext gl_context;

#define WINDOW_MAX_TRANSLATION_RATE 1.f
#define WINDOW_TRANSLATION_RATE     0.05f
#define WINDOW_ROTATION_SCALE       0.005f

#define clamp(x, y, z) x<y?y:(x>z?z:x);

static bool quitting = false;

void SDL_CheckError()
{
  const char *sdl_error = SDL_GetError();
  if (sdl_error[0])
  {
    std::cout << "Error: " << sdl_error << std::endl;
    abort();
    }
}

int SDLCALL watch(void *userdata, SDL_Event* event) {

  if (event->type == SDL_APP_WILLENTERBACKGROUND) {
    quitting = true;
  }

  return 1;
}

void ComputeFOVProjection(float result[], float fov, float aspect, float nearDist, float farDist)
{
  float scale = tan(0.5f * fov * M_PI / 180.f) * nearDist;
  float r = aspect * scale;
  float l = -r;
  float t = scale;
  float b = -t;

  result[0] = 2.f * nearDist / (r - l);
  result[1] = 0;
  result[2] = 0;
  result[3] = 0;

  result[4] = 0;
  result[5] = 2.f * nearDist / (t - b);
  result[6] = 0;
  result[7] = 0;

  result[8] = (r + l) / (r - l);
  result[9] = (t + b) / (t - b);
  result[10] = -(farDist + nearDist) / (farDist - nearDist);
  result[11] = -1.f;

  result[12] = 0;
  result[13] = 0;
  result[14] = -2.f * farDist * nearDist / (farDist - nearDist);
  result[15] = 0;
}

void Window::init(int argc, char** argv, int width, int height,
                  const char* name)
{
  if (SDL_Init(SDL_INIT_VIDEO) < 0)
  {
    printf ("SDL_Init failed: %s\n", SDL_GetError());
    assert(0);
  }

  win_width = width;
  win_height = height;

  SDL_GL_SetAttribute(SDL_GL_RED_SIZE,     8);
  SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE,   8);
  SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE,    8);
  SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE,   8);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE,   32);
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

  SDL_CheckError();

  // Create an application window with the following settings:
  sdl_window = SDL_CreateWindow("Particle Physics", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                win_width, win_height,
                                SDL_WINDOW_RESIZABLE|SDL_WINDOW_OPENGL);
  SDL_CheckError();

  gl_context = SDL_GL_CreateContext(sdl_window);
  SDL_CheckError();

  SDL_GL_MakeCurrent(sdl_window, gl_context);
  SDL_CheckError();

  if (SDL_GL_SetSwapInterval(1))
  {
    printf ("Warning: Unable to set VSync! SDL Error: %s\n", SDL_GetError());
  }

  std::cout<<glGetString(GL_VERSION)<<"\n";

  rx = 0;
  ry = 0;
  translate[0] = 0;
  translate[1] = 0;
  translate[2] = 10.25;
  translationRate[0] = 0.f;
  translationRate[1] = 0.f;
  translationRate[2] = 0.f;

  /*glutTimerFunc(del_time, glwRefreshTimer, 0);

   glutDisplayFunc(glwDisplay);
   glutKeyboardFunc(glwKeyboard);
   glutMouseWheelFunc(glwMouseWheel);
   glutMouseFunc(glwMouse);
   glutMotionFunc(glwMouseDrag);
   glutReshapeFunc(glwReshape);*/

  GL_CHECK(glViewport(0, 0, (GLsizei)width, (GLsizei)height));

  clearColor[0] = 0.0f;
  clearColor[1] = 0.0f;
  clearColor[2] = 0.0f;
  clearColor[3] = 1.0f;
}

Window::~Window()
{
  SDL_DelEventWatch(watch, NULL);
  SDL_GL_DeleteContext(gl_context);
  SDL_DestroyWindow(sdl_window);
  SDL_Quit();
}

void Window::display()
{
  GL_CHECK(glViewport(0, 0, (GLsizei)width(), (GLsizei)height()));
  GL_CHECK(glClearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]));
  GL_CHECK(glClearDepthf(100.0f));
  GL_CHECK(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));

  step();
  render();
}

bool Window::keyboard(unsigned char key, int x, int y)
{
  switch (key)
  {
    case 'w':
      translationRate[2] -= WINDOW_TRANSLATION_RATE;
      translationRate[2] = clamp(translationRate[2], -WINDOW_MAX_TRANSLATION_RATE, WINDOW_MAX_TRANSLATION_RATE);
      break;
    case 's':
      translationRate[2] += WINDOW_TRANSLATION_RATE;
      translationRate[2] = clamp(translationRate[2], -WINDOW_MAX_TRANSLATION_RATE, WINDOW_MAX_TRANSLATION_RATE);
      break;
    case 'a':
      translationRate[0] -= WINDOW_TRANSLATION_RATE;
      translationRate[0] = clamp(translationRate[0], -WINDOW_MAX_TRANSLATION_RATE, WINDOW_MAX_TRANSLATION_RATE);
      break;
    case 'd':
      translationRate[0] += WINDOW_TRANSLATION_RATE;
      translationRate[0] = clamp(translationRate[0], -WINDOW_MAX_TRANSLATION_RATE, WINDOW_MAX_TRANSLATION_RATE);
      break;
    case 'r':
      return false;
      break;
    case ' ':
      return false;
      break;
    case 'm':
      return false;
      break;
    case 'g':
      return false;
      break;
    case 'o':
      return false;
      break;
    case 'i':
      return false;
      break;
    case ',':
      return false;
      break;
    case '.':
      return false;
      break;
    case ';':
      return false;
      break;
    case '\'':
      return false;
      break;
    case '[':
      return false;
      break;
    case ']':
      return false;
      break;
    case '-':
      return false;
      break;
    case '=':
      return false;
      break;
    case '9':
      return false;
      break;
    case '0':
      return false;
      break;
    default:
      return false;
      break;
  }
  return false;
}

void Window::reshape(int width, int height)
{
  GL_CHECK(glViewport(0, 0, (GLsizei)width, (GLsizei)height));
  win_width = width;
  win_height = height;
  ComputeFOVProjection(projectionMatrix, 60.f, (float)this->width()/(float)this->height(), .01f, 100.f);
}

void Window::mouse(int button, int dir, int x, int y)
{
  //if (button == GLUT_LEFT_BUTTON)
  {
    intial_mouse_x = x;
    intial_mouse_y = y;
  }
}

void Window::mouseDrag(int x, int y)
{
  float deltaX = float(x - intial_mouse_x);
  float deltaY = float(y - intial_mouse_y);

  ry += WINDOW_ROTATION_SCALE * deltaX;
  rx -= WINDOW_ROTATION_SCALE * deltaY;
}

void Window::mouseWheel(int button, int dir, int x, int y)
{
  if (dir > 0)
  {
    //dz *= .9f;
  }
  else
  {
    //dz /= .9f;
  }
}

void Window::start()
{
  if (SDL_GetError()[0])
  {
    std::cout << "Error: " << SDL_GetError() << std::endl;
    abort();
  }

  while(!quitting)
  {
    for (int i=0;i<16;i++)
    {
      modelMatrix[i] = 0.f;
    }
    modelMatrix[0]  = 1.f;
    modelMatrix[5]  = 1.f;
    modelMatrix[10] = 1.f;
    modelMatrix[15] = 1.f;

    modelMatrix[12] = -translate[0];
    modelMatrix[13] = -translate[1];
    modelMatrix[14] = -translate[2];

    translationRate[0] *= 0.8f;
    translationRate[1] *= 0.8f;
    translationRate[2] *= 0.8f;
    translate[0] += translationRate[0];
    translate[1] += translationRate[1];
    translate[2] += translationRate[2];

    ComputeFOVProjection(projectionMatrix, 67.f, (float)width()/(float)height(), .01f, 100.f);

    SDL_Event event;
    if (SDL_GetError()[0])
    {
      std::cout << "Error: " << SDL_GetError() << std::endl;
      abort();
    }
    while(SDL_PollEvent(&event) != 0)
    {
      if(event.type == SDL_QUIT)
      {
        quitting = true;
      }
      if (event.type == SDL_WINDOWEVENT)
      {
        switch (event.window.event)
        {
          case SDL_WINDOWEVENT_RESIZED:
            reshape(event.window.data1, event.window.data2);
            break;
        }
      }
      if (event.type == SDL_FINGERDOWN)
      {

      }
      if (event.type == SDL_FINGERUP)
      {

      }
      if (event.type == SDL_FINGERMOTION)
      {

      }
      if (event.type == SDL_KEYDOWN)
      {
        if (event.key.type == SDLK_ESCAPE)
        {
          quitting = true;
        }
        else
        {
          this->keyboard(event.key.keysym.sym, 0, 0);
        }
      }
    }

    display();

    if (SDL_GetError()[0])
    {
      std::cout << "Error: " << SDL_GetError() << std::endl;
      abort();
    }
//    SDL_Delay(10);
    SDL_GL_SwapWindow(sdl_window);
  }
}

void Window::loop()
{
}

void glwRefreshTimer(int value)
{
  //glutPostRedisplay();
  //glutTimerFunc(Window::del_time, glwRefreshTimer, 0);
}

void glwDisplay()
{
  main_window->display();
}

void glwKeyboard(unsigned char key, int x, int y)
{
  main_window->keyboard(key, x, y);
}

void glwReshape(int width, int height)
{
  main_window->reshape(width, height);
}

void glwMouseWheel(int button, int dir, int x, int y)
{
  main_window->mouseWheel(button, dir, x, y);
}

void glwMouse(int button, int dir, int x, int y)
{
  main_window->mouse(button, dir, x, y);
}

void glwMouseDrag(int x, int y)
{
  main_window->mouseDrag(x, y);
}

#endif
