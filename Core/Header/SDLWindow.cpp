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

static bool quit = false;

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
    quit = true;
  }

  return 1;
}

void setProjectionMatrix(float result[], float aspect)
{
  float fov = 60.f;
  float farDist = 100.f;
  float nearDist = .01f;
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

// Custom implementation of the LookAt function
void setLookAtMatrix(float view[], const Real3& position, const Real3& target, Real3 up)
{
  Real3 zaxis = position - target;
  zaxis.normalize();

  up.normalize();
  Real3 xaxis = up.cross(zaxis);
  xaxis.normalize();

  Real3 yaxis = zaxis.cross(xaxis);

  Matrix4 translation;
  translation.set(Matrix3::getIdentity(), Real3(-position.x, -position.y, -position.z));

  Matrix4 rotation;
  rotation.setIdentity();
  rotation.set(xaxis.x, xaxis.y, xaxis.z, 0, yaxis.x, yaxis.y, yaxis.z, 0, zaxis.x, zaxis.y, zaxis.z, 0);

  rotation *= translation;

  for (int i=0; i<3; i++)
  {
    Real3 col = rotation.getColumn(i);
    view[i*4+0] = col[0];
    view[i*4+1] = col[1];
    view[i*4+2] = col[2];
    view[i*4+3] = 0.f;
  }

  Real3 col = rotation.getPos();
  view[12] = col[0];
  view[13] = col[1];
  view[14] = col[2];
  view[15] = 1.f;
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
  sdl_window = SDL_CreateWindow(name, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
    win_width, win_height, SDL_WINDOW_RESIZABLE|SDL_WINDOW_OPENGL);
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

  cameraForwardSpeed = 0.f;
  cameraSideSpeed = 0.f;

  GL_CHECK(glViewport(0, 0, (GLsizei)width, (GLsizei)height));

  clearColor[0] = 0.0f;
  clearColor[1] = 0.0f;
  clearColor[2] = 0.0f;
  clearColor[3] = 1.0f;

  // camera settings
  cameraUp = Real3(0.0f, 1.0f, 0.0f);
  cameraFront = Real3(0.0f, 0.0f, -1.0f);
  cameraPosition = Real3(0.0f, 0.0f, 0.0f);

  yaw   = -90.0f;
  pitch =  0.0f;
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
      cameraForwardSpeed += WINDOW_TRANSLATION_RATE;
      cameraForwardSpeed = clamp(cameraForwardSpeed, -WINDOW_MAX_TRANSLATION_RATE, WINDOW_MAX_TRANSLATION_RATE);
      break;
    case 's':
      cameraForwardSpeed -= WINDOW_TRANSLATION_RATE;
      cameraForwardSpeed = clamp(cameraForwardSpeed, -WINDOW_MAX_TRANSLATION_RATE, WINDOW_MAX_TRANSLATION_RATE);
      break;
    case 'a':
      cameraSideSpeed -= WINDOW_TRANSLATION_RATE;
      cameraSideSpeed = clamp(cameraSideSpeed, -WINDOW_MAX_TRANSLATION_RATE, WINDOW_MAX_TRANSLATION_RATE);
      break;
    case 'd':
      cameraSideSpeed += WINDOW_TRANSLATION_RATE;
      cameraSideSpeed = clamp(cameraSideSpeed, -WINDOW_MAX_TRANSLATION_RATE, WINDOW_MAX_TRANSLATION_RATE);
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
  setProjectionMatrix(projectionMatrix, (float)this->width()/(float)this->height());
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
  float deltaY = float(intial_mouse_y - y);

  intial_mouse_x = x;
  intial_mouse_y = y;

  float sensitivity = 1.f;
  deltaX *= sensitivity;
  deltaY *= sensitivity;

  // update angles
  yaw   += deltaX;
  yaw   = yaw>360 ? (yaw-360) : yaw<360 ? (yaw+360) : yaw;
  pitch += deltaY;

  if (pitch > 89.0f)  pitch = 89.0f;
  if (pitch < -89.0f) pitch = -89.0f;

  // update camera look at
  cameraFront.x = cos(yaw * M_PI / 180.f) * cos(pitch * M_PI / 180.f);
  cameraFront.y = sin(pitch * M_PI / 180.f);
  cameraFront.z = sin(yaw * M_PI / 180.f) * cos(pitch * M_PI / 180.f);
  cameraFront.normalize();

//  printf("Yaw: %f, Pitch: %f\n", yaw, pitch);
//  printf("Front: %f %f %f\n", cameraFront.x, cameraFront.y, cameraFront.z);
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
  SDL_CheckError();

  setProjectionMatrix(projectionMatrix, (float)width()/(float)height());

  bool mouseDown = false;
  Real3 eyeVector(0.f);

  while (!quit)
  {
    // update camera settings
    Real3 cross = cameraFront.cross(cameraUp);
    cross.normalize();
    cameraPosition += cameraFront * cameraForwardSpeed + cross * cameraSideSpeed;

    setLookAtMatrix(modelMatrix, cameraPosition, cameraPosition + cameraFront, cameraUp);

    cameraForwardSpeed *= 0.8f;
    cameraSideSpeed *= 0.8f;

    // handle events
    SDL_Event event;
    while (SDL_PollEvent(&event))
    {
      switch (event.type)
      {
        case SDL_QUIT:
        {
          quit = true;
        }
          break;

        case SDL_WINDOWEVENT:
        {
          switch (event.window.event)
          {
            case SDL_WINDOWEVENT_RESIZED:
              reshape(event.window.data1, event.window.data2);
              break;
          }
        }
          break;

        case SDL_FINGERDOWN:
        {
          if (!mouseDown)
          {
            mouse(0, 0, event.tfinger.x*width(), event.tfinger.y*height());
            mouseDown = true;
          }
          SDL_Init (SDL_INIT_EVENTS);
        }
          break;

        case SDL_FINGERUP:
        {
          mouseDown = false;
          SDL_Init (SDL_INIT_EVENTS);
        }
          break;

        case SDL_FINGERMOTION:
        {
          if (mouseDown)
          {
            mouseDrag(event.tfinger.x*width(), event.tfinger.y*height());
          }
          SDL_Init (SDL_INIT_EVENTS);
        }
          break;

        case SDL_KEYDOWN:
        {
          if (event.key.keysym.sym == SDLK_ESCAPE)
          {
            quit = true;
          }
          else
          {
            this->keyboard(event.key.keysym.sym, 0, 0);
          }
        }
          break;

        default:
          break;
      }
    }

    // main work
    display();

    SDL_GL_SwapWindow(sdl_window);
    SDL_CheckError();
  }
}

#endif
