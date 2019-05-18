#include "GLWindow.h"

#if ENV_APPLE
#include "SDL.h"

int Window::del_time = 5;
Window *main_window = NULL;
SDL_Window *sdl_window = NULL;

#define WINDOW_MAX_TRANSLATION_RATE 1.f
#define WINDOW_TRANSLATION_RATE     0.05f
#define WINDOW_ROTATION_SCALE       0.005f

#define clamp(x, y, z) x<y?y:(x>z?z:x);

void Window::init(int argc, char** argv, int width, int height,
  const char* name)
{
  SDL_Init(SDL_INIT_VIDEO);
  
  //glutInit(&argc, argv);
  //glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA | GLUT_STENCIL | GLUT_ACCUM);
  
  win_width = width;
  win_height = height;
  
  // Create an application window with the following settings:
  sdl_window = SDL_CreateWindow(
                                "An SDL2 window",         // window title
                                SDL_WINDOWPOS_UNDEFINED,  // initial x position
                                SDL_WINDOWPOS_UNDEFINED,  // initial y position
                                win_width,                               // width, in pixels
                                win_height,                               // height, in pixels
                                SDL_WINDOW_OPENGL         // flags - see below
                                );
  
  SDL_GL_SetAttribute( SDL_GL_RED_SIZE,     8 );
  SDL_GL_SetAttribute( SDL_GL_GREEN_SIZE,   8 );
  SDL_GL_SetAttribute( SDL_GL_BLUE_SIZE,    8 );
  SDL_GL_SetAttribute( SDL_GL_ALPHA_SIZE,   8 );
  SDL_GL_SetAttribute( SDL_GL_DEPTH_SIZE,   32 );
  SDL_GL_SetAttribute( SDL_GL_DOUBLEBUFFER, 1 );

  //index = glutCreateWindow(name);
  //glutInitWindowSize(width, height);

//  int glew_ok = glewInit();
//  if (glew_ok != GLEW_OK)
//  {
//    std::cout << "Glew Error..." << std::endl;
//  }

  rx = 0;
  ry = 0;
  translate[0] = 0;
  translate[1] = 0;
  translate[2] = .25;
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

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glViewport(0, 0, (GLsizei)width, (GLsizei)height);

  clearColor[0] = 0.0f;
  clearColor[1] = 0.0f;
  clearColor[2] = 0.0f;
  clearColor[3] = 1.0f;
}

void Window::display()
{
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glPushMatrix();

  translationRate[0] *= 0.8f;
  translationRate[1] *= 0.8f;
  translationRate[2] *= 0.8f;
  translate[0] += translationRate[0];
  translate[1] += translationRate[1];
  translate[2] += translationRate[2];

  glClearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);
  glClearDepth(100.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

  glTranslatef(translate[0], translate[1], translate[2]);
  glRotatef(rx, 1, 0, 0);
  glRotatef(ry, 0, 1, 0);
  step();
  render();

  glPopMatrix();
  glFlush();
  SDL_GL_SwapWindow(sdl_window);
  //glutSwapBuffers();
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
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(60, ((float)width) / ((float)height), .01, 100000.0);
  gluLookAt(0, 0, 0, 0, 0, 1, 0, 1, 0);
  glViewport(0, 0, (GLsizei)width, (GLsizei)height);
  win_width = width;
  win_height = height;
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
  //glutMainLoop();
}

void Window::loop()
{
  while (true)
  {
    //glutMainLoopEvent();
  }
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
