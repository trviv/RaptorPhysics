#include "GLWindow.h"

int Window::del_time = 5;
Window *main_window = NULL;

void Window::init(int argc, char** argv, int width, int height,
  const char* name)
{
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA | GLUT_STENCIL
    | GLUT_ACCUM);

  win_width = width;
  win_height = height;

  index = glutCreateWindow(name);
  glutInitWindowSize(width, height);

  int glew_ok = glewInit();
  if (glew_ok != GLEW_OK)
  {
    std::cout << "Glew Error..." << std::endl;
  }

  rx = 0;
  ry = 0;
  dx = 0;
  dy = 0;
  dz = .25;

  glutTimerFunc(del_time, glwRefreshTimer, 0);

  glutDisplayFunc(glwDisplay);
  glutKeyboardFunc(glwKeyboard);
  glutMouseWheelFunc(glwMouseWheel);
  glutMouseFunc(glwMouse);
  glutMotionFunc(glwMouseDrag);
  glutReshapeFunc(glwReshape);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  //glEnable(GL_DEPTH_TEST);  // Enables Depth Testing
  //glDepthFunc(GL_LEQUAL);

  glViewport(0, 0, (GLsizei)width, (GLsizei)height);

  clearColor[0] = 0.0f;
  clearColor[1] = 0.0f;
  clearColor[2] = 0.0f;
  clearColor[3] = 1.0f;
}

void Window::display()
{
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glPushMatrix();

  glClearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);
  glClearDepth(100.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

  gluPerspective(60, float(win_width) / float(win_height), 0.1, 100.0);
  gluLookAt(0, 0, 1, 0, 0, 0, 0, 1, 0);

  glViewport(0, 0, win_width, win_height);
  glRotatef(rx, 1, 0, 0);
  glRotatef(ry, 0, 1, 0);
  glScalef(abs(dz), abs(dz), abs(dz));

  step();
  render();

  glPopMatrix();
  glFlush();
  glutSwapBuffers();
}

bool Window::keyboard(unsigned char key, int x, int y)
{
  switch (key)
  {
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
  case 'd':
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
  gluPerspective(60, ((float)width) / ((float)height), .001, 100000.0);
  gluLookAt(0, 0, 1, 0, 0, 0, 0, 1, 0);
  glViewport(0, 0, (GLsizei)width, (GLsizei)height);
  win_width = width;
  win_height = height;
}

void Window::mouse(int button, int dir, int x, int y)
{
  if (button == GLUT_LEFT_BUTTON)
  {
    intial_mouse_x = x;
    intial_mouse_y = y;
  }
}

void Window::mouseDrag(int x, int y)
{
  float deltaX = float(x - intial_mouse_x);
  float deltaY = float(y - intial_mouse_y);

  ry += (float).01*deltaX;
  rx += (float).01*deltaY;
}

void Window::mouseWheel(int button, int dir, int x, int y)
{
  if (dir > 0)
  {
    dz *= .9f;
  }
  else
  {
    dz /= .9f;
  }
}

void Window::start()
{
  glutMainLoop();
}

void Window::loop()
{
  while (true)
  {
    glutMainLoopEvent();
  }
}

void glwRefreshTimer(int value)
{
  glutPostRedisplay();
  glutTimerFunc(Window::del_time, glwRefreshTimer, 0);
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