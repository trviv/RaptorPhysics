/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#include "GLWindow.h"
#include "ImageLoader.h"
#include "FontLoader.h"

#define NO_SDL_GLEXT
#include <SDL2/SDL.h>
#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <imgui/imgui_impl_sdl.h>
#include <imgui/imgui_impl_opengl3.h>

int Window::del_time = 5;
Window        *main_window = NULL;
SDL_Window    *sdl_window = NULL;
SDL_GLContext gl_context;

#define WINDOW_MAX_TRANSLATION_RATE 2.f
#define WINDOW_TRANSLATION_RATE     0.10f
#define WINDOW_ROTATION_SCALE       0.005f
#define MOUSE_SENSITIVITY           0.50f
#define MOVE_FRICTION               0.50f
#define RESET_CAMERA_SPEED          0.75f

#define USE_ON_SCREEN_CONTROLLER

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

int SDLCALL watch(void *userdata, SDL_Event* event)
{
  if (event->type == SDL_APP_WILLENTERBACKGROUND)
  {
    quit = true;
  }
  return 1;
}

void setProjectionMatrix(float result[], float aspect)
{
  float fov = 60.f;
  float farDist = 10000.f;
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
void setLookAtMatrix(float view[], const Real3& position, const Real3& target, Real3& up)
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

// function to process yaw and pitch so they remain in a valid range
void constrainYawAndPitch(float& yaw, float& pitch)
{
  yaw   = (yaw >= M_2_PI) ? (yaw - M_2_PI) : (yaw <= -M_2_PI) ? (yaw + M_2_PI) : yaw;
  pitch = (pitch >= M_2_PI) ? (pitch - M_2_PI) : (pitch <= -M_2_PI) ? (pitch + M_2_PI) : pitch;
  pitch = mCrop(pitch, -M_PI_2 + M_PI / 180.f, M_PI_2 - M_PI / 180.f);
}

void Window::init(int argc, char** argv, int width, int height,
                  const char* name)
{
  if (SDL_Init(SDL_INIT_VIDEO|SDL_INIT_JOYSTICK) < 0)
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
  // 24 is to maintain compatiblity with older windows GPU
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE,   24);
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

  int windowFlags = SDL_WINDOW_RESIZABLE|SDL_WINDOW_OPENGL|SDL_WINDOW_ALLOW_HIGHDPI;
#if TARGET_OS_IPHONE
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  windowFlags |= SDL_WINDOW_MAXIMIZED;
#else
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
#endif

  SDL_CheckError();

  // Create an application window with the following settings:
  sdl_window = SDL_CreateWindow(name, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
    win_width, win_height, windowFlags);
  SDL_CheckError();

  gl_context = SDL_GL_CreateContext(sdl_window);
  SDL_CheckError();

  SDL_GL_MakeCurrent(sdl_window, gl_context);
  SDL_CheckError();

#if ENV_WIN
  glewExperimental = true;
  if (glewInit() != GLEW_OK)
  {
    std::cout << "Glew Error..." << std::endl;
  }
#endif

  // Setup GUI
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::StyleColorsDark();

  statFrame = new UIFrame("Stat", 176, 128, 0);
  optionFrame = new UIFrame("Options", 176, 128, UIFrame::Right|UIFrame::FloatX);
  timeSliderFrame = new UIFrame("TimeSlider", 176, 128, UIFrame::Bottom|UIFrame::Right|UIFrame::FloatX);

  timeSliderFrame->setCompactText(UIElement::getIconAsString("fa-solid-900", 0xF04C)+"Pause ");
  timeSliderFrame->setCompactOptionText(UIElement::getIconAsString("fa-solid-900", 0xF04B)+"Resume Sim ");

  uiFrames.push_back(statFrame);
  uiFrames.push_back(optionFrame);
  uiFrames.push_back(timeSliderFrame);

  // Setup Platform/Renderer bindings
  ImGui_ImplSDL2_InitForOpenGL(sdl_window, gl_context);
  #if TARGET_OS_IPHONE
    ImGui_ImplOpenGL3_Init("#version 300 es\n");
  #else
    ImGui_ImplOpenGL3_Init("#version 150\n");
  #endif

  SDL_CheckError();

  down.set(0.f, 0.f, 0.f);

  // enable joystick if found
  if (SDL_NumJoysticks() >= 1)
  {
    SDL_JoystickOpen(0);
  }

  logComputeMessage("OpenGL version: %s\n", glGetString(GL_VERSION));

  cameraUpSpeed = 0.f;
  cameraSideSpeed = 0.f;
  cameraForwardSpeed = 0.f;

  // this initial check fails on windows systems for some reason
  glGetError();

  SDL_GL_GetDrawableSize(sdl_window, &width, &height);
  reshape(width, height);

  clearColor[0] = 0.0f;
  clearColor[1] = 0.0f;
  clearColor[2] = 0.0f;
  clearColor[3] = 1.0f;

  // camera settings
  cameraUp = Real3(0.0f, 1.0f, 0.0f);
  cameraFront = Real3(0.0f, 0.0f, -1.0f);
  cameraPosition = Real3(0.0f, 0.0f, 0.0f);

  yaw   = -M_PI_2;
  pitch = 0.0f;

  initialYaw = yaw;
  initialPitch = pitch;

  float mainFontSize = 14;
  float iconFontSize = 28;

  ImFontConfig fontConfig = ImFontConfig();
  fontConfig.FontDataOwnedByAtlas = false;

  string fontData = IOInterface::readFile("DefaultFont.ttf");
  ImGui::GetIO().Fonts->AddFontFromMemoryTTF((void*)fontData.c_str(), (int)fontData.size(), mainFontSize, &fontConfig);

  fontConfig.MergeMode = true;
  fontConfig.PixelSnapH = true;
  fontConfig.GlyphOffset.y = (iconFontSize - mainFontSize) * 0.5f;
  FontLoader::readFontFile("fa-regular-400", iconFontSize, &fontConfig);
  FontLoader::readFontFile("fa-solid-900", iconFontSize, &fontConfig);
  FontLoader::readFontFile("fa-brands-400", iconFontSize, &fontConfig);

  ImGui::GetIO().Fonts->Build();

  controlWindowHeight = 128.0f;
  controlWindowSidePos = 96.0f;
  controlWindowBottomPos = 192.0f;
  ImageLoader::readImageFile("MoveControl", &moveControlImage, 96, 96);
  ImageLoader::readImageFile("MoveControlBack", &moveControlImageBack, 96, 96);

  uiFrames.cornerPadding[0][0] = 16.f;
  uiFrames.cornerPadding[0][1] = 16.f;
  uiFrames.cornerPadding[1][0] = 16.f;
  uiFrames.cornerPadding[1][1] = 16.f;

  forceRefreshUICount = 0;
}

Window::~Window()
{
  for (int i=0; i<uiFrames.size(); i++)
  {
    delete uiFrames[i];
    uiFrames[i] = NULL;
  }

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplSDL2_Shutdown();
  ImGui::DestroyContext();

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
      cameraForwardSpeed = mCrop(cameraForwardSpeed, -WINDOW_MAX_TRANSLATION_RATE, WINDOW_MAX_TRANSLATION_RATE);
      break;
    case 's':
      cameraForwardSpeed -= WINDOW_TRANSLATION_RATE;
      cameraForwardSpeed = mCrop(cameraForwardSpeed, -WINDOW_MAX_TRANSLATION_RATE, WINDOW_MAX_TRANSLATION_RATE);
      break;
    case 'a':
      cameraSideSpeed -= WINDOW_TRANSLATION_RATE;
      cameraSideSpeed = mCrop(cameraSideSpeed, -WINDOW_MAX_TRANSLATION_RATE, WINDOW_MAX_TRANSLATION_RATE);
      break;
    case 'd':
      cameraSideSpeed += WINDOW_TRANSLATION_RATE;
      cameraSideSpeed = mCrop(cameraSideSpeed, -WINDOW_MAX_TRANSLATION_RATE, WINDOW_MAX_TRANSLATION_RATE);
      break;
    case 'q':
      cameraUpSpeed += WINDOW_TRANSLATION_RATE;
      cameraUpSpeed = mCrop(cameraUpSpeed, -WINDOW_MAX_TRANSLATION_RATE, WINDOW_MAX_TRANSLATION_RATE);
      break;
    case 'e':
      cameraUpSpeed -= WINDOW_TRANSLATION_RATE;
      cameraUpSpeed = mCrop(cameraUpSpeed, -WINDOW_MAX_TRANSLATION_RATE, WINDOW_MAX_TRANSLATION_RATE);
      break;
    case SDLK_ESCAPE:
      quit = true;
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
  intial_mouse_x = x;
  intial_mouse_y = y;
}

void Window::mouseDrag(int x, int y)
{
  float deltaX = float(x - intial_mouse_x);
  float deltaY = float(intial_mouse_y - y);

  intial_mouse_x = x;
  intial_mouse_y = y;

  deltaX *= MOUSE_SENSITIVITY;
  deltaY *= MOUSE_SENSITIVITY;

  // update angles
  yaw   += deltaX * M_2_PI / width();
  pitch += deltaY * M_2_PI / height();

  constrainYawAndPitch(yaw, pitch);

  // update camera look at
  cameraFront().x = cos(yaw) * cos(pitch);
  cameraFront().y = sin(pitch);
  cameraFront().z = sin(yaw) * cos(pitch);
  cameraFront().normalize();

//  printf("Yaw: %f, Pitch: %f\n", yaw, pitch);
//  printf("Front: %f %f %f\n", cameraFront.x, cameraFront.y, cameraFront.z);
}

void Window::mouseWheel(int button, int dir, int x, int y)
{
}

void Window::scroll(float x, float y)
{
  float dx = x - scroll_prev_x;
  float dy = y - scroll_prev_y;

  scroll_prev_x = x;
  scroll_prev_y = y;

  dx *= MOVE_FRICTION;
  dy *= MOVE_FRICTION;

  if(abs(dx) < 0.001f) dx = 0.f;
  if(abs(dy) < 0.001f) dy = 0.f;

  cameraSideSpeed = -dx * 100.f;
  cameraUpSpeed = dy * 100.f;
}

void Window::pinch(float d)
{
  cameraForwardSpeed += d;
  cameraForwardSpeed = mCrop(cameraForwardSpeed, -WINDOW_MAX_TRANSLATION_RATE, WINDOW_MAX_TRANSLATION_RATE);
}

Real3 initialLeft, initialRight;
SDL_FingerID leftFingerId, rightFingerId;

void Window::processOnScreenController(void* eventData, bool end)
{
  const SDL_TouchFingerEvent fingerData = *(SDL_TouchFingerEvent*)eventData;

  // end will be set for finger up event
  if (end)
  {
    // mark left or right finger up
    if (fingerData.fingerId == leftFingerId)
    {
//      logComputeMessage("Left Up");
      leftFingerId = -1;
    }
    else
    if (fingerData.fingerId == rightFingerId)
    {
//      logComputeMessage("Right Up");
      rightFingerId = -1;

      initialYaw = yaw;
      initialPitch = pitch;
    }
    return;
  }

  // if finger not recognized
  if (fingerData.fingerId != leftFingerId && fingerData.fingerId != rightFingerId)
  {
    // mark it as right or left based on location
    if (fingerData.x <= 0.5f)
    {
//      logComputeMessage("Left Down");
      initialLeft = Real3(fingerData.x, fingerData.y, 0.f);
      leftFingerId = fingerData.fingerId;
    }
    else
    {
//      logComputeMessage("Right Down");
      initialRight = Real3(fingerData.x, fingerData.y, 0.f);
      rightFingerId = fingerData.fingerId;

      initialYaw = yaw;
      initialPitch = pitch;
    }
  }

  // process finger if recognized
  if (fingerData.fingerId == leftFingerId)
  {
//    logComputeMessage("Left Move");
    cameraSideSpeed = (fingerData.x - initialLeft.x);
    cameraSideSpeed = mCrop(cameraSideSpeed, -WINDOW_MAX_TRANSLATION_RATE, WINDOW_MAX_TRANSLATION_RATE);
    cameraForwardSpeed = (initialLeft.y - fingerData.y);
    cameraForwardSpeed = mCrop(cameraForwardSpeed, -WINDOW_MAX_TRANSLATION_RATE, WINDOW_MAX_TRANSLATION_RATE);
  }
  else if (fingerData.fingerId == rightFingerId)
  {
//    logComputeMessage("Right Move");
    float deltaX = (fingerData.x - initialRight.x) * MOUSE_SENSITIVITY;
    float deltaY = (initialRight.y - fingerData.y) * MOUSE_SENSITIVITY;

    // update angles
    yaw   = initialYaw + deltaX * 2.0f * M_2_PI;
    pitch = initialPitch + deltaY * 2.0f * M_2_PI;

    constrainYawAndPitch(yaw, pitch);

    // update camera look at
    cameraFront().x = cos(yaw) * cos(pitch);
    cameraFront().y = sin(pitch);
    cameraFront().z = sin(yaw) * cos(pitch);
    cameraFront().normalize();
  }
}

void Window::start()
{
  SDL_CheckError();

  setProjectionMatrix(projectionMatrix, (float)width()/(float)height());

  bool mouseDown = false;
  bool gesture = false;
  bool scrolling = false;

  SDL_FingerID fingerId;
  Real3 eyeVector(0.f);

  leftFingerId = -1;
  rightFingerId = -1;

  while (!quit)
  {
    const uint frameStartTime = SDL_GetTicks();

    if(abs(cameraUpSpeed)*height() < 1.f)       cameraUpSpeed = 0.f;
    if(abs(cameraSideSpeed)*width() < 1.f)      cameraSideSpeed = 0.f;
    if(abs(cameraForwardSpeed)*height() < 1.f)  cameraForwardSpeed = 0.f;

    // update camera settings
    Real3 cross = cameraFront().cross(cameraUp);
    cross.normalize();
    cameraPosition += cameraFront * cameraForwardSpeed + cross * cameraSideSpeed + cameraUp * cameraUpSpeed;

    setLookAtMatrix(modelMatrix, cameraPosition, cameraPosition + cameraFront, cameraUp);

    cameraUpSpeed *= MOVE_FRICTION;
    cameraSideSpeed *= MOVE_FRICTION;
    cameraForwardSpeed *= MOVE_FRICTION;

#ifndef USE_ON_SCREEN_CONTROLLER
    gesture = SDL_HasEvent(SDL_MULTIGESTURE);

    if (!gesture)
    {
      scrolling = false;
    }
#endif

    Real3 downVector(0.f, 0.f, 0.f);

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL2_NewFrame(sdl_window);

    // handle events
    SDL_Event event;
    while (SDL_PollEvent(&event))
    {
      // process before io processing so it works even after loosing focus
      if (event.type == SDL_JOYAXISMOTION)
      {
        if (event.jaxis.axis == 0)
        {
          downVector[1] = event.jaxis.value;
        }
        else if (event.jaxis.axis == 1)
        {
          downVector[0] = event.jaxis.value;
        }
        else if (event.jaxis.axis == 2)
        {
          downVector[2] = event.jaxis.value;
        }
        continue;
      }

      ImGui_ImplSDL2_ProcessEvent(&event);
      if (ImGui::GetIO().WantCaptureMouse || ImGui::GetIO().WantCaptureKeyboard)
      {
        // do not track once in the GUI space
        fingerId = -1;
        mouseDown = false;
        continue;
      }

      // Frame list option can be toggled using keys 1 - frameOptionList.size()
      if (event.type == SDL_KEYDOWN && event.key.keysym.sym >= SDLK_1 && event.key.keysym.sym < (SDLK_1+optionFrame->getElements().size()))
      {
        ((UIElement*)optionFrame->getElements()[event.key.keysym.sym - SDLK_1])->boolValue = !((UIElement*)optionFrame->getElements()[event.key.keysym.sym - SDLK_1])->boolValue;
      }

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

#ifndef USE_ON_SCREEN_CONTROLLER
        case SDL_MULTIGESTURE:
        {
          switch(event.mgesture.numFingers)
          {
            case 2:
              // scroll x, y axis for 2 fingers
              if (scrolling)
              {
                scroll(event.mgesture.x, event.mgesture.y);
              }
              scroll_prev_x = event.mgesture.x;
              scroll_prev_y = event.mgesture.y;
              break;
            case 3:
            case 4:
              // scroll x axis for 3 and 4 fingers
              if (scrolling)
              {
                pinch((event.mgesture.y - scroll_prev_z) * 10.f);
              }
              scroll_prev_z = event.mgesture.y;
              break;
            default:
              break;
          }
          scrolling = true;
          fingerId = -1;
          mouseDown = false;
        }
          break;
#endif

        case SDL_MOUSEWHEEL:
        {
          pinch(event.wheel.y * 0.0075f);
          mouseDown = false;
        }
          break;

        case SDL_FINGERDOWN:
        {
#ifdef USE_ON_SCREEN_CONTROLLER
          processOnScreenController(&event.tfinger, false);
#else
          if (!mouseDown)
          {
            fingerId = event.tfinger.fingerId;
            mouse(0, 0, event.tfinger.x*width(), event.tfinger.y*height());
            mouseDown = true;
          }
#endif
        }
          break;

#ifndef USE_ON_SCREEN_CONTROLLER
        case SDL_MOUSEBUTTONDOWN:
        {
          if (!mouseDown)
          {
            mouse(0, 0, event.motion.x, event.motion.y);
            mouseDown = true;
          }
        }
          break;
#endif

        case SDL_FINGERUP:
        {
#ifdef USE_ON_SCREEN_CONTROLLER
          processOnScreenController(&event.tfinger, true);
#else
          mouseDown = false;
#endif
        }
          break;

        case SDL_MOUSEBUTTONUP:
        {
          mouseDown = false;
        }
          break;

        case SDL_FINGERMOTION:
        {
#ifdef USE_ON_SCREEN_CONTROLLER
          processOnScreenController(&event.tfinger, false);
#else
          if (mouseDown && !gesture && fingerId == event.tfinger.fingerId)
          {
            mouseDrag(event.tfinger.x*width(), event.tfinger.y*height());
          }
#endif
        }
          break;

#ifndef USE_ON_SCREEN_CONTROLLER
        case SDL_MOUSEMOTION:
        {
          if (mouseDown)
          {
            mouseDrag(event.motion.x, event.motion.y);
          }
        }
          break;
#endif

        case SDL_KEYDOWN:
        {
          this->keyboard(event.key.keysym.sym, 0, 0);
        }
          break;

        default:
          break;
      }
    }

    // set down only if down vector recorded
    if (downVector.length() > 0.f)
    {
      downVector *= 5.f / 32767.f;
      downVector[2] += 0.64f;
      down = downVector;
    }

    SDL_Init (SDL_INIT_EVENTS);

    ImGui::NewFrame();

    ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize;

    char temp[32] = {NULL};
    if (statFrame->isShrunk())
    {
      sprintf(temp, "%.f", ImGui::GetIO().Framerate);
      statFrame->setText(temp);
    }
    else
    {
      if (ImStrnicmp("Frame Rate: ", statFrame->getText().c_str(), sizeof("Frame Rate:")) != 0)
      {
        sprintf(temp, "Frame Rate:  %.f\n", ImGui::GetIO().Framerate);
        statFrame->setText(temp + statFrame->getText());
      }
    }

    if (uiFrames.render())
    {
      forceRefreshUICount = 2;
    }

    // remove right finger data when this window in focus
    if (ImGui::IsWindowFocused())
    {
      rightFingerId = -1;
    }

    // add buttons
    if (ImGui::GetCurrentContext()->LastActiveId == optionFrame->getElement("Reset Camera")->getUIID())
    {
      const float animationTime = ImSaturate(ImGui::GetCurrentContext()->LastActiveIdTimer * RESET_CAMERA_SPEED);
      if (animationTime == 0.f)
      {
        cameraUp.begin() = cameraUp;
        cameraFront.begin() = cameraFront;
        cameraPosition.begin() = cameraPosition;

        cameraUp().normalize();
        cameraFront().normalize();
      }
      if (animationTime < 1.f)
      {
        cameraUp.interpolate(animationTime);
        cameraFront.interpolate(animationTime);

        cameraUp().normalize();
        cameraFront().normalize();

        float distance = mSqrt(cameraFront().z * cameraFront().z + cameraFront().x * cameraFront().x);
        yaw = M_PI + atan2(cameraFront().x, cameraFront().z);
        pitch = asin(cameraFront().y / distance);

        // TODO: Check why this offsetting is needed
        pitch += M_PI/18.f;
        constrainYawAndPitch(yaw, pitch);

        cameraPosition.interpolate(animationTime);

        initialYaw = yaw;
        initialPitch = pitch;
      }
    }

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0.0f, 0.0f});
    ImGui::Begin("Control", NULL, ((windowFlags ^ ImGuiWindowFlags_AlwaysAutoResize) |
                                   ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoInputs));

    ImGui::SetWindowSize({ImGui::GetIO().DisplaySize.x, controlWindowBottomPos});
    ImGui::SetWindowPos({0.0f, ImGui::GetIO().DisplaySize.y - controlWindowBottomPos});

    ImVec2 moveControlPositionBack = {controlWindowSidePos, (controlWindowBottomPos - controlWindowHeight) * 0.5f};

    ImGui::SetCursorPos(moveControlPositionBack);
    ImGui::Image((void*)(intptr_t)moveControlImageBack.get(), {(float)moveControlImageBack.width(), (float)moveControlImageBack.height()});

    ImVec2 moveControlPosition = moveControlPositionBack;
    moveControlPosition.x += moveControlImageBack.width() * (0.25f + cameraSideSpeed/WINDOW_MAX_TRANSLATION_RATE);
    moveControlPosition.y += moveControlImageBack.height() * (0.25f - cameraForwardSpeed/WINDOW_MAX_TRANSLATION_RATE);

    ImGui::SetCursorPos(moveControlPosition);
    ImGui::Image((void*)(intptr_t)moveControlImage.get(), {moveControlImage.width() * 0.5f, moveControlImage.height() * 0.5f});

    ImGui::End();
    ImGui::PopStyleVar(ImGuiStyleVar_WindowPadding);

    ImGui::Render();

    // main work
    display();

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    SDL_GL_SwapWindow(sdl_window);
    SDL_CheckError();

    // limit the frame rate
    const uint frameEndTime = SDL_GetTicks();
    if ((frameEndTime - frameStartTime) < (1000.f / 60.f))
    {
      SDL_Delay((1000.f / 60.f) - (frameEndTime - frameStartTime));
    }
  }
}

ComputeGraphicsSharedTexture Window::createSharedTexture(ComputeInterface* compute, uint textureSize[2], SharedTextureFormat textureFormat)
{
#if ENV_APPLE
  return ComputeGraphicsSharedTexture(compute, (__bridge GLContext*)SDL_GL_GetCurrentContext(), textureFormat, textureSize);
#else
  return ComputeGraphicsSharedTexture(compute, (GLContext*)SDL_GL_GetCurrentContext(), textureFormat, textureSize);
#endif
}
