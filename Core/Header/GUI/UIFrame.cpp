#include "UIFrame.h"
#include <algorithm>

UIFrame::UIFrame(const string &name, float width, float height, float posX, float posY)
  : shrink(false), name(name)
{
  pos[0] = posX;
  pos[1] = posY;
  size[0] = width;
  size[1] = height;
  alignment = 0;
}

UIFrame::UIFrame(const string &name, float width, float height, uint alignment)
  : shrink(false), name(name)
{
  pos[0] = 0;
  pos[1] = 0;
  size[0] = width;
  size[1] = height;
  this->alignment = alignment;
}

UIFrame::~UIFrame()
{
}

void UIFrame::addElement(const UIElement& option)
{
  uiElements.push_back(option);
  uiElementMap[option.identifier] = (uint)uiElements.size() - 1;
}

UIElement& UIFrame::getElement(const string& name)
{
  return uiElements[uiElementMap[name]];
}

void UIFrame::render()
{
  ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize;

  ImGui::Begin(name.c_str(), NULL, windowFlags);
  ImGui::SetWindowSize({size[0], size[1]});

  if (ImGui::IsWindowFocused())
  {
    shrink = !shrink;
  }

  ImGui::Text("%s", text.c_str());

  size[0] = ImGui::GetWindowSize().x;
  size[1] = ImGui::GetWindowSize().y;

  ImGui::SetWindowPos({pos[0], pos[1]});

  ImGui::End();
}
