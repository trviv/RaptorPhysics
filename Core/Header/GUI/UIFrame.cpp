#include "UIFrame.h"
#include <algorithm>

UIFrame::UIFrame(const string &name, float width, float height, float posX, float posY)
  : shrink(false), name(name)
{
  pos[0] = posX;
  pos[1] = posY;
  size[0] = width;
  size[1] = height;
  fixed = true;
}

UIFrame::UIFrame(const string &name, float width, float height, uint alignment)
  : shrink(false), name(name)
{
  size[0] = width;
  size[1] = height;
  this->alignment = alignment;
}

UIFrame::~UIFrame()
{
}

void UIFrame::addElement(UIElement* object)
{
  uiElements.push_back(object);
  uiElementMap[object->identifier] = (uint)uiElements.size() - 1;
}

UIElement* UIFrame::getElement(const string& name)
{
  return (UIElement*)uiElements[uiElementMap[name]];
}

bool UIFrame::render()
{
  bool changed = false;
  ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize;

  ImGui::Begin(name.c_str(), NULL, windowFlags);
  ImGui::SetWindowSize({size[0], size[1]});

  if (ImGui::IsWindowFocused())
  {
    shrink = !shrink;
    changed = true;
  }

  ImGui::Text("%s", text.c_str());

  size[0] = ImGui::GetWindowSize().x;
  size[1] = ImGui::GetWindowSize().y;

  ImGui::SetWindowPos({pos[0], pos[1]});

  // do not display sub elements if shrunk
  if (!shrink)
  {
    for (auto i : uiElements)
    {
      changed |= i->render();
    }
  }

  ImGui::End();
  return changed;
}
