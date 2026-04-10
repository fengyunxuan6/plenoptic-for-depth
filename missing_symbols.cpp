#include <libv/graphic/viewer_context.hpp>

// 定义缺失的符号
bool _ZN6Viewer7_enableE = false;  // Viewer::_enable
void* _ZN6Viewer7_viewerE[2] = {nullptr, nullptr};  // Viewer::_viewer
unsigned int _ZN6Viewer6_layerE[2] = {0, 0};  // Viewer::_layer
unsigned int _ZN6Viewer12_saved_layerE[2] = {0, 0};  // Viewer::_saved_layer

// 提供缺失的符号定义
bool Viewer::_enable = false;  // 默认禁用GUI
v::ViewerContext Viewer::_viewer[2] = {
    v::viewer(0).title("2D view"),
    v::viewer(1).title("3D view")
};

unsigned int Viewer::_layer[2] = {0, 0};
unsigned int Viewer::_saved_layer[2] = {0, 0};