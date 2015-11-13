#version 330

in vec2 Position;

void main() {
    gl_Position.xy = Position;
    gl_Position.z = 0.0;
    gl_Position.w = 1.0;
}
