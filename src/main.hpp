/**
 * @file      main.hpp
 * @brief     Main file for CUDA rasterizer. Handles CUDA-GL interop for display.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya
 * @date      2012-2015
 * @copyright University of Pennsylvania
 */

#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <util/glslUtility.hpp>
#include <util/utilityCore.hpp>
#include <util/objloader.hpp>
#include "rasterize.h"

using namespace std;

//-------------------------------
//------------GL STUFF-----------
//-------------------------------
int frame;
int fpstracker;
double seconds;
int fps = 0;
GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
const char *attributeLocations[] = { "Position", "Tex" };
GLuint pbo = (GLuint)NULL;
GLuint displayImage;
uchar4 *dptr;

GLFWwindow *window;

//-------------------------------
//----------CUDA STUFF-----------
//-------------------------------

int width = 800;
int height = 800;

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char **argv);

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda();

#ifdef __APPLE__
void display();
#else
void display();
void keyboard(unsigned char key, int x, int y);
#endif

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------
bool init();
void initPBO();
void initCuda();
void initTextures();
void initVAO();
GLuint initShader();

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void cleanupCuda();
void deletePBO(GLuint *pbo);
void deleteTexture(GLuint *tex);

//------------------------------
//-------GLFW CALLBACKS---------
//------------------------------
void mainLoop();
void errorCallback(int error, const char *description);
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);

//Mouse Control
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void mouseMotionCallback(GLFWwindow* window, double xpos, double ypos);
void mouseWheelCallback(GLFWwindow* window, double xoffset, double yoffset);


void updateCamera();
float scale = 1.0f;
float x_trans = 0.0f, y_trans = 0.0f, z_trans = 10.0f;
float x_angle = 0.0f, y_angle = 0.0f;




//Samping test
glm::mat4 model = glm::scale(glm::vec3(scale, scale, scale));
glm::mat4 view;

glm::mat4 projection;
glm::vec3 cameraPosition(x_trans, y_trans, z_trans);


int num_points;
GLuint vertexbuffer;

GLuint vertexBufferObjID[3];
GLuint program;
const char *samplingTest_attributeLocations[] = { "Position" };




bool samplingTest_Init();
void samplingTest_Loop();

void samplingTest_InitVAO();

void samplingTest_InitShaders(GLuint & program);


//program location
GLuint u_modelView;
GLuint u_projMatrix;
GLuint u_screenSize;
GLuint u_spriteSize;

GLuint u_color;
GLuint u_lightDir;