/**
 * @file      main.cpp
 * @brief     Main file for CUDA rendering. Handles CUDA-GL interop for display.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya
 * @date      2012-2015
 * @copyright University of Pennsylvania
 */

#include "main.hpp"
#include <chrono>

#include "RigidBody.h"
#include "particleSampling.h"
#include "simulate.h"

#define GRID_LENGTH_DEFAULT (0.273438f)

//#define OBJ_ARR_SIZE 9
#define OBJ_ARR_SIZE 3

#define GRAVITY (glm::vec3(0.0f,-9.8f,0.0f))
#define FPS 60.0f
#define FRAME_TIME 1.0f/FPS
#define SIMU_STEP 8
#define DELTA_T FRAME_TIME/SIMU_STEP

//-------------------------------
//-------------MAIN--------------
//-------------------------------

RigidBody rigid_body[OBJ_ARR_SIZE];
float uniform_grid_length;
GLfloat *v_buffer_ptr;
int buffer_size;



int main(int argc, char **argv) {
    if (argc < 2) {
        cout << "Usage: [obj file]" << endl;
        return 0;
    }

	//GL init
	frame = 0;
	seconds = time(NULL);
	fpstracker = 0;

	if (samplingTest_Init()) {

		//// Rigid body Suzanne
		//rigid_body[0].setPhase(0);
		//rigid_body[0].setTranslate(glm::vec3(0.0f, 0.0f, 0.0f));
		//rigid_body[0].setRotation(glm::rotate(5.0f*(float)PI/180.0f, glm::vec3(0.0f, 0.0f, 1.0f)));
		//rigid_body[0].setInitVelocity(glm::vec3(0.0f, 0.0f, 0.0f));
		////rigid_body[0].setInitVelocity(glm::vec3(0.0f, 9.8f, 0.0f));
		//rigid_body[0].setMassScale(1.0f);
		//rigid_body[0].setType(SOLID);
		//rigid_body[0].setColor(glm::vec4(0.0f, 0.5f, 1.0f, 1.0f));
		//rigid_body[0].initObj(argv[1]);
		//rigid_body[0].initParticles(10);

		int c = 0;
		uniform_grid_length = GRID_LENGTH_DEFAULT;
		// water fill
		c = 0;
		rigid_body[c].setPhase(c);
		//rigid_body[c].setScale(glm::vec3(2.5f, 2.5f, 2.5f));
		rigid_body[c].setTranslate(glm::vec3(-7.0f, 8.0f, 0.0f));
		//rigid_body[c].setTranslate(glm::vec3(1.0f, 5.0f, 0.0f));
		//rigid_body[c].setRotation(glm::rotate(5.0f*(float)PI / 180.0f, glm::vec3(0.0f, 0.0f, 1.0f)));
		rigid_body[c].setInitVelocity(glm::vec3(0.0f, 0.0f, 0.0f));
		//rigid_body[c].setInitVelocity(glm::vec3(0.0f, 9.8f, 0.0f));
		rigid_body[c].setMassScale(1.0f);
		rigid_body[c].setType(SOLID);
		rigid_body[c].setColor(glm::vec4(1.0f, 0.0f, 1.0f, 1.0f));
		rigid_body[c].initObj(argv[1]);
		rigid_body[c].initParticles(uniform_grid_length);

		uniform_grid_length = rigid_body[c].getGridLength();
		std::cout << uniform_grid_length;

		//for (int i = 1; i <= OBJ_ARR_SIZE - 2; i++)
		//{
		//	c = i;
		//	rigid_body[c] = rigid_body[0];
		//	rigid_body[c].setPhase(c);
		//	rigid_body[c].setTranslate(glm::vec3(4.0f - (float)(c % 4) * 2.5f, 1.0f + (float)(c / 4)* 3.0f, 4.0f - (float)(c % 4) * 2.5f));
		//	rigid_body[c].setRotation(glm::rotate(5.0f* (float)c*(float)PI / 180.0f, glm::vec3(0.0f, 0.0f, 1.0f)));
		//	rigid_body[c].setInitVelocity(glm::vec3(0.0f, 0.0f, 0.0f));
		//	rigid_body[c].setMassScale(1.0f);
		//	rigid_body[c].setColor(COLOR_PRESET[(i - 1) % NUM_COLOR_PRESET]);
		//	rigid_body[c].initParticles(uniform_grid_length);
		//}


		////water test
		//c = 1;
		//rigid_body[c].setPhase(c);
		//rigid_body[c].setScale(glm::vec3(2.5f,2.5f,2.5f));
		////rigid_body[c].setScale(glm::vec3(1.5f, 1.5f, 1.5f));
		//rigid_body[c].setTranslate(glm::vec3(1.0f, 1.0f, 1.0f));
		////rigid_body[c].setRotation(glm::rotate(5.0f*(float)PI / 180.0f, glm::vec3(0.0f, 0.0f, 1.0f)));
		//rigid_body[c].setInitVelocity(glm::vec3(0.0f, 0.0f, 0.0f));
		////rigid_body[c].setInitVelocity(glm::vec3(0.0f, 9.8f, 0.0f));
		//rigid_body[c].setMassScale(1.0f);
		//rigid_body[c].setType(FLUID);
		//rigid_body[c].setColor(glm::vec4(0.0f, 0.5f, 1.0f, 1.0f));
		//rigid_body[c].initObj("../objs/cube.obj");
		//rigid_body[c].initParticles(uniform_grid_length);


		//water fill
		c = 1;
		rigid_body[c].setPhase(c);
		//rigid_body[c].setScale(glm::vec3(2.5f, 2.5f, 2.5f));
		rigid_body[c].setTranslate(glm::vec3(0.0f, 0.0f, 0.0f));
		//rigid_body[c].setRotation(glm::rotate(5.0f*(float)PI / 180.0f, glm::vec3(0.0f, 0.0f, 1.0f)));
		rigid_body[c].setInitVelocity(glm::vec3(0.0f, 0.0f, 0.0f));
		//rigid_body[c].setInitVelocity(glm::vec3(0.0f, 9.8f, 0.0f));
		rigid_body[c].setMassScale(1.0f);
		rigid_body[c].setType(FLUID);
		rigid_body[c].setColor(glm::vec4(0.0f, 0.5f, 1.0f, 1.0f));
		rigid_body[c].initObj("../objs/fill.obj");
		rigid_body[c].initParticles(uniform_grid_length);



		if (argc == 3){
			c = OBJ_ARR_SIZE - 1;
			rigid_body[c].setPhase(c);
			rigid_body[c].setTranslate(glm::vec3(0.0f, -3.0f, 0.0f));
			rigid_body[c].setRotation(glm::rotate(180.0f*(float)PI / 180.0f, glm::vec3(0.0f, 1.0f, 0.0f)) *
				glm::rotate(90.0f*(float)PI / 180.0f, glm::vec3(1.0f, 0.0f, 0.0f)));
			rigid_body[c].setInitVelocity(glm::vec3(0.0f, 0.0f, 0.0f));
			rigid_body[c].setMassScale(0.0f);
			rigid_body[c].setType(SOLID);
			rigid_body[c].setColor(glm::vec4(1.0f, 0.5f, 0.0f, 0.1f));
			rigid_body[c].initObj(argv[2]);
			rigid_body[c].initParticles(uniform_grid_length);
		}


		

		/*

		float t = 0;
		for (int i = 1; i < OBJ_ARR_SIZE; i++)
		{
			float s = 1.0 - 0.2 * (float)i;
			//float t = -2.3 * (float)i;
			t += -3.5 * s - 0.2;
			rigid_body[i].setScale(glm::vec3(s, s, s));
			rigid_body[i].setTranslate(glm::vec3(t, 0.0f, 0.0f));
			rigid_body[i].initObj(argv[1]);
			rigid_body[i].initParticles(uniform_grid_length);
		}
		*/
		
		initSimulation();
		samplingTest_InitVAO();
		samplingTest_InitShaders(program);

		// GLFW main loop
		//mainLoop();
		
		samplingTest_Loop();



	}

	

	
	return 0;
}




void samplingTest_Loop()
{

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		
		time_t seconds2 = time(NULL);

		frame++;
		fpstracker++;

		if (seconds2 - seconds >= 1) {

			fps = fpstracker / (seconds2 - seconds);
			fpstracker = 0;
			seconds = seconds2;
		}

		string title = "CIS565 Final | " + utilityCore::convertIntToString((int)fps) + " FPS";
		glfwSetWindowTitle(window, title.c_str());


		//update camera
		updateCamera();
		

		
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		//glClear(GL_COLOR_BUFFER_BIT);

		//std::cout << glewGetErrorString(glGetError()) << '\n';
		
		
		glUseProgram(program);
		//float tmp[16] = { 1.0, 0.0, 0.0, 0.0,
		//	0.0, 1.0, 0.0, 0.0,
		//	0.0, 0.0, 1.0, 0.0,
		//	0.0, 0.0, 0.0, 1.0 };

		glm::vec2 screenSize(width, height);
		glm::mat4 modelView = view * model;
		glUniformMatrix4fv(u_modelView, 1, GL_FALSE, &modelView[0][0]);
		glUniformMatrix4fv(u_projMatrix, 1, GL_FALSE, &projection[0][0]);
		glUniform2fv(u_screenSize, 1, &screenSize[0]);
		glUniform1f(u_spriteSize, uniform_grid_length);		//radius tmp

		glUniform3f(u_color, 1.0, 1.0, 0.0);
		glUniform3f(u_lightDir, 0.0, 0.0, 1.0);


		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);

		
		
		//update buffer data
		
		// Get buffer pointer for animation
		v_buffer_ptr = (GLfloat*)glMapBufferRange(GL_ARRAY_BUFFER, 0, buffer_size * sizeof(GLfloat), GL_MAP_WRITE_BIT | GL_MAP_READ_BIT);

		// Do simulation & animations
		//v_buffer_ptr[0] += 0.01f;

		// Granulate simulation steps to compensate frame rate cap
		// 0.0167 is 60 FPS
		// More steps can reduce tunneling effect (due to high velocity and thin object)
		for (int i = 0; i < SIMU_STEP; i++){
			simulate(GRAVITY, DELTA_T, v_buffer_ptr,rigid_body);
		}

		// Unmap the buffer pointer so that openGL will start rendering
		glUnmapBuffer(GL_ARRAY_BUFFER);

		///////////////////////////


		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

		
		//glPointSize(30);
		
		glDrawArrays(GL_POINTS, 0, num_points);
		glDisableVertexAttribArray(0);
		
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer_color);
		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
		

		glUseProgram(0);

		glfwSwapBuffers(window);
		
	}
	glfwDestroyWindow(window);
	glfwTerminate();

	endSimulation();
}



//sampling Test init

bool samplingTest_Init()
{
	//GL window
	glfwSetErrorCallback(errorCallback);

	if (!glfwInit()) {
		return false;
	}

	width = 800;
	height = 800;

	
	
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	
	window = glfwCreateWindow(width, height, "Particles Simulation", NULL, NULL);
	if (!window) {
		fprintf(stderr, "Failed to open GLFW window.\n");
		glfwTerminate();
		return false;
	}
	glfwMakeContextCurrent(window);		//Initialize GLEW
	glfwSetKeyCallback(window, keyCallback);
	//MY Mouse Control
	glfwSetMouseButtonCallback(window, mouseButtonCallback);
	glfwSetCursorPosCallback(window, mouseMotionCallback);
	glfwSetScrollCallback(window, mouseWheelCallback);

	// Set up GL context
	glewExperimental = GL_TRUE;			//Needed in core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		return false;
	}


	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	return true;
}

void samplingTest_InitVAO()
{
	// Initialize buffer data
	GLfloat *g_vertex_buffer_data;

	GLfloat *g_vertex_buffer_color;
	

	//init opengl vertex buffer
	vector<float> all_particles;
	vector<float> particles_color;

	for (int i = 0; i < OBJ_ARR_SIZE; i++){
		all_particles.insert(all_particles.end(), rigid_body[i].m_particle_pos.begin(), rigid_body[i].m_particle_pos.end());

		//particles_color.insert(particles_color.end(), rigid_body[i].m_particle_pos.size(), 1.0f);
		int size = rigid_body[i].m_particle_pos.size() / 3;
		for (int j = 0; j < size; j++)
		{
			//int p = rigid_body[i].getPhase() % NUM_COLOR_PRESET;
			//particles_color.insert(particles_color.end(), COLOR_PRESET + 4 * p, COLOR_PRESET + 4 * p + 4);
			particles_color.insert(particles_color.end(), rigid_body[i].m_color, rigid_body[i].m_color + 4);
		}
	}

	g_vertex_buffer_data = (GLfloat*)malloc(all_particles.size() * sizeof(GLfloat));
	std::copy(all_particles.begin(), all_particles.end(), g_vertex_buffer_data);

	g_vertex_buffer_color = (GLfloat*)malloc(particles_color.size() * sizeof(GLfloat));
	std::copy(particles_color.begin(), particles_color.end(), g_vertex_buffer_color);
	
	GLuint VertexArrayID[2];
	glGenVertexArrays(2, VertexArrayID);
	glBindVertexArray(VertexArrayID[0]);
	glBindVertexArray(VertexArrayID[1]);

	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, all_particles.size()*sizeof(GLfloat), g_vertex_buffer_data, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &vertexbuffer_color);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer_color);
	glBufferData(GL_ARRAY_BUFFER, particles_color.size()*sizeof(GLfloat), g_vertex_buffer_color, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	num_points = all_particles.size() / 3;
	buffer_size = all_particles.size();
}


void samplingTest_InitShaders(GLuint & program) {
	GLint location;

	program = glslUtility::createProgram(
		"../shaders/particle.vert.glsl",
		"../shaders/particle.frag.glsl",
		samplingTest_attributeLocations, 1);
	glUseProgram(program);

	//if ((location = glGetUniformLocation(program, "u_projMatrix")) != -1) {
	//	glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
	//}
	//if ((location = glGetUniformLocation(program, "u_cameraPos")) != -1) {
	//	glUniform3fv(location, 1, &cameraPosition[0]);
	//}

	glEnable(GL_PROGRAM_POINT_SIZE);
	glEnable(GL_DEPTH_TEST);

	//alpha
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	
	//vertex shader
	u_modelView = glGetUniformLocation(program, "u_modelView");
	u_projMatrix = glGetUniformLocation(program, "u_projMatrix");
	u_screenSize = glGetUniformLocation(program, "u_screenSize");
	u_spriteSize = glGetUniformLocation(program, "u_spriteSize");


	//fragment shader
	u_color = glGetUniformLocation(program, "u_color");
	u_lightDir = glGetUniformLocation(program, "u_lightDir");
}



//------siumulate---------
const glm::vec3 AREA_BOX_MIN(-10.0f, -10.0f, -10.0f);
const glm::vec3 AREA_BOX_MAX(10.0f, 10.0f, 10.0f);

void initSimulation()
{
	//cuda particle init simulate
	assembleParticleArray(OBJ_ARR_SIZE, rigid_body);

	initUniformGrid(AREA_BOX_MIN, AREA_BOX_MAX, uniform_grid_length);
}


///////////////////////////////////////////////////////















//----------------------------------------------
//---------OLD CUDA rasterizer pipeline---------
//----------------------------------------------


//old rasterizer main loop
//render a 2d rect, texture is given by cuda
void mainLoop() {
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        runCuda();

        time_t seconds2 = time (NULL);

        if (seconds2 - seconds >= 1) {

            fps = fpstracker / (seconds2 - seconds);
            fpstracker = 0;
            seconds = seconds2;
        }

        string title = "CIS565 Final | " + utilityCore::convertIntToString((int)fps) + " FPS";
        glfwSetWindowTitle(window, title.c_str());

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, displayImage);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glClear(GL_COLOR_BUFFER_BIT);

        // VAO, shader program, and texture already bound
        glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);
        glfwSwapBuffers(window);
    }
    glfwDestroyWindow(window);
    glfwTerminate();
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda() {
    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
    dptr = NULL;

    cudaGLMapBufferObject((void **)&dptr, pbo);
	//rasterize(dptr);
    cudaGLUnmapBufferObject(pbo);

    frame++;
    fpstracker++;

}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

bool init() {

	//GL window
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit()) {
        return false;
    }

    width = 800;
    height = 800;
    window = glfwCreateWindow(width, height, "Particles Simulation", NULL, NULL);
    if (!window) {
		fprintf(stderr, "Failed to open GLFW window.\n");
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);		//Initialize GLEW
	glfwSetKeyCallback(window, keyCallback);

    // Set up GL context
    glewExperimental = GL_TRUE;			//Needed in core profile
    if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
        return false;
    }


	

    initVAO();
    initTextures();
    initCuda();
    initPBO();

    

    GLuint passthroughProgram;
    passthroughProgram = initShader();

    glUseProgram(passthroughProgram);
    glActiveTexture(GL_TEXTURE0);

    return true;
}

void initPBO() {
    // set up vertex data parameter
    int num_texels = width * height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;

    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1, &pbo);

    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject(pbo);

}

void initCuda() {
    // Use device with highest Gflops/s
    cudaGLSetGLDevice(0);

	//rasterizeInit(width, height);

    // Clean up on program exit
    atexit(cleanupCuda);
}

void initTextures() {
    glGenTextures(1, &displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA,
                  GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void) {
    GLfloat vertices[] = {
        -1.0f, -1.0f,
        1.0f, -1.0f,
        1.0f,  1.0f,
        -1.0f,  1.0f,
    };

    GLfloat texcoords[] = {
        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f
    };

    GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}


GLuint initShader() {
    const char *attribLocations[] = { "Position", "Tex" };
    GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
    GLint location;

    //glUseProgram(program);
    if ((location = glGetUniformLocation(program, "u_image")) != -1) {
        glUniform1i(location, 0);
    }

    return program;
}

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void cleanupCuda() {
    if (pbo) {
        deletePBO(&pbo);
    }
    if (displayImage) {
        deleteTexture(&displayImage);
    }
}

void deletePBO(GLuint *pbo) {
    if (pbo) {
        // unregister this buffer object with CUDA
        cudaGLUnregisterBufferObject(*pbo);

        glBindBuffer(GL_ARRAY_BUFFER, *pbo);
        glDeleteBuffers(1, pbo);

        *pbo = (GLuint)NULL;
    }
}

void deleteTexture(GLuint *tex) {
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}

void shut_down(int return_code) {
	//rasterizeFree();
	samplingFree();	//TODO:test

    cudaDeviceReset();
#ifdef __APPLE__
    glfwTerminate();
#endif
    exit(return_code);
}

//------------------------------
//-------GLFW CALLBACKS---------
//------------------------------

void errorCallback(int error, const char *description) {
    fputs(description, stderr);
}

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
	if (action == GLFW_PRESS) {
		switch (key) {
		case GLFW_KEY_ESCAPE:
			glfwSetWindowShouldClose(window, GL_TRUE);
			break;
		}
	}
}





enum ControlState { NONE = 0, ROTATE, TRANSLATE };
ControlState mouseState = NONE;
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
	if (action == GLFW_PRESS)
	{
		if (button == GLFW_MOUSE_BUTTON_LEFT)
		{
			mouseState = ROTATE;
		}
		else if (button == GLFW_MOUSE_BUTTON_RIGHT)
		{
			mouseState = TRANSLATE;
		}

	}
	else if (action == GLFW_RELEASE)
	{
		mouseState = NONE;
	}
	//printf("%d\n", mouseState);
}

double lastx = (double)width / 2;
double lasty = (double)height / 2;
void mouseMotionCallback(GLFWwindow* window, double xpos, double ypos)
{
	const double s_r = 0.01;
	const double s_t = 0.01;

	double diffx = xpos - lastx;
	double diffy = ypos - lasty;
	lastx = xpos;
	lasty = ypos;

	if (mouseState == ROTATE)
	{
		//rotate
		x_angle += (float)s_r * diffy;
		y_angle += (float)s_r * diffx;
		//x_angle = (float)s_r * diffy;
		//y_angle = (float)s_r * diffx;
	}
	else if (mouseState == TRANSLATE)
	{
		//translate
		x_trans += (float)(s_t * diffx);
		y_trans += (float)(-s_t * diffy);
	}
}

void mouseWheelCallback(GLFWwindow* window, double xoffset, double yoffset)
{
	const double s_s = 0.1;

	z_trans += -s_s * yoffset;

	
	//cout << cameraPosition.x << "," << cameraPosition.y << "," << cameraPosition.z << endl;
	//cout << cameraPosition.length()<<endl;
}


void updateCamera()
{
	//tmp
	glm::mat4 R =  glm::rotate(x_angle, glm::vec3(1.0f, 0.0f, 0.0f))
		* glm::rotate(y_angle, glm::vec3(0.0f, 1.0f, 0.0f));

	glm::mat4 T = glm::translate(glm::vec3(x_trans, y_trans, z_trans));

	glm::vec4 tmp = R * T * glm::vec4(0.0, 0.0, 0.0, 1.0);
	tmp /= tmp.w;
	cameraPosition = glm::vec3(tmp);

	view = glm::translate(glm::vec3(-x_trans, -y_trans, -z_trans)) * glm::transpose(R);

	projection = glm::perspective(45.0f, 1.0f, 0.1f, 100.0f);

}