#include <stdexcept>
#include <iostream>
#include <csignal>
#include <vector>
#include <SDL2/SDL.h>
#include <glad/glad.h>
#include <glm/glm.hpp>


constexpr int max_width = 512;
constexpr int max_height = 512;

constexpr const char* vertex_source = R"(
    #version 330 core
    #extension GL_ARB_explicit_uniform_location : enable

    layout (location = 0) in vec2 i_world_pos;
    layout (location = 1) in vec3 i_curv_pose;

    out vec3 o_curv_pose;

    layout (location = 0) uniform float scale;
    layout (location = 1) uniform float width;

    void main() {
        gl_Position = vec4(scale * i_world_pos, abs(i_curv_pose.y) / width, 1.0);
        o_curv_pose = i_curv_pose;
    }
)";

constexpr const char* fragment_source = R"(
    #version 330 core

    in vec3 o_curv_pose;

    out vec4 FragColor;

    void main() {
        FragColor = vec4(abs(o_curv_pose.y), 0.0f, 0.0f, 0.0f);
    }
)";


extern void cuda_test(GLuint rbo, uint width, uint height);


struct GlPath {
    GLuint vao;
    GLuint vbo;
    GLuint ebo;
};


void debugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam) {
    std::cerr << "\nGL DEBUG: \n" << message << "\n" << std::endl;
}

SDL_Window* init_sdl2_gl(const char* title, int x, int y, int width, int height, Uint32 flags = 0) {
    if (SDL_InitSubSystem(SDL_INIT_VIDEO) < 0) {
        throw std::runtime_error("Failed to initialize SDL2 library");
    }

    SDL_Window* window = SDL_CreateWindow(title, x, y, width, height, SDL_WINDOW_OPENGL | flags);

    if (!window) {
        throw std::runtime_error("Failed to create window");
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_DEBUG_FLAG);
    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    if (!gl_context) {
        std::cerr << SDL_GetError() << std::endl;
        throw std::runtime_error("Failed to create GL context");
    }

    if (!gladLoadGLLoader(SDL_GL_GetProcAddress)) {
        throw std::runtime_error("Failed to initialize GLAD");
    }

    if( SDL_GL_SetSwapInterval(1) < 0 ) {
        throw std::runtime_error("Failed to set vsync");
    }

    glDebugMessageCallbackARB(debugCallback, nullptr);

    return window;
}

void print_program_log(GLuint program)
{
    //Make sure name is shader
    if( glIsProgram( program ) )
    {
        //Program log length
        int infoLogLength = 0;
        int maxLength = infoLogLength;

        //Get info string length
        glGetProgramiv( program, GL_INFO_LOG_LENGTH, &maxLength );

        //Allocate string
        char* infoLog = new char[ maxLength ];

        //Get info log
        glGetProgramInfoLog( program, maxLength, &infoLogLength, infoLog );
        if( infoLogLength > 0 )
        {
            //Print Log
            printf( "%s\n", infoLog );
        }

        //Deallocate string
        delete[] infoLog;
    }
    else
    {
        printf( "Name %d is not a program\n", program );
    }
}

void print_shader_log(GLuint shader)
{
    //Make sure name is shader
    if( glIsShader( shader ) )
    {
        //Shader log length
        int infoLogLength = 0;
        int maxLength = infoLogLength;

        //Get info string length
        glGetShaderiv( shader, GL_INFO_LOG_LENGTH, &maxLength );

        //Allocate string
        char* infoLog = new char[ maxLength ];

        //Get info log
        glGetShaderInfoLog( shader, maxLength, &infoLogLength, infoLog );
        if( infoLogLength > 0 )
        {
            //Print Log
            printf( "%s\n", infoLog );
        }

        //Deallocate string
        delete[] infoLog;
    }
    else
    {
        printf( "Name %d is not a shader\n", shader );
    }
}

GLuint compile_shader(const char *vertex_source, const char *fragment_source) {
    GLuint program_id = glCreateProgram();
    GLuint vertexShader = glCreateShader( GL_VERTEX_SHADER );

    glShaderSource(vertexShader, 1, &vertex_source, nullptr);
    glCompileShader(vertexShader);

    GLint vShaderCompiled = GL_FALSE;
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &vShaderCompiled);
    if(vShaderCompiled != GL_TRUE) {
        print_shader_log(vertexShader);
        throw std::runtime_error("Unable to compile vertex shader");
    }

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(fragmentShader, 1, &fragment_source, nullptr);
    glCompileShader(fragmentShader);

    GLint fShaderCompiled = GL_FALSE;
    glGetShaderiv( fragmentShader, GL_COMPILE_STATUS, &fShaderCompiled );
    if( fShaderCompiled != GL_TRUE ) {
        print_shader_log(fragmentShader);
        throw std::runtime_error("Unable to compile fragment shader");
    }

    glAttachShader(program_id, vertexShader);
    glAttachShader(program_id, fragmentShader);
    glLinkProgram(program_id);

    GLint vProgramLinked;
    glGetProgramiv(program_id, GL_LINK_STATUS, &vProgramLinked);
    if(!vProgramLinked) {
        print_program_log(program_id);
        throw std::runtime_error("Failed to link program");
    }

    return program_id;
}

GLuint create_triangle() {
    GLuint VAO, VBO, EBO;

    float vertices[] = {
        -0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, // left
        0.5f, -0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, // right
        0.0f,  0.5f, 0.0f , 0.0f, 0.0f, 1.0f, 1.0f // top
    };

    GLuint indices [] = {
        0, 1, 2
    };

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glBindVertexArray(0);

    return VAO;
}

void draw_triangles(GLuint shader, GLuint VAO, GLsizei count) {
    glUseProgram(shader);
    glBindVertexArray(VAO);
    // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glDrawElements(GL_TRIANGLES, count * 3, GL_UNSIGNED_INT, nullptr);
}

void gen_framebuffer(GLsizei width, GLsizei height, GLuint& fbo, GLuint& rbo) {
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo);

    //add a depth buffer
    GLuint depth_rbo;
    glGenRenderbuffers(1, &depth_rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, depth_rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rbo);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        throw std::runtime_error("Framebuffer is not complete");
    }
}

void update_path(const GlPath& path, glm::fvec2 samples[], size_t n, float width) {
    struct Vertex {
        struct {
            float x;
            float y;
        } world;

        struct {
            float progress;
            float offset;
            float heading;
        } curv;
    };

    const float radius = width * 0.5f;
    constexpr float cone_cost = 10.0f;
    constexpr float cone_cost_squared = cone_cost * cone_cost;

    std::vector<Vertex> vertices;
    vertices.reserve(n);

    std::vector<GLuint> indices;
    indices.reserve(n * 6 * 3);

    float total_progress = 0;
    for (size_t i = 0; i < n - 1; i++) {
        glm::fvec2 p1 = samples[i];
        glm::fvec2 p2 = samples[i + 1];

        glm::fvec2 disp = p2 - p1;
        float new_progress = glm::length(disp);
        float segment_heading = std::atan2(disp.y, disp.x);


        glm::fvec2 prev = i == 0 ? p1 : samples[i - 1];
        float secant_heading = std::atan2(p2.y - prev.y, p2.x - prev.x);

        glm::fvec2 dir = glm::normalize(disp);
        glm::fvec2 normal = glm::fvec2(-dir.y, dir.x);

        glm::fvec2 low1 = p1 - normal * radius;
        glm::fvec2 low2 = p2 - normal * radius;
        glm::fvec2 high1 = p1 + normal * radius;
        glm::fvec2 high2 = p2 + normal * radius;

        if (i == 0) {
            vertices.push_back({{p1.x, p1.y}, {total_progress, 0.0f, segment_heading}});
        }
        vertices.push_back({{p2.x, p2.y}, {total_progress + new_progress, 0.0f, secant_heading}});

        vertices.push_back({{low1.x, low1.y}, {total_progress, -radius, segment_heading}});
        vertices.push_back({{low2.x, low2.y}, {total_progress + new_progress, -radius, segment_heading}});
        vertices.push_back({{high1.x, high1.y}, {total_progress, radius, segment_heading}});
        vertices.push_back({{high2.x, high2.y}, {total_progress + new_progress, radius, segment_heading}});

        const GLuint p1i = i == 0 ? 0 : (i - 1) * 5 + 1;
        const GLuint p2i = i * 5 + 1;
        const GLuint l1i = i * 5 + 2;
        const GLuint l2i = i * 5 + 3;
        const GLuint h1i = i * 5 + 4;
        const GLuint h2i = i * 5 + 5;

        indices.push_back(p1i);
        indices.push_back(p2i);
        indices.push_back(h2i);

        indices.push_back(h1i);
        indices.push_back(p1i);
        indices.push_back(h2i);

        indices.push_back(l1i);
        indices.push_back(l2i);
        indices.push_back(p2i);

        indices.push_back(p1i);
        indices.push_back(l1i);
        indices.push_back(p2i);

        if (i > 0) {
            const GLuint lpi = (i - 1) * 5 + 3;
            const GLuint hpi = (i - 1) * 5 + 5;

            indices.push_back(hpi);
            indices.push_back(p1i);
            indices.push_back(h1i);

            indices.push_back(lpi);
            indices.push_back(l1i);
            indices.push_back(p1i);
        }

        total_progress += new_progress;
    }

    glBindBuffer(GL_ARRAY_BUFFER, path.vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * vertices.size(), vertices.data(), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, path.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * indices.size(), indices.data(), GL_DYNAMIC_DRAW);
}

std::vector<glm::fvec2> sine_spline(float period, float amplitude, float progress, float density) {
    using namespace glm;

    std::vector<fvec2> result;

    fvec2 point {0.0f, 0.0f};
    float total_dist = 0;

    while (total_dist < progress) {
        result.push_back(point);

        fvec2 delta = normalize(fvec2(1.0f, amplitude * 2 * M_PI / period * cos(2 * M_PI / period * point.x)))
                    * density;
        total_dist += density;
        point += delta;
    }

    return result;
}

GlPath gen_path() {
    GlPath path {};

    glGenVertexArrays(1, &path.vao);
    glGenBuffers(1, &path.vbo);
    glGenBuffers(1, &path.ebo);

    glBindVertexArray(path.vao);
    glBindBuffer(GL_ARRAY_BUFFER, path.vbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, path.ebo);

    glBindVertexArray(0);

    return path;
}


int main() {
    constexpr float width = 5;
    SDL_Window* window = init_sdl2_gl("gl experiment", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, max_width, max_height, 0);

    GLuint fbo, rbo;
    gen_framebuffer(max_width, max_height, fbo, rbo);

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
0
    GLuint shader = compile_shader(vertex_source, fragment_source);
    constexpr GLint scale_loc = 0;
    constexpr GLint width_loc = 1;
    glUseProgram(shader);
    glUniform1f(scale_loc, 0.25f);
    glUniform1f(width_loc, width);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    GlPath path = gen_path();

    std::vector<glm::fvec2> samples = sine_spline(6.28f, 1.0f, 10.0f, 0.5f);
    update_path(path, samples.data(), samples.size(), width);

    draw_triangles(shader, path.vao, samples.size() * 6 - 2);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glBlitFramebuffer(0, 0, max_width, max_height, 0, 0, max_width, max_height, GL_COLOR_BUFFER_BIT, GL_NEAREST);

    SDL_GL_SwapWindow(window);

    sigset_t set;
    sigfillset(&set);
    sigdelset(&set, SIGINT);
    sigdelset(&set, SIGTERM);
    sigdelset(&set, SIGQUIT);
    sigsuspend(&set);

    return 0;
}