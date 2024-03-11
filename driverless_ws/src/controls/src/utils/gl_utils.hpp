#pragma once

#include <glad/glad.h>
#include <SDL2/SDL.h>


namespace controls {
    namespace utils {
        struct GLObj {
            GLuint vao;
            GLuint vbo;
            GLuint ebo;
        };

        SDL_Window* create_sdl2_gl_window(const char *title, int width, int height, Uint32 additional_flags = 0, SDL_GLContext* gl_context = nullptr);
        void print_program_log(GLuint program);
        void print_shader_log(GLuint shader);
        void gl_debug_callback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam);
        GLuint compile_shader(const char* vertex_source, const char* fragment_source);
    }
}