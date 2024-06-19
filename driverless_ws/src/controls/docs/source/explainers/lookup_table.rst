
Motivation: Given some inertial coordinates, we need to know how far from spline and how far along the spline
Iterative Closest Points (ICP) is very slow, hard to do on GPU because of a lot of branching instructions.
Create a lookup table with input inertial coordiantes and output distance from spline and distance along spline
Task becomes interpolating this mapping given a series of known points
Linear interpolation is very similar to graphics calculation

"render a triangle and render a texture onto that triangle"
Exploit hardware acceleration: texture/rasterization cores

CUDA has access to texture memory
Reason for OpenGL: Very quickly create textures - that is what it was designed to do
OpenGL takes in some vertices and textures and renders some triangles, then use shaders to render into a new texture.
Texture pipeline

Sacrifice: pretty far from track: have no data.

Method
Use points from path planning
Fake cones around the spline (based on a predetermined track width - overestimate). These will be vertices
- Because mppi calculates a bunch of terrible trajectories that go pretty out-of-bounds. We still want to get
meaningful cost data from these trajectories.
Draw triangles between cones.
Color the cones/vertices
Vertices include the points given by path planning

Color the vertices
R = distance along spline;
G = distance from spline;
B = yaw relative to spline; (UNSUED)
A = 1 everywhere in bounds, -1 out of bounds (fake bounds)
        (0,0,0,-1) is background colour, cone has 1 A
Along the spline, distance from spline (G) is 0

Vertex: A point with some known color information, vertex of the rendering triangles
Texture: a generalized image (in graphics terms) - mapping from 2D coordinates to RGBA/normal vector/whatever
Screen is a texture in OpenGL, and is usually the end goal, but it doesn't have to be.


Abuse depth testing
- Each point gets a depth (z = g)
- Points with lower z coordinate get clipped (occluded), because it is "blocked" by whatever in front
- Hardware accelerated using the "depth buffer"
Overlapping: two different reasonable measurements for distance from spline. How far along the spline should be

Note: the actual visualizer does not represent the labelled color


Generates a spline, sends to controller, rotates, iterates
test_node uses thomas' model. make sure to change mass to 210
Don't use controls_sim


Pipeline:
Load in a lot of data in a "Vertex Buffer Object": Continguous GPU memory, a buffer for the vertices
Specify how you lay it out with a "Vertex Array Object":



OPENGL CHEATSHEET

glGenFramebuffers(1, &m_curv_frame_lookup_fbo); - Generates a framebuffer object name.
glBindFramebuffer(GL_FRAMEBUFFER, m_curv_frame_lookup_fbo); - Binds a framebuffer to a framebuffer target.
glGenRenderbuffers(1, &m_curv_frame_lookup_rbo); - Generates a renderbuffer object name.
glBindRenderbuffer(GL_RENDERBUFFER, m_curv_frame_lookup_rbo); - Binds a renderbuffer to a renderbuffer target.
glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, curv_frame_lookup_tex_width, curv_frame_lookup_tex_width); - Creates and initializes a renderbuffer object's data store.
glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, m_curv_frame_lookup_rbo); - Attaches a renderbuffer object to a framebuffer object.
glCheckFramebufferStatus(GL_FRAMEBUFFER); - Checks the completeness status of a framebuffer.
glClearColor(0.0f, 0.0f, 0.0f, -1.0f); - Specifies clear values for the color buffers.
glEnable(GL_DEPTH_TEST); - Enables or disables server-side GL capabilities, here it enables depth testing.
glDepthFunc(GL_LESS); - Specifies the function used to compare each incoming pixel depth value with the depth value present in the depth buffer.
glViewport(0, 0, curv_frame_lookup_tex_width, curv_frame_lookup_tex_width); - Sets the viewport, which is the area on the window to which will be mapped the coordinates of the final image.
glGenVertexArrays(1, &m_gl_path.vao); - Generates a vertex array object name.
glGenBuffers(1, &m_gl_path.vbo); - Generates a buffer object name.
glBindVertexArray(m_gl_path.vao); - Binds a vertex array object.
glBindBuffer(GL_ARRAY_BUFFER, m_gl_path.vbo); - Binds a buffer object to the specified buffer binding point.
glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0); - Defines an array of generic vertex attribute data.
glEnableVertexAttribArray(0); - Enables a generic vertex attribute array.
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_gl_path.ebo); - Binds a buffer object to the specified buffer binding point.
glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * vertices.size(), vertices.data(), GL_DYNAMIC_DRAW); - Creates and initializes a buffer object's data store.
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); - Clears buffers to preset values.
glUseProgram(m_gl_path_shader); - Installs a program object as part of current rendering state.
glUniform1f(shader_scale_loc, 2.0f / m_curv_frame_lookup_tex_info.width); - Specifies the value of a uniform variable for the current program object.
glDrawElements(GL_TRIANGLES, (m_spline_frames.size() * 6 - 2) * 3, GL_UNSIGNED_INT, nullptr); - Renders primitives from array data.
glFinish(); - Blocks until all GL execution is complete.