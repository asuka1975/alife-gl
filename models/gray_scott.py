import typing as tp

import numpy as np

from OpenGL.GL import *

from . import shader

class GrayScott:
    vert = """
#version 450

layout (std430, binding = 0) buffer us {
    float u[];
};

layout (location = 0) out vec4 vertexColor;

uniform ivec2 rect;

void main() {
    float fid = gl_VertexID;
    float x = mod(fid, rect.x) / rect.x;
    float y = int(fid / rect.x) / float(rect.y);

    gl_Position = vec4(vec2(x, y) * 2 - 1, 0, 1);
    vertexColor = vec4(u[gl_VertexID]);
}
"""

    geom = """
#version 450

layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

layout (location = 0) in vec4 vertexColor[];
layout (location = 0) out vec4 outColor;

uniform ivec2 rect;

void main(void) {
    gl_Position = gl_in[0].gl_Position;
    outColor = vertexColor[0];
    EmitVertex();
    gl_Position = gl_in[0].gl_Position + vec4(2.0 / rect.x, 0, 0, 0);
    outColor = vertexColor[0];
    EmitVertex();
    gl_Position = gl_in[0].gl_Position + vec4(0, 2.0 / rect.y, 0, 0);
    outColor = vertexColor[0];
    EmitVertex();
    gl_Position = gl_in[0].gl_Position + vec4(2.0 / rect.x, 2.0 / rect.y, 0, 0);
    outColor = vertexColor[0];
    EmitVertex();
    EndPrimitive();
}    
"""

    frag = """
#version 450 

layout (location = 0) in vec4 fragColor;
layout (location = 0) out vec4 outColor;

void main() {
    outColor = fragColor;
}
"""

    comp = """
#version 450

layout (local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout (std430, binding = 0) buffer us {
    float u[];
};

layout (std430, binding = 1) buffer u_temps {
    float u_temp[];
};

layout (std430, binding = 2) buffer vs {
    float v[];
};

layout (std430, binding = 3) buffer v_temps {
    float v_temp[];
};

uniform ivec2 rect;

float get_u(int x, int y) {
    return u[x + y * rect.x];
}

float neighbor_u(int x, int y, int dx, int dy) {
    int _x = x + dx < 0 ? rect.x - 1 : x + dx >= rect.x ? 0 : x + dx;
    int _y = y + dy < 0 ? rect.y - 1 : y + dy >= rect.y ? 0 : y + dy;
    return get_u(_x, _y);
}

float get_v(int x, int y) {
    return v[x + y * rect.x];
}

float neighbor_v(int x, int y, int dx, int dy) {
    int _x = x + dx < 0 ? rect.x - 1 : x + dx >= rect.x ? 0 : x + dx;
    int _y = y + dy < 0 ? rect.y - 1 : y + dy >= rect.y ? 0 : y + dy;
    return get_v(_x, _y);
}

const float dx = 0.01;
const float dt = 1;
const float Du = 2e-5;
const float Dv = 1e-5;
const float f = 0.022;
const float k = 0.051;

void main() {
    uint work_id = gl_WorkGroupID.z * gl_NumWorkGroups.x * gl_NumWorkGroups.y + 
                   gl_WorkGroupID.y * gl_NumWorkGroups.x +
                   gl_WorkGroupID.x;
    uint id = gl_LocalInvocationIndex + work_id * 32 * 32;

    int x = int(mod(id, rect.x));
    int y = int(id / rect.x);

    float laplacian_u = (neighbor_u(x, y, 1, 0) + neighbor_u(x, y, -1, 0) + neighbor_u(x, y, 0, 1) + neighbor_u(x, y, 0, -1) - 4 * u[id]) / (dx * dx);
    float laplacian_v = (neighbor_v(x, y, 1, 0) + neighbor_v(x, y, -1, 0) + neighbor_v(x, y, 0, 1) + neighbor_v(x, y, 0, -1) - 4 * v[id]) / (dx * dx);
    float dudt = Du * laplacian_u - u[id] * v[id] * v[id] + f * (1 - u[id]);
    float dvdt = Dv * laplacian_v + u[id] * v[id] * v[id] - (f + k) * v[id];
    u_temp[id] = u[id] + dt * dudt;
    v_temp[id] = v[id] + dt * dvdt;
}
"""

    copy = """
#version 450

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout (std430, binding = 0) buffer us {
    float u[];
};

layout (std430, binding = 1) buffer u_temps {
    float u_temp[];
};

layout (std430, binding = 2) buffer vs {
    float v[];
};

layout (std430, binding = 3) buffer v_temps {
    float v_temp[];
};

void main() {
    uint work_id = gl_WorkGroupID.z * gl_NumWorkGroups.x * gl_NumWorkGroups.y + 
                   gl_WorkGroupID.y * gl_NumWorkGroups.x +
                   gl_WorkGroupID.x;
    uint id = gl_LocalInvocationIndex + work_id * 32 * 32;

    u[id] = u_temp[id];
    v[id] = v_temp[id];
}
"""

    def __init__(self) -> None:
        self.rect = (1000, 1000)

    def setup(self, window: tp.Any) -> None:
        self.program = shader.Shader()
        self.program.attach_shader(self.vert, GL_VERTEX_SHADER)
        self.program.attach_shader(self.geom, GL_GEOMETRY_SHADER)
        self.program.attach_shader(self.frag, GL_FRAGMENT_SHADER)
        self.program.link()

        self.program.use()
        glUniform2i(glGetUniformLocation(self.program.handle, "rect"), *self.rect)
        self.program.unuse()

        us = np.ones(self.rect[0] * self.rect[1], dtype=np.float32)
        vs = np.zeros(self.rect[0] * self.rect[1], dtype=np.float32)
        square = 20
        for j in range(square):
            for i in range(square):
                index = self.rect[0] // 2 - square // 2 + i + (self.rect[1] // 2 - square // 2 + j) * self.rect[0]
                us[index] = 0.5
                vs[index] = 0.25
        us += np.random.rand(us.shape[0]).astype(np.float32) * 0.1
        vs += np.random.rand(vs.shape[0]).astype(np.float32) * 0.1
        temporary = np.zeros(us.shape[0])
        ssbo = glGenBuffers(4)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[0])
        glBufferData(GL_SHADER_STORAGE_BUFFER, us.itemsize * us.shape[0], us, GL_STATIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[1])
        glBufferData(GL_SHADER_STORAGE_BUFFER, temporary.itemsize * temporary.shape[0], temporary, GL_STATIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[2])
        glBufferData(GL_SHADER_STORAGE_BUFFER, vs.itemsize * vs.shape[0], vs, GL_STATIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[3])
        glBufferData(GL_SHADER_STORAGE_BUFFER, temporary.itemsize * temporary.shape[0], temporary, GL_STATIC_DRAW)

        self.compute = shader.Shader()
        self.compute.attach_shader(self.comp, GL_COMPUTE_SHADER)
        self.compute.link()

        self.memory_copy = shader.Shader()
        self.memory_copy.attach_shader(self.copy, GL_COMPUTE_SHADER)
        self.memory_copy.link()

        self.compute.use()
        glUniform2i(glGetUniformLocation(self.compute.handle, "rect"), *self.rect)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[0])
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo[1])
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssbo[2])
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssbo[3])
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        self.compute.unuse()

        self.memory_copy.use()
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[0])
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo[1])
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssbo[2])
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssbo[3])
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        self.memory_copy.unuse()

    def render(self) -> None:
        self.compute.use()
        glDispatchCompute(64, 64, 1)
        self.memory_copy.use()
        glDispatchCompute(64, 64, 1)

        self.program.use()
        glDrawArrays(GL_POINTS, 0, self.rect[0] * self.rect[1])
        self.program.unuse()

