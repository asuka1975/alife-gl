import numpy as np

from OpenGL.GL import *

from . import shader

class Lifegame:
    vert = """
#version 450

layout (std430, binding = 0) buffer cells {
    int cell[];
};

layout (location = 0) out vec4 vertexColor;

uniform ivec2 rect;

void main() {
    float x = mod(gl_VertexID, rect.x) / rect.x;
    float y = float(gl_VertexID) / rect.x / rect.y;

    gl_Position = vec4(vec2(x, y) * 2 - 1, 0, 1);
    vertexColor = vec4(cell[gl_VertexID]);
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

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout (std430, binding = 0) buffer cells {
    int cell[];
};

shared int tmp[1024];

uniform ivec2 rect;

int neighbor(int x, int y, int dx, int dy) {
    return cell[x + dx < 0 ? rect.x - 1 : x + dx >= rect.x ? 0 : x + dx +
               (y + dy < 0 ? rect.y - 1 : y + dy >= rect.y ? 0 : y + dy) * rect.x];
}

void main() {
    uint work_id = gl_WorkGroupID.z * gl_NumWorkGroups.x * gl_NumWorkGroups.y + 
                   gl_WorkGroupID.y * gl_NumWorkGroups.x +
                   gl_WorkGroupID.x;
    uint id = gl_LocalInvocationIndex + work_id * 32 * 32;

    int x = int(mod(id, rect.x));
    int y = int(id / rect.x);

    int num = 0;
    num += neighbor(x, y, -1, 0);
    num += neighbor(x, y, -1, -1);
    num += neighbor(x, y, 0, -1);
    num += neighbor(x, y, 1, -1);
    num += neighbor(x, y, 1, 0);
    num += neighbor(x, y, 1, 1);
    num += neighbor(x, y, 0, 1);
    num += neighbor(x, y, -1, 1);
    tmp[gl_LocalInvocationIndex] = num == 3 ? 1 : num == 2 ? cell[id] : 0;

    barrier();

    cell[id] = tmp[gl_LocalInvocationIndex];
}
"""

    def __init__(self) -> None:
        self.rect = (1000, 1000)

    def setup(self, resolution: tuple[int, int]) -> None:
        self.program = shader.Shader()
        self.program.attach_shader(self.vert, GL_VERTEX_SHADER)
        self.program.attach_shader(self.frag, GL_FRAGMENT_SHADER)
        self.program.link()

        self.program.use()
        glUniform2i(glGetUniformLocation(self.program.handle, "rect"), *self.rect)
        self.program.unuse()

        field = np.random.randint(0, 2, self.rect[0] * self.rect[1])
        ssbo = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, field.itemsize * field.shape[0], field, GL_STATIC_DRAW)

        self.compute = shader.Shader()
        self.compute.attach_shader(self.comp, GL_COMPUTE_SHADER)
        self.compute.link()

        self.compute.use()
        glUniform2i(glGetUniformLocation(self.compute.handle, "rect"), *self.rect)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        self.compute.unuse()

    def render(self) -> None:
        self.compute.use()
        glDispatchCompute(64, 64, 1)

        self.program.use()
        glDrawArrays(GL_POINTS, 0, self.rect[0] * self.rect[1])
        self.program.unuse()

