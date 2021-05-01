import numpy as np

from OpenGL.GL import *
import glfw

import models

def main():
    if not glfw.init():
        print("failed to initialize GLFW")
        return

    resolution = (1000, 1000)
    window = glfw.create_window(*resolution, 'lifegame', None, None)
    if not window:
        glfw.terminate()
        print("failed to create window")
        return

    glfw.make_context_current(window)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 5)

    model: models.Model = models.Lifegame()
    model.setup(resolution)

    fps = 30
    glfw.set_time(0)
    current = glfw.get_time()
    previous = glfw.get_time()
    glClearColor(0, 0, 0, 0)
    while not glfw.window_should_close(window):
        if current - previous >= 1.0 / fps:
            glClear(GL_COLOR_BUFFER_BIT)

            model.render()

            glfw.swap_buffers(window)
            previous = current

        glfw.poll_events()
        current = glfw.get_time()

    glfw.destroy_window(window)
    glfw.terminate()

if __name__ == "__main__":
    main()