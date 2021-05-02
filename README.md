# alife-gl

Alife-gl provides some multi-agent models accelerated by GPGPU. The models include CellMove, Lifegame, Gray=Scott and Boids.

## Getting Started

You can start to use this program easily. Enter the following commands at a Bash or Windows Powershell:

```
$ git clone https://github.com/asuka1975/alife-gl.git
$ cd alife-gl
$ pip install -r requirements.txt
```

## Execution

You can choose the model you want to simulate.
Open main.py, and change the model class as you can see in below code.

```python
def main():
    # initialize OpenGL

    model: models.Model = models.Boids() # simulate Boids model
    model.setup(window)

    # omitted

             |
             |
             V

def main():
    # initialize OpenGL

    model: model.Model = models.Lifegame() # simulate Lifegame
    model.setup(window)

    # omitted
```

At a commandline, enter the following command.

```
$ python main.py
```
