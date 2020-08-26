# About Donkey&reg;

Donkey is a high level self driving library written in Python. It was
developed with a focus on enabling fast experimentation and easy contribution.

---------

### Build your own Donkey

Donkey is the standard car that most people build first. The parts cost about $250 to $300 and take 2 hours to assemble. Here are the main steps to build your own car:

1. [Assemble hardware.](guide/build_hardware.md)
2. [Install software.](guide/install_software.md)
3. [Create Donkey App.](guide/create_application.md)
4. [Calibrate your car.](guide/calibrate.md)
5. [Start driving.](guide/get_driving.md)
6. [Train an autopilot.](guide/train_autopilot.md)
7. [Experiment with simulator.](guide/simulator.md)

---------------



### Hello World.

Donkeycar is designed to make adding new parts to your car easy. Here's an
example car application that captures images from the camera and saves them.

```python
import donkey as dk

#initialize the vehicle
V = dk.Vehicle()

#add a camera part
cam = dk.parts.PiCamera()
V.add(cam, outputs=['image'], threaded=True)

#add tub part to record images
tub = dk.parts.Tub(path='~/mycar/data',
                   inputs=['image'],
                   types=['image_array'])
V.add(tub, inputs=inputs)

#start the vehicle's drive loop
V.start(max_loop_count=100)
```
----------------

### Installation

[How to install](guide/install_software.md)

-----------------------

### Why the name Donkey?

The ultimate goal of this project is to build something useful. Donkey's were
one of the first domesticated pack animals, they're notoriously stubborn, and
they are kid safe. Until the car can navigate from one side of a city to the
other, we'll hold off naming it after some celestial being.
