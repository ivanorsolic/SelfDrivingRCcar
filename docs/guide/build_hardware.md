# How to Build a Donkey&reg;

&nbsp;

* [Overview](build_hardware.md#overview)
* [Parts Needed](build_hardware.md#parts-needed)
* [Hardware:](build_hardware.md#hardware)
  * [Step 1: Print Parts](build_hardware.md#step-1-print-parts)
  * [Step 2: Clean up parts](build_hardware.md#step-2-clean-up-parts)
  * [Step 3: Assemble Top plate and Roll Cage](build_hardware.md#step-3-assemble-top-plate-and-roll-cage)
  * [Step 4: Connect Servo Shield to Raspberry Pi](build_hardware.md#step-4-connect-servo-shield-to-raspberry-pi)
  * [Step 5: Attach Raspberry Pi to 3D Printed bottom plate](build_hardware.md#step-5-attach-raspberry-pi-to-3d-printed-bottom-plate)
  * [Step 6: Attach Camera](build_hardware.md#step-6-attach-camera)
  * [Step 7: Put it all together](build_hardware.md#step-7-put-it-all-together)
* [Software](install_software.md)

## Overview

These are updated instructions from the 2017 [Make Magazine article](https://makezine.com/projects/build-autonomous-rc-car-raspberry-pi/).  The latest version of the software installation instructions are maintained in the [software instructions](install_software.md) section.   Be sure to follow those instructions after you've built your car.

## Choosing a Car

There are 4 fully supported chassis all made under the "Exceed" Brand:

*  Exceed Magnet [Blue](https://www.amazon.com/gp/product/9269803775/?tag=donkeycar-20), [Red](http://amzn.to/2EIC1CF)
*  Exceed Desert Monster [Blue](http://amzn.to/2HLXJmc),  [Red](http://amzn.to/2pnIitV)
*  Exceed Short Course Truck  [Blue](https://amzn.to/2KsYF1e),  [Red](https://amzn.to/2rdtQ8z)
*  Exceed Blaze [Hyper Blue](https://amzn.to/2rf4MgS), [Yellow](https://amzn.to/2jlf3EA)

These cars are electrically identical but have different tires, mounting and other details.  It is worth noting that the Desert Monster, Short Course Truck and Blaze all require adapters which can be easily printed or purchased from the donkey store.  These are the standard build cars because they are mostly plug and play, both have a brushed motor which makes training easier, they handle rough driving surfaces well and are inexpensive.

In a pinch, the Latrax prerunner also works, with the existing adapters and plastics.  
LaTrax Prerunner [link](https://www.amazon.com/Traxxas-LaTrax-Electric-Prerunner-Control/dp/B07B3PQTRD)

Here is a [video](https://youtu.be/UucnCmCAGTI) overview of the different cars and how to assemble them.

In addition there are 3 more cars supported under the "Donkey Pro" name.  These are 1/10 scale cars which means that they are bigger, perform a little better and are slightly more expensive.  They can be found here:

* HobbyKing Trooper (not pro version) [found here](https://hobbyking.com/en_us/turnigy-trooper-sct-4x4-1-10-brushless-short-course-truck-arr.html?affiliate_code=XFPFGDFDZOPWEHF&_asc=9928905034)
* HobbyKing Mission-D [found here](https://hobbyking.com/en_us/1-10-hobbykingr-mission-d-4wd-gtr-drift-car-arr.html?affiliate_code=XFPFGDFDZOPWEHF&_asc=337569952)
* Tamaya TT01 or Clone - found worldwide but usually has to be built as a kits.  The other two cars are ready to be donkified, this one, however is harder to assemble.  

Here is a [video](https://youtu.be/K-REL9aqPE0) that goes over the different models.  The Donkey Pro models are not yet very well documented, just a word of warning.  

For more detail and other options, follow the link to: [supported cars](/supported_cars)

![donkey](/assets/build_hardware/donkey.PNG)

## Roll Your Own Car

Alternatively If you know RC or need something the standard Donkey does not support, you can roll your own.  Here is a quick reference to help you along the way.  [Roll Your Own](/roll_your_own.md)

## Video Overview of Hardware Assembly

This [video](https://www.youtube.com/watch?v=OaVqWiR2rS0&t=48s) covers how to assemble a standard Donkey Car, it also covers the Sombrero, the Raspberry Pi and the nVidia Jetson Nano.  

[![IMAGE ALT TEXT HERE](/assets/HW_Video.png)](https://www.youtube.com/watch?v=OaVqWiR2rS0&t=48s)

## Parts Needed

The following instructions are for the Raspbeery Pi, below in Optional Upgrades section, you can find the NVIDIA Jetson Nano instructions.  

### Option 1: Buying through an official Donkey Store

There are two official stores:

If you are in the US, you can use the [Donkey store](https://store.donkeycar.com).  The intention of the Donkey Store is to make it easier and less expensive to build the Donkey Car.  The Donkey Store is run by the original founders of donkey car and profits are used to fund development of the donkey cars.  Also it is worth noting the design of the parts out of the Donkey store is slightly improved over the standard build as it uses better parts that are only available in large quantities or are harder to get.  The Donkey Store builds are open source like all others.   

If you are in Asia, the DIYRobocars community in Hong Kong also sells car kits at [Robocar Store](https://www.robocarstore.com/products/donkey-car-starter-kit).  They are long term Donkey community members and use proceeds to support the R&D efforts of this project. It is worth noting they can also sell to Europe and the US but it is likely less cost effective.  

| Part Description                                                                    | Link                                                                                  | Approximate Cost |
|-------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|------------------|
| Exceed Magnet, Desert Monster, Blaze, or Short Course Truck                                                                       | See links above                                     | ~$90              |
| USB Battery with microUSB cable (any battery capable of 2A 5V output is sufficient) | [Anker 6700 mAh](http://amzn.to/2ptshm0)                                           | $17              |
| Raspberry Pi 3b+                                                                      | [amazon.com/gp/product/B01CD5VC92](https://www.amazon.com/ELEMENT-Element14-Raspberry-Pi-Motherboard/dp/B07BDR5PDW?tag=donkeycar-20)                                          | $38              |
| MicroSD Card (many will work, we strongly recommend this one)             | [amazon.com/gp/product/B01HU3Q6F2](https://www.amazon.com/SanDisk-128GB-Extreme-microSD-Adapter/dp/B07FCMKK5X?tag=donkeycar-20)                            | $18.99           |
| Donkey Partial Kit                                                      | [KIT](https://store.donkeycar.com/collections/frontpage)                                        | $82 to $125              |

### Option 2: Bottoms Up Build

If you want to buy the parts yourself, want to customize your donkey or live out to of the US, you may want to choose the bottoms up build.  Keep in mind you will have to print the donkey car parts which can be found [here](https://www.thingiverse.com/thing:2566276)

| Part Description                                                                    | Link                                                                                  | Approximate Cost |
|-------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|------------------|
| Magnet Car or alternative                                                                        | [Blue](https://www.amazon.com/gp/product/9269803775/?tag=donkeycar-20), [Red](http://amzn.to/2EIC1CF)                                         | $92              |
| M2x6 screws (4)                                                                     | [Zinc](https://www.amazon.com/uxcell-Stainless-Phillips-Tapping-Screws/dp/B01KXTSW6Q?tag=donkeycar-20)                                          | $3.50 &ast;          |
| M3x10 screws (8)                                                                  | [Black Oxide](https://www.amazon.com/Screws-Mushroom-Phillips-Self-Tapping-Electronic/dp/B07NQCG6JP?tag=donkeycar-20)                                          | $7.89 &ast;          |
| USB Battery with microUSB cable (any battery capable of 2A 5V output is sufficient) | [Anker 6700 mAh](http://amzn.to/2ptshm0)                                           | $17              |
| Raspberry Pi 3b+                                                                      | [amazon.com/gp/product/B01CD5VC92](https://www.amazon.com/ELEMENT-Element14-Raspberry-Pi-Motherboard/dp/B07BDR5PDW?tag=donkeycar-20)                                          | $38              |
| MicroSD Card (many will work, I like this one because it boots quickly)             | [amazon.com/gp/product/B01HU3Q6F2](https://www.amazon.com/SanDisk-128GB-Extreme-microSD-Adapter/dp/B07FCMKK5X?tag=donkeycar-20)                                         | $18.99           |
| Wide Angle Raspberry Pi Camera                                                      | [amazon.com/gp/product/B00N1YJKFS](https://www.amazon.com/gp/product/B00N1YJKFS?tag=donkeycar-20)                                         | $25              |
| Female to Female Jumper Wire                                                        | [amazon.com/gp/product/B010L30SE8](https://www.amazon.com/gp/product/B010L30SE8?tag=donkeycar-20)                                          | $7 &ast;             |
| Servo Driver PCA 9685                                                               | [amazon.com/gp/product/B014KTSMLA](https://www.amazon.com/gp/product/B014KTSMLA?tag=donkeycar-20)                                          | $12 &ast;&ast;           |
| 3D Printed roll cage and top plate.                                                 | Purchase: [Donkey Store](https://store.donkeycar.com/collections/plastics-and-screws/products/standard-donkey-chassis-includes-screws) Files: [thingiverse.com/thing:2260575](https://www.thingiverse.com/thing:2566276) | $50                 |

&ast; If it is hard to find these components there is some wiggle room. Instead of an M2 you can use an M2.2, m2.3 or #4 SAE screw.  Instead of an M3 a #6 SAE screw can be used.  Machine screws can be used in a pinch.  

&ast;&ast; This component can be purchased from Ali Express for ~$2-4 if you can wait the 30-60 days for shipping.

### Optional Upgrades

* **NVIDIA JetsonNano Hardware Options**  The NVIDIA Jetson Nano is fully supported by the donkey Car.  To assemble the Donkey Car you will need a few parts including the Wifi card, Antennas and camera.  In addition you will need this [Adapter](https://store.donkeycar.com/products/jetson-donkey-adapter) if you want to print it yourself it is on the Thingiverse page for the project.

![adapter](/assets/Jetson_Adapter.jpg)

Plug in the Servo driver the same as the Raspberry Pi, just keep in mind that the Jetson pinout is reversed and that the Sombrero is not supported.

![Jetson Servo](/assets/Servo_Wiring.png)

Finally this is the Donkey Assembled.  

![Jetson Assembled](/assets/Jetbot_Assembled.png)

| Part Description                                      | Link                                                              | Approximate Cost |
|-------------------------------------------------------|-------------------------------------------------------------------|------------------|
| Nvidia Jetson Nano | [Jetson Nano](https://www.amazon.com/NVIDIA-Jetson-Nano-Developer-Kit/dp/B07PZHBDKT?tag=donkeycar-20)| $99 |
| Jetson Nano Adapter | [Adapter](https://store.donkeycar.com/products/jetson-donkey-adapter) | $7          |
| Camera Module | [Camera](https://store.donkeycar.com/products/nvidia-jetson-camera-for-donkey)| $27 |
| WiFi Card | [Card](https://www.amazon.com/Intel-Dual-Band-Wireless-Ac-8265/dp/B01MZA1AB2?tag=donkeycar-20) | $18|
| Antennas | [Antennas](https://store.donkeycar.com/products/2x-molex-wifi-antennas-for-jetson-nano)|$7|

For other options for part, feel free to look at the jetbot documentation [here](https://github.com/NVIDIA-AI-IOT/jetbot).

* **Sombrero Hat** The sombrero hat replaces the Servo driver and the USB battery and can be purchased at the Donkeycar store [here](https://store.donkeycar.com/collections/accessories/products/sombrero) and video instructions can be found [here](https://www.youtube.com/watch?v=vuAXdrtNjpk). Implementing the Sombrero hat requires a LiPo battery (see below).  Documentation is in [Github](https://github.com/autorope/Sombrero-hat).

![sombrero](/assets/Sombrero_assembled.jpg)

* **LiPo Battery and Accessories:** LiPo batteries have significantly better energy density and have a better dropoff curve.  See below (courtesy of Traxxas).

![donkey](/assets/build_hardware/traxxas.PNG)

| Part Description                                      | Link                                                              | Approximate Cost |
|-------------------------------------------------------|-------------------------------------------------------------------|------------------|
| LiPo Battery                                          | [hobbyking.com/en_us/turnigy-1800mah-2s-20c-lipo-pack.html](https://hobbyking.com/en_us/turnigy-1800mah-2s-20c-lipo-pack.html?affiliate_code=XFPFGDFDZOPWEHF&_asc=1096095044) or [amazon.com/gp/product/B0072AERBE/](https://www.amazon.com/gp/product/B0072AERBE/) | $8.94 to $~17           |
| Lipo Charger (takes 1hr to charge the above battery)  | [amazon.com/gp/product/B00XU4ZR06](https://www.amazon.com/gp/product/B00XU4ZR06?tag=donkeycar-20)                                               | $13              |
| Lipo Battery Case (to prevent damage if they explode) | [amazon.com/gp/product/B00T01LLP8](https://www.amazon.com/gp/product/B00T01LLP8?tag=donkeycar-20)                                               | $8               |

## Hardware

If you purchased parts from the Donkey Car Store, skip to step 3.

### Step 1: Print Parts

If you do not have a 3D Printer, you can order parts from [Donkey Store](https://store.donkeycar.com/collections/plastics-and-screws/products/standard-donkey-chassis-includes-screws), [Shapeways](https://www.shapeways.com/) or [3dHubs](https://www.3dhubs.com/).  I printed parts in black PLA, with 2mm layer height and no supports.  The top roll bar is designed to be printed upside down.   Remember that you need to print the adapters unless you have a "Magnet"

I printed parts in black PLA, with .3mm layer height with a .5mm nozzle and no supports.  The top roll bar is designed to be printed upside down.  

### Step 2: Clean up parts

Almost all 3D Printed parts will need clean up.  Re-drill holes, and clean up excess plastic.

![donkey](/assets/build_hardware/2a.PNG)

In particular, clean up the slots in the side of the roll bar, as shown in the picture below:

![donkey](/assets/build_hardware/2b.PNG)

### Step 3: Assemble Top plate and Roll Cage

If you have an Exceed Short Course Truck, Blaze or Desert Monster watch this [video](https://youtu.be/UucnCmCAGTI)

This is a relatively simple assembly step.   Just use the 3mm self tapping screws to scew the plate to the roll cage.  

When attaching the roll cage to the top plate, ensure that the nubs on the top plate face the roll-cage. This will ensure the equipment you mount to the top plate fits easily.

### Step 4: Connect Servo Shield to Raspberry Pi

***note: this is not necessary if you have a Sombrero, the Sombrero just plugs into the Pi***

You could do this after attaching the Raspberry Pi to the bottom plate, I just think it is easier to see the parts when they are laying on the workbench.  Connect the parts as you see below:

![donkey](/assets/build_hardware/4a.PNG)

For reference, below is the Raspberry Pi Pinout for reference.  You will notice we connect to 3.3v, the two I2C pins (SDA and SCL) and ground:

![donkey](/assets/build_hardware/4b.PNG)

### Step 5: Attach Raspberry Pi to 3D Printed bottom plate

Before you start, now is a good time to insert the already flashed SD card and bench test the electronics.  Once that is done, attaching the Raspberry Pi and Servo is as simple as running screws through the board into the screw bosses on the top plate.  The M2.5x12mm screws should be the perfect length to go through the board, the plastic and still have room for a washer.  The “cap” part of the screw should be facing up and the nut should be on the bottom of the top plate.  The ethernet and USB ports should face forward.  This is important as it gives you access to the SD card and makes the camera ribbon cable line up properly.

Attach the USB battery to the underside of the printed bottom plate using cable ties or velcro.

![donkey](/assets/build_hardware/5ab.PNG)

### Step 6: Attach Camera

Slip the camera into the slot, cable end first.  However, be careful not to push on the camera lens and instead press the board.
![donkey](/assets/build_hardware/assemble_camera.jpg)

If you need to remove the camera the temptation is to push on the lens, instead push on the connector as is shown in these pictures.  
![donkey](/assets/build_hardware/Remove--good.jpg) ![donkey](/assets/build_hardware/Remove--bad.jpg)

Before using the car, remove the plastic film or lens cover from the camera lens.

![donkey](/assets/build_hardware/6a.PNG)

It is easy to put the camera cable in the wrong way so look at these photos and make sure the cable is put in properly.  There are loads of tutorials on youtube if you are not used to this.

![donkey](/assets/build_hardware/6b.PNG)

### Step 7: Put it all together

*** Note if you have a Desert Monster Chassis see 7B section below ***
The final steps are straightforward.  First attach the roll bar assembly to the car.  This is done using the same pins that came with the vehicle.  

![donkey](/assets/build_hardware/7a.PNG)

Second run the servo cables up to the car.  The throttle cable runs to channel 0 on the servo controller and steering is channel 1.

![donkey](/assets/build_hardware/7b.PNG)

Now you are done with the hardware!!

### Step 7b: Attach Adapters (Desert Monster only)

The Desert monster does not have the same set up for holding the body on the car and needs two adapters mentioned above.  To attach the adapters you must first remove the existing adapter from the chassis and screw on the custom adapter with the same screws as is shown in this photo:

![adapter](/assets/build_hardware/Desert_Monster_adapter.png)

Once this is done, go back to step 7

## Software

Congrats!  Now to get your get your car moving, see the [software instructions](install_software.md) section.

![donkey](/assets/build_hardware/donkey2.PNG)

> We are a participant in the Amazon Services LLC Associates Program, an affiliate advertising program designed to provide a means for us to earn fees by linking to Amazon.com and affiliated sites.
