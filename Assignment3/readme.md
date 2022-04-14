# Object Tracking

## Run Command
python3 ObjectTracking.py

## Requirements
Need scikit and open cv for running the program.

## Motion Detection
The main idea behind this motion detector is that we find the changes for 3 successive frames. We have threee frames: cframe(current frame), pframe (previous frame), and ppframe (previous previous frame). Then we find the difference between cframe and pframe, and then between pframe and ppframe. We take the difference of frame with minimum change and then using a threshold value, change all the pixels aboce that value to 1 and rest to 0.

![thresh_frame](https://user-images.githubusercontent.com/60827845/163307836-22096f37-dfdf-4aa9-ba53-04412ca3a233.jpeg)

Then, these white spots in thee image that depict change over 2 frames are turned to blobs using dilation.

![dilated_frame](https://user-images.githubusercontent.com/60827845/163307860-f8f5f576-a46a-4e75-b102-5266f77229aa.jpeg)

We ccan create a rectangle around these blobs in original frame in order to track the motion. The image below was not converted from BGR to RGB.

![frame](https://user-images.githubusercontent.com/60827845/163308268-fa9545b8-bc8b-405e-85a0-36b50f2a2ff9.jpeg)
