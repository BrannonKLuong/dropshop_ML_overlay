# Design and Implementation of DropShop Algorithm
## Summary:
A brief high-level explanation of how the current implementation works is that every 
detection provided by the Machine Learning Model YoloV8 will check to find its closest droplet. 
Every frame in the video is analyzed with the machine learning (ML) model which produces
an array of detections with corresponding information 
x1, y1, x2, y2, id (this may or not be present depending on how the data was labeled), confidence percentage, class defined in the model (also dependent on how data was labeled)
This is denoted in the algorithm as one of the following:
```
   xone, yone, xtwo, ytwo, id, confidence, class_in_model = data
   xone, yone, xtwo, ytwo, confidence, class_in_model = data
```
Each detection is mapped to the Droplets that were initialized based on the facts that with the assumptions of when and where they begin. 
Each detection is then mapped to each droplet and subsequently updated in the positions of that Droplet.
Green boxes are straight segments, Blue boxes are Curve segments, and red dots are points along the curve to calculate the quadratic coefficients a, b, and c.
Black boxes are dispensers, purple boxes are the ML model's detections (the left number is ID, the right number is confidence),
Cyan/Aqua boxes inside the purple boxes are Python-initialized droplets to store data. 
If a detection is missed then the algorithm will attempt to predict where it'll be using the fact the Droplet is in a straight or curve traveling in 1 direction
#### Note get_droplet_on_screen() is hard-coded to initialize droplets at time T and location (x, y) referring to a dispenser. So this hard code varies by video

# The Design
## This following portion will elaborate on the design process.

# The Problem: 
#### Previous Algorithms would lose track of the droplets in the chip. When those droplets are reacquired by computer vision the lost droplet would be labeled a new droplet. Droplets being lost by computer vision or former models are described as disappearances. Additionally, previous implementations used limited forms of Labeling such as Model provided labeling or incrementing counter labeling. 

For example: if droplet 0 was lost in frame 2 denoted as d0. The counter would increment and label it new:
Frame 1: d0
Frame 2: 
Frame 3: d1

# Solution and Approach:

#### The solution leverages the simplicity of knowing that the course is consistent. Meaning that the droplets travel across a fixed path. The implementation must also keep in mind that Machine Learning Models do not explicitly have a form of carrying information over to determine and help detect in consecutive frames. Tracking algorithms such as Kalman's algorithm estimate where an object will be based on its trajectory from previous frames and its current position in current frames. Kalman's filter is ideal for more complex environments. Maintaining this logic the design of this implementation of Dropshop uses Droplet objects to retain as much information to infer the position. Due to the simplicity of the Course lay the following was used to infer the future position of droplets. Initialize a bounding box that shows the direction of the segment of the course and whether the segment is a curve or a straight. A straight, simple in its implementations, modifies the x or y value of a droplet by the direction it's heading. A curve needs to be labeled accurately to utilize the Quadratic equation to predict its traversal along the course's curve. From this point let d<sub>n</sub> denote droplets, course to be the entire path, segments be any part of the course, Det<sub>n</sub> to denote detections.

## Step 0: Loading Roboflow and Acquiring Weights and Data

## Step 1: Main and Variable Initialization
The main function begins with a Python clock timer to measure the run time of the algorithm.

The main function takes two arguments the weights and the video. The weights were generated in the previous step and should be an accessible file path. The video can be passed in any acceptable format. Note: for future implementation, the hope is to replace this video with the camera path.

1. Start the section of the code and call the main
    ```if __name__ == '__main__':
    '''Start Time and End Time is a timer to measure run time'''
    start_time = time.perf_counter()
    main("runs/detect/train10/weights/best.pt", "droplet_videos/video_data_Rainbow 11-11-22.m4v")
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
    ```
2. The first block of the main function before entering the while loops runs a handful of key functions. All Droplets initialize a global data structure to store the droplet objects. Where the list of segments is entered in the form of Type((x1, y2), (x2, y2), (Direction_x, direction y)). A corresponding list of start and middle-end points are of the form ((start x, start y), (mid x, mid y) (end x, y)) <b>if and only if</b> it is also a Curve Type.
   
    ```
    all_droplets = set()
    def build_course() -> Path:
    course = Path()
    lst_of_segments = [Straight((85, 50), (460, 70), (-1, 0)),  Curve((45, 50), (85, 110), (-1, 1)), Straight((45, 110), (60, 160), (0, 1)),
                       Curve((45, 160), (100, 205), (1, 1)), Straight((100, 180), (560, 205), (1, 0)),  Curve((560, 180), (600, 220), (1, 1)),
                       Straight((580, 220), (600, 300), (0, 1)), Curve((560, 300), (600, 340), (-1, 1)), Straight((0, 320), (560, 340), (-1, 0))]
    
    lst_of_sme = [None, ((85, 60), (60, 80), (50, 110)), None, ((50, 160), (70, 190), (100, 195)), None, ((560, 193), (580, 200), (590, 220)), None, ((590, 300), (580, 322), (560, 330)), None]

    for i in range(len(lst_of_segments)):
        segment = lst_of_segments[i]
        course.add_segment(segment)
        if isinstance(segment, Curve):
            s, m, e = lst_of_sme[i]
            segment.add_sme(s, m, e)
            
    return course
    ```
    
3. Course or build course is designed in one of two forms. One uses the User Interface the bounding box the user draws returns the same data that the testing purpose uses. For testing purposes, the course is initialized in a hard-coded way in two lists holding each Segment's data. For the sake of explanation, I'll be following a hard-coded version. The following function is designed with the idea that the algorithm will do one of two things. Load the data structure once and save it every time the same course is used or load the course before the experiment begins. The variable <b>x_y_map</b> will run through every segment of the course from the arrays and map every single (x, y) coordinate to its corresponding segment. The idea is to do a trade-off where the initial cost is some O(n<sup>2</sup>) for each of the x by y boxes for a future O(1) look-up time during the actual run of the algorithm. This way every detection returning some (x, y) can be mapped to a dictionary of its corresponding section. This allows us to quickly receive the data of the segment and the direction in which the detection should be traveling.
    ```
    x_y_map = build_x_y_map(course)
   def build_x_y_map(course: Path) -> {(int, int): Path}:
       ret_dic = {}
       for course in course.segments_in_order:
           x1, y1 = course.top_left
           x2, y2 = course.bottom_right
           smaller_x, bigger_x = min(x1, x2), max(x1, x2)
           smaller_y, bigger_y = min(y1, y2), max(y1, y2)
           for i in range(smaller_x, bigger_x + 1): 
               for j in range(smaller_y, bigger_y + 1):
                   ret_dic[(i, j)] = course
       return ret_dic
    ```
4. The remaining variables: box = sv.BoxAnnotator() is an ultralytics provided class to draw boxes. Speed Threshold is some arbitrarily chosen threshold for the speed of droplets. model is an initialized Yolo Model with the weights from the weighted path in Step 0 as its parameter. video cap is a cv2 loaded VideoCapture function using weight path as its parameter. t is a counter that helps show what frame the video is on. Droplets on screen are a global variable so that the algorithm knows how many droplets have been initialized this too is hard coded.

   ```
   all_droplets = set()
   course = build_course()
   x_y_map = build_x_y_map(course)
   box = sv.BoxAnnotator(text_scale=0.3)
   speed_threshold = 5
   # model, video_cap = load_mac_files()
   model = YOLO(weights_path)
   video_cap = cv2.VideoCapture(video_path)
   
   if not video_cap.isOpened():
        print("Error: Video file could not be opened.")
        return
   
   '''Initializes Time t to help debug on specific frames or time intervals of the video. Droplets on screen is a counter for how many droplets to expect at any given time t'''    
   t = 0
   droplets_on_screen = 0
   ```
## Step 2: The Main While Loop
This portion will cover what the main while loop will do which is how the algorithm runs on every frame. 

1. Droplets on screen are part of the hard-coded Droplet Assumptions. ret, frame and those are the frame opening portion of CV2.
```
    while video_cap.isOpened():
        t += 1 #Increment the time
        '''Open the video frames and play it'''
        ret, frame = video_cap.read()
        if not ret:
            print("Video ended")
            break
        if t > 0:
            droplets_on_screen = get_droplets_on_screen(t, droplets_on_screen, all_droplets, course)
```
2.  The result is an array returned by model.track of every detection on the given parameters: The current frame, a tracker (in this case arbitrarily chosen), persist = True which tells it to use previous frames to subsequently carry over information. The number detected is the length of the result used to determine if there's a missing detection. A found set is also used to see which droplets have been found to compare to all droplets to know which one is missing. Labels are a list used to store strings for the labeling of the droplets on the frame.

```
            droplets_on_screen = get_droplets_on_screen(t, droplets_on_screen, all_droplets, course)
            result = model.track(frame, tracker="bytetrack.yaml", persist=True)[0]
            numbers_detected = len(result)
            found = set()
            labels = []
```
3. 






    
