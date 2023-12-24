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
### Requirements
### Python 3.11 >= Python 3.x
Ultralytics is not yet compatible with Python 3.12

### Robowflow
```
pip install roboflow
```

### Ultralytics
```
pip install ultralytics
```

### Opencv CV2
```
pip install opencv-python==4.8.0.76
```

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
2. The first block of the main function before entering the while loops runs a handful of key functions. All Droplets initialize a global data structure to store the droplet objects. Where the list of segments is entered in the form of Type((x1, y2), (x2, y2), (Direction_x, direction y)). A corresponding list of start and middle-end points are of the form ((start x, start y), (mid x, mid y) (end x, y)) <b>IF AND ONLY IF</b> it is also a Curve Type.
   
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
3. The next following try-except block is the most error-prone be it the algorithm running into type errors or the lack of sufficient data from the model. Note that there exist many more nested exceptions that will continue onto the next detection cause despite faults the algorithm will need to retain as much information as possible to reduce general inaccuracies.  <b>Future implementations</b> will attempt to address this issue. The next following block will utilize the data from each detection in the format of xone, yone, xtwo, ytwo, id, confidence, class_in_model <b>OR</b> xone, yone, xtwo, ytwo, confidence, class_in_model. Mid is the middle point of each detection. Drops to consider were initially designed to be a data structure local to a particular path segment but that was error-prone. A safer approach is currently implemented using a global approach. A better implementation will be discussed in the <b>Future Implementation Section</b>. Let Drops to consider be all droplets. 

```
            try:
                for data in result.boxes.data.tolist():
                    try:
                        xone, yone, xtwo, ytwo, _, confidence, _ = data
                    except ValueError:
                        print("No Data given by detection")

                    mid = get_mid_point(xone, yone, xtwo, ytwo)

                    try:
                        drops_to_consider = all_droplets

                    except KeyError:
                        print("Detection occurred outside of the course. Data: ", data)
                        continue

                    except Exception as e:
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)
                        continue
```
4.  The core of the logic is to find the closest to a given detection's (x, y) coordinate.
```
                    closest_droplet = find_closest_droplet(drops_to_consider, mid)
def find_closest_droplet(drops_to_consider: {Droplet}, mid:(int, int)) -> Droplet:
    closest = float('inf')
    closest_drop = None
    for drop in drops_to_consider:
        drop_point = (drop.x, drop.y)
        distance = get_distance(drop_point, mid) 
        if distance < closest:
            closest_drop = drop
            closest = distance
    return closest_drop  
```
5. Once acquiring the closest droplet to a detection it'll be added to a data structure of found. Each droplet will then update its information with the new (x, y) position as well as try to calculate its average trajectory since it's last been seen. If the detection occurred in a different segment than the droplet it just acquired, it has moved to a new segment. The algorithm will then compare the (x and y) coordinates to compare and update the section accordingly. The initial implementation used a primitive way to carry over data. The idea was to reduce overall average run time using localized data structures for particular segments. This would allow the algorithm to compare detections to only droplets in the segment it was discovered reducing the overall run time on average from O(n<sub>2</sub>) to some O(n detections * m droplets in that smaller segment). However, this implementation was error-prone and not the ideal way to both carryover data and compare detections to droplets.

An example of how the first implementation was supposed to work: 
As you can imagine each segment would ideally have a localized data structure that kept track of the droplets in itself.
![screenshot of the idea of local data storage](https://github.com/BrannonKLuong/dropshop_ML_overlay/blob/main/img_assets/datastorage_first.PNG)

## What ended up happening was this:
![screen shot of data carry over implementation](https://github.com/BrannonKLuong/dropshop_ML_overlay/blob/main/img_assets/datastorage_first_result.PNG)

In attempts to compensate for the uncertainty of when the data should be transferred over, a copy of the data would be carried in the next set to allow for ease of deletion and transfer.
```
                    closest_droplet = find_closest_droplet(drops_to_consider, mid)
                    found.add(closest_droplet)

                    closest_droplet.update_last_seen(mid, t, x_y_map, speed_threshold)

                    if x_y_map[mid] != course.segments_in_order[closest_droplet.current_section]:
                        closest_droplet.update_section(course, closest_droplet)
```
6. Box Drops are a way to draw a bounding box around each droplet and then the labeling string is appended to labels for the frame to be labeled.
```
                    box_drops(all_droplets, frame)

                    if confidence:
                        labels.append(f"{closest_droplet.id} {confidence:0.2f}")
```
7. The next section is a core part of the logic where <b>Det<sub>n</sub> < d<sub>n</sub></b> which means there are missing Droplets. With this, we do the following. Find the difference between all the droplets and the droplets that have been found. For each missing droplet infer their position. This inference with the update position and update section will be elaborated in the Droplet Class Object Section.
```
                if numbers_detected < droplets_on_screen:
                    handle_missings(all_droplets, found, course)

def handle_missings(drops: {Droplet}, found: set, map_course: Path) -> None:
    missing = drops.difference(found)
    for drop in missing:
        drop.update_position(map_course, drop)
        found.add(drop)
```
8. The remainder of this algorithm finishes labeling the data.

```
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)

        # where_droplets_should_start(frame)  #Call to show dispenser locations

        detections = sv.Detections.from_ultralytics(result)
        label_course(frame, course) 
        label_curves_s_m_e(frame, course)

        frame = box.annotate(scene=frame, detections=detections, labels = labels)
        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(10) == 27):
            break
```
## Step 3: The Droplet Object
1. The Droplet object is initialized with the following arguments an ID (Identification) number, starting x, starting y positions, trajectory or speed, current section which path segment of the course it starts in which defaults at 0.
```
class Droplet():
    def __init__(self, id, x: int = None, y:int = None, trajectory: int = 1, current_section: int = 0) -> None:
        '''Initialize Droplet Object'''
        self.id = id
        self.x = x
        self.y = y
        self.trajectory = trajectory
        self.current_section = current_section
        self.last_detection = None
        self.curve_speed = trajectory
```
2. Update position is logic logic for inferring the position of a given droplet when it is not detected by the Machine Learning Model. The Straights are simple to update the x or y depending on if the Droplet is inside of a straight going up or down. If a droplet is in a curve and not detected then it requires it to run an inference using the quadratic formula provided the start, middle, and end points inside of a Curve. The function then finally calls update_section which checks if the droplet itself moved into a new section and updates the data accordingly. Note the dashed section "-----" will have to be replaced to be used in parallel with the user interface and <b>IF</b> there is a notable tilt to the camera angle of the video that might have the droplet move out of a drawn bounding box.
![Example of a tilted straight](https://github.com/BrannonKLuong/dropshop_ML_overlay/blob/main/img_assets/tilted_straight.PNG)

If the camera angle is too far tilted then the drawn boxes must be larger to accommodate the whole curve as well as considering the y-axis movement.
```
    def update_position(self, course: Path, droplet) -> (int, int):
        segment = course.segments_in_order[self.current_section]
        direction_x, direction_y = segment.direction
        if isinstance(segment, Straight):
            #-----------------------------------------------#
            if direction_x and not direction_y:
                self.x += (self.trajectory * direction_x)
            else:
                self.y += (self.trajectory * direction_y)
            #-----------------------------------------------#   
        else:
            self.x += (self.curve_speed * direction_x)
            self.y = segment.predict_y(self.x)
        self.update_section(course, droplet)
        return (self.x, self.y)
```
3. The Update section takes any time that a Droplet moves position and checks if that droplet has moved into a segment along the course. It simply keys into the current segment gets the top left and bottom right (x, y) coordinates, and checks if the droplet is outside of it. If so update the section. The ladder half of the code in the commented "-----" has some implementation attempting to carry over data. This portion should be able to be removed entirely since the algorithm uses a global data structure. The idea with the previous design and this implementation was to have the droplets stored in the current segment and the next segment. The reason was because there was no easy way to transfer the data without it being lost. Ideally, the data would be seen in between two segments at the perfect time and update the data. Removing it and adding it was rarely perfect so the droplet would simply appear in the next segment, remove itself, then be lost before it's updated into the next segment. 

```
    def update_section(self, course: Path, droplet) -> None:
        segment = course.segments_in_order[self.current_section]
        left, right, top, bot = segment.top_left[0], segment.bottom_right[0], segment.top_left[1], segment.bottom_right[1]
        if self.x < left or self.x > right or self.y < top or self.y > bot:
            if self.current_section < len(course.segments_in_order):
                course.segments_in_order[self.current_section].remove_droplet(droplet)
                self.current_section += 1
                course.segments_in_order[self.current_section].add_droplet(droplet)
               #----------------------------------------------------------------------------#
                if self.current_section + 1 < len(course.segments_in_order):
                    course.segments_in_order[self.current_section + 1].add_droplet(droplet)
               #----------------------------------------------------------------------------#
```
4. Update Last Seen attempts to dynamically update the average speed of a droplet. There are a handful of complexities to consider while designing this portion of the algorithm. The logic for Straight uses the difference between where it was last seen either on an x or a y. The difference in distance/time passed = new speed. 
## Note Dec 14, 2023, 8:19 PM Y logic for a trajectory is not present, needs to be added. Also, it only slows down towards the center but doesn't speed it back up as it leaves.
The curve speed is updated dynamically as well but utilizes a percentage threshold based on the proximity to the center of the curve. The total length of the curve is denoted by its width in x and proximity to the center by any given x value to the center. Since droplets slow down in curves, the algorithm attempts to replicate this process by reducing the speed as it approaches the center of the curve. This prevents the Droplet from exponentially leaving the Curve since it is a quadratic equation and the threshold here is hard coded as 0.3 preventing it from coming to a complete stop. <b>Future implementations</b> will need to find a better way to handle it to prevent it from coming to a complete stop. The following example shows the blue droplet traverses along the path to the start, middle, and end points. As it approaches the start it should be throttled by percentage and increases as it moves further away. 

## This implementation is not complete, IP as of 12/22/2023

![example of trajectory being throttled by percentage calculated by proximity to center](https://github.com/BrannonKLuong/dropshop_ML_overlay/blob/main/img_assets/curve_speed.png)

```
    def update_last_seen(self, mid : (int, int), t : int, x_y_map: {(int, int): Path}, speed_threshold : int) -> None:
        self.x = mid[0]
        self.y = mid[1]

        if not self.last_detection:
            self.last_detection = (mid, t)
            return
        else:
            if isinstance(x_y_map[mid], Straight):
                last_x, curr_x, last_t = self.last_detection[0][0], mid[0], self.last_detection[1]
                if t != last_t: #This line prevents Zero Division Error
                    new_trajectory =  max((last_x - curr_x), (curr_x - last_x))//max((last_t - t), (t - last_t))
                    if new_trajectory and new_trajectory <= speed_threshold:
                        self.trajectory = new_trajectory
            else:
                current_curve = x_y_map[mid]
                middle_curve_x = current_curve.mid[0]
                start_x, end_x = current_curve.start[0], current_curve.end[0]
                total_length = abs((start_x - end_x))
                proximity_to_center = abs(middle_curve_x - self.x)
                if proximity_to_center/total_length * self.curve_speed >= 0.3 # <--- Hard Coded Speed Threshold: 
                    self.curve_speed *= proximity_to_center/total_length 
            self.last_detection = (mid, t)
```
## Step 4: The Path Object
1. The Python Path Object has an array that holds the Path Segments in order. The add droplets to queues is an attempt to implement a local data structure to each segment to hold the Droplets.
```
class Path():
    def __init__(self) -> None:
        self.segments_in_order = []
    
    def add_segment(self, new_segment) -> None:
        self.segments_in_order.append(new_segment)
    #--------------------------------------------------------------------------------------"
    def add_droplet_to_queues(self, droplet) -> None:
        length = len(self.segments_in_order)
        if length > 1 and droplet.current_section + 1 < length:
            self.segments_in_order[droplet.current_section + 1].add_droplet(droplet)
        self.segments_in_order[droplet.current_section].add_droplet(droplet)
    #--------------------------------------------------------------------------------------"
```
## Step 5: The Straight Object
1. Straight Object is initialized with top left points, bottom right point, direction, and the attempt to localize the data structure. The variable top right might be required if the camera tilt previously mentioned becomes a significant problem to address.

```
class Straight():
    def __init__(self, point1: (int, int), point2: (int, int), direction: int) -> None:
        self.top_left = point1
        self.bottom_right = point2
        self.direction = direction
        self.queue = set()
        #self.top_right = (460, 45) # Will have to be a passed in argument once Interface is integrated
    #------------------------------------------------------"
    def add_droplet(self, droplet: Droplet) -> None:
        self.queue.add(droplet)
    
    def remove_droplet(self, droplet: Droplet) -> None:
        self.queue.remove(droplet)
    #------------------------------------------------------"
```
## Step 6: The Curve Object
1. The Python Curve Object is initialized with a top left point, bottom right point, and direction. It does require an addition of the start, middle, and end. The code can be modified to accept it as initializing arguments. The variable quadratic_coef is the a, b, c in f(x) = ax<sup>2</sup> + bx + c given start, middle, and end. The functions get_quadratic find these coefficients and predict_y passes in an x to calculate the respective y value.

```
class Curve():
    def __init__(self, point1: (int, int), point2: (int, int), direction: int) -> None:
        self.top_left = point1
        self.bottom_right = point2
        self.direction = direction 
        self.start = None
        self.mid = None
        self.end = None
        self.queue = set()
        self.quadratic_coef = None #Holds a, b, c coefficients of quadratic formula

    def add_droplet(self, droplet: Droplet) -> None:
        '''Add a droplet to the queue'''
        self.queue.add(droplet)
    
    def remove_droplet(self, droplet: Droplet) -> None:
        '''Remove a droplet to the queue'''
        self.queue.remove(droplet)
    
    def add_sme(self, s: (int, int), m: (int, int), e: (int, int)) -> None:
        self.start = s
        self.mid = m
        self.end = e
        self.quadratic_coef = self.get_quadratic(s, m, e)
    
    def get_quadratic(self, s: (int, int), m: (int, int), e: (int, int)) -> (int, int, int):
        x_1 = s[0]
        x_2 = m[0]
        x_3 = e[0]
        y_1 = s[1]
        y_2 = m[1]
        y_3 = e[1]

        a = y_1/((x_1-x_2)*(x_1-x_3)) + y_2/((x_2-x_1)*(x_2-x_3)) + y_3/((x_3-x_1)*(x_3-x_2))

        b = (-y_1*(x_2+x_3)/((x_1-x_2)*(x_1-x_3))
            -y_2*(x_1+x_3)/((x_2-x_1)*(x_2-x_3))
            -y_3*(x_1+x_2)/((x_3-x_1)*(x_3-x_2)))

        c = (y_1*x_2*x_3/((x_1-x_2)*(x_1-x_3))
            +y_2*x_1*x_3/((x_2-x_1)*(x_2-x_3))
            +y_3*x_1*x_2/((x_3-x_1)*(x_3-x_2)))
        return a,b,c

    def predict_y(self, x: int) -> int:
        a, b, c = self.quadratic_coef
        return a * (x ** 2) + b * x + c
```
## Step 7: Future Implementations
The Future of this current implementation should consist of a handful of things. 
First and foremost would be the removal of the localized data set logic denoted above in # ----- #. It is not all of it but remnants still exist. After this transition to a better data structure to reduce the overall run time to compare detections. I speculate that it is possible to use a queue in the following way. Initialize a global queue that can be accessed anywhere. The queue might need to be a priority queue weighted by what segment or how far the droplet is in the course. For example, if a droplet is closer to the end of the course then it should be further in the queue. The initial reason I didn't use a queue was to avoid the fact that we don't know exactly in what order a dispenser would fire and initialize a new droplet. However upon further thought it should be possible to initialize a new droplet at a (x, y) of a dispenser. Do an O(N) insertion by checking every pair of droplets to see if the droplet can be fitted in between two droplets. It's difficult to calculate whether a droplet is in between two given the difference in (x, y) can be in any direction. However, I think there are some cases we could check for we would traverse FIFO from the end of the course to the start of the course.
#### Case 1: There are no droplets in the queue add the new Droplets to the Queue
#### Case 2: There is exactly one Droplet depending on the segment and what direction it's heading either the new droplet is in front or before it
#### Case 3: The new droplet lands in between 2 Droplets in the same segment. Use the direction of that segment to see which one is further down the course. Place it in between the droplet
#### Case 4: The new droplet d<sub>new</sub> is inserted between two Droplets. One Droplet d<sub>0</sub> in the segment and one outside in the next segment d<sub>1</sub>. The segment weights must be as follows: d<sub>0</sub> <  d<sub>new</sub>  < d<sub>1</sub>.

#### Case 5: The new droplet d<sub>new</sub> is inserted between two Droplets. One Droplet d<sub>0</sub> in the segment and one outside in the previous segment d<sub>1</sub>. The segment weights must be as follows: d<sub>1</sub> <  d<sub>new</sub>  < d<sub>0</sub>.

#### Case 6: The new droplet d<sub>new</sub> is inserted between two Droplets. One Droplet in the segment before d<sub>0</sub> and one in the segment after d<sub>1</sub>. The segment weights must be as follows: d<sub>0</sub> <  d<sub>new</sub>  < d<sub>1</sub>.

If the insertion implementation is successful then we have what is a sorted list of droplets. Ordered in FIFO. Since the list is ordered we could then use a Binary Search Implementation where the detection is first compared to the middle segment. If it is closer to the start throw out the half closer to the end and vice versa. If this is achieved and accurately accomplished in most cases we can bring down the worst-case run time to be <b>O(nlogn)</b>

The second would be incorporating the usage of a live camera. 
The third would be incorporating the pneumatic actuators of the chip itself to replace the hard-coded droplets.
The fourth would be incorporating any other additional hardware to work together with the chip itself. This step will be the most difficult since the Software will have to be designed to work with the hardware and algorithm.
The fifth step of improving hardware might need to be step 3 depending on if the current hardware can run the algorithm live.
The sixth and last priority would be to improve the logic of dynamically updating the speed because a worst-case scenario would be user input to adjust the speed.
