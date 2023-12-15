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

#### The solution leverages the simplicity of knowing that the course is consistent. Meaning that the droplets travel across a fixed path. The implementation must also keep in mind that Machine Learning Models do not explicitly have a form of carrying information over to determine consecutive frames. Tracking algorithms such as Kalman's algorithm estimate where an object will be based on its trajectory from previous frames and its current position in current frames. Kalman's filter is ideal for more complex environments. Maintaining this logic the design of this implementation of Dropshop uses Droplet objects to retain as much information to infer the position. Due to the simplicity of the Course lay the following was used to infer the future position of droplets. Initialize a bounding box that shows the direction of the segment of the course and whether the segment is a curve or a straight. A straight, simple in its implementations, modifies the x or y value of a droplet by the direction it's heading. A curve needs to be labeled accurately to utilize the Quadratic equation to predict its traversal along the course's curve. From this point let d<subscript>n</subscript> denote droplets, course to be the entire path, segments be any part of the course, Det<subscript>n</subscript> to denote detections.

1. Pass in weights_path which is a file of weights trained with YoloV8 and a video path to main()
    ```if __name__ == '__main__':
    '''Start Time and End Time is a timer to measure run time'''
    start_time = time.perf_counter()
    main("runs/detect/train10/weights/best.pt", "droplet_videos/video_data_Rainbow 11-11-22.m4v")
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
    ```
2. The main function initializes variables and runs the core logic
all_droplets stores every droplet in the course at any given point.
course is a Path object that stores the segments in order. The Path object has a function to add new segments and add droplets to each nested segment's queue.

    ```class Path():
          def __init__(self) -> None:
              self.segments_in_order = []
          
          def add_segment(self, new_segment) -> None:
              '''Adds a segment of the course into the Paths array'''
              self.segments_in_order.append(new_segment)
      
          def add_droplet_to_queues(self, droplet) -> None:
              '''Adds droplets to the corresponding queue in each segment. If the segment isn't last in the list then
              add it to the correct one and the next one. That way we can infer will it will be and remove it accordingly'''
              length = len(self.segments_in_order)
              if length > 1 and droplet.current_section + 1 < length:
                  self.segments_in_order[droplet.current_section + 1].add_droplet(droplet)
              self.segments_in_order[droplet.current_section].add_droplet(droplet)
              print([drop.id for drop in self.segments_in_order[droplet.current_section].queue])
    ```
    
3. Each Segment is either a Straight or a Curve and each one holds a data structure that helps store using the top left corner point and bottom right-hand corner point. 
Straights are simple only having an add droplet and remove droplet feature with most parameters passed in by the User. 

    class Straight():
          def __init__(self, point1: (int, int), point2: (int, int), direction: int) -> None:
              self.top_left = point1
              self.bottom_right = point2
              self.direction = direction
              self.queue = set()
             
          def add_droplet(self, droplet: Droplet) -> None:
              '''Add a droplet to the queue'''
              self.queue.add(droplet)
          
          def remove_droplet(self, droplet: Droplet) -> None:
              '''Removes a droplet from this segments queue'''
              self.queue.remove(droplet)
              
4. Curves are more complex it needs a corresponding start, middle, and endpoint which calls a quadratic function to solve for a, b, and c in ax^2 + bx + c and a function
predict y that helps infer the location of the droplet

    class Curve():
          def __init__(self, point1: (int, int), point2: (int, int), direction: int) -> None:
              '''Initialize a curve's box and it's direction. Assuming a start, middle, end point are provided.
              Initialize a tuple that holds the coefficients to a quadratic formula (a, b, c) for the respective
              f(x) = ax^2 + bx + c
              '''
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
              '''Adds the start middle end points and gets then uses those points to get the coefficients'''
              self.start = s
              self.mid = m
              self.end = e
              self.quadratic_coef = self.get_quadratic(s, m, e)
          
          def get_quadratic(self, s: (int, int), m: (int, int), e: (int, int)) -> (int, int, int):
              '''Returns a tuple that holds the coefficients to a quadratic formula (a, b, c) for the respective
              f(x) = ax^2 + bx + c '''
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
              '''Given an integer x return the respective y value from the quadratic formula'''
              a, b, c = self.quadratic_coef
              return a * (x ** 2) + b * x + c

5. Every Python object droplet holds its id, (x, y), trajectory, what section it's currently in, the last time it was seen, and a different speed for curves.
Its primary function is
### update_position which updates its position this is the logic in the case it's not detected it updates depending on where it was in the curve.
### update_section When the droplet is detected outside of the section it's currently in move the data over to the next segment and the next segment.
For example, if a droplet was in S0 and saw in S1 add it to S1 and S2. The design choice for this was to allow for a seamless transition in data since there was too much
variability in accuracy this provided a generally confident way that the Droplet would at least be seen in S1 or S2 and update. This makes the algorithm heavily reliant on the model
being accurate to a certain degree.
### update_last_seen which dynamically updates the speed of the droplet using last seen to calculate it

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
    
        def update_position(self, course: Path, droplet) -> (int, int):
            segment = course.segments_in_order[self.current_section]
            direction_x, direction_y = segment.direction
            if isinstance(segment, Straight):
    
                # self.x += (self.trajectory * direction_x)
                # #slope = (segment.top_left[1] - segment.top_right[1])/(segment.top_left[0] - segment.top_right[0]) #ideally most  cases slope is 0
                # slope = 0 
                # self.y += slope
                if direction_x and not direction_y:
                    self.x += (self.trajectory * direction_x)
                else:
                    self.y += (self.trajectory * direction_y)
            else:
                try:
                    try:
                        self.x += (self.curve_speed * direction_x)
                    except AttributeError:
                        print("Occured o nself.x")
                    self.y = segment.predict_y(self.x)
                except AttributeError:
                    print("Occurred here")
            self.update_section(course, droplet)
            return (self.x, self.y)
        
        def update_section(self, course: Path, droplet) -> None:
            segment = course.segments_in_order[self.current_section]
            left, right, top, bot = segment.top_left[0], segment.bottom_right[0], segment.top_left[1], segment.bottom_right[1]
            if self.x < left or self.x > right or self.y < top or self.y > bot:
                if self.current_section < len(course.segments_in_order):
    
                    #Error Probably Occurring Here
                    course.segments_in_order[self.current_section].remove_droplet(droplet)
                    self.current_section += 1
                    course.segments_in_order[self.current_section].add_droplet(droplet)
                    if self.current_section + 1 < len(course.segments_in_order):
                        course.segments_in_order[self.current_section + 1].add_droplet(droplet)
        
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
                    if proximity_to_center/total_length * self.curve_speed >= 0.3: 
                        self.curve_speed *= proximity_to_center/total_length 
                self.last_detection = (mid, t)



<p align="right">(<a href="#readme-top">back to top</a>)</p>
