import cv2
from ultralytics import YOLO
from roboflow import Roboflow
import supervision as sv
import sys, os
import math
import time
import math

class Path():
    def __init__(self) -> None:
        self.segments_in_order = []
        "This dictionary maps every segment to an idx"
        self.segments_index_map = {} #Maps a segment to it's (index, distance traveled)
        self.idx = 0
    
    def add_segment(self, new_segment) -> None:
        self.segments_index_map[new_segment] = (self.idx, self.distance_traversed())
        self.idx += 1
        "This assigns the index"
        self.segments_in_order.append(new_segment)
    
    def distance_traversed(self):
        if not self.idx:
            return 0
        else:
            distance_traveled = 0
            for i in range(self.idx):
                seg = self.segments_in_order[i]
                if isinstance(seg, Straight):
                    if seg.direction[0]: #if it's going left or right
                        distance_traveled += seg.bottom_right[0] - seg.top_left[0]
                    else: #If it's going up or down
                        distance_traveled += seg.bottom_right[1] - seg.top_left[1]
                else:
                    distance_traveled += get_distance(seg.start, seg.end)
            return distance_traveled
                            
class Droplet():
    def __init__(self, id, x: int = None, y:int = None, trajectory: int = 1, current_section: int = 0) -> None:
        self.id = id
        self.x = x
        self.y = y
        self.trajectory = trajectory
        self.current_section = current_section
        self.last_detection = None
        self.curve_speed = trajectory

    def update_position(self, course: Path, map_of_segments) -> (int, int):
        segment = course.segments_in_order[self.current_section]
        direction_x, direction_y = segment.direction
        if isinstance(segment, Straight):
            # self.x += (self.trajectory * direction_x)
            # slope = (segment.top_left[1] - segment.top_right[1])/(segment.top_left[0] - segment.top_right[0]) #ideally most  cases slope is 0
            # slope = 0 
            # self.y += slope
            if direction_x and not direction_y:
                self.x += (self.trajectory * direction_x)
            else:
                self.y += (self.trajectory * direction_y)
        else:
            self.x += (self.curve_speed * direction_x)
            self.y = segment.predict_y(self.x)
        self.update_section(course, map_of_segments)
        return (self.x, self.y)
    
    def update_section(self, course: Path, map_of_segments) -> None:
        segment = course.segments_in_order[self.current_section]
        left, right, top, bot = segment.top_left[0], segment.bottom_right[0], segment.top_left[1], segment.bottom_right[1]
        if self.x < left or self.x > right or self.y < top or self.y > bot:
            if self.current_section <= len(course.segments_in_order):
                # This new section uses the index of the segment it was detected in as opposed to incrementing it
                # Implemented this because in the video when the dispensers are fired there's a chance the droplets move backwards up the channel
                try:
                    self.current_section = course.segments_index_map[map_of_segments[(self.x, self.y)]][0]
                except:
                    # Adding this clause in the case the detection is outside of the course if the box is not perfect it will remain what the current section of the droplet is
                    # Might have to assign it to be the segment closest to it.
                    return
    
    def update_last_seen(self, mid : (int, int), t : int, x_y_map: {(int, int): Path}, speed_threshold : int) -> None:
        self.x = mid[0]
        self.y = mid[1]
        if not self.last_detection:
            self.last_detection = (mid, t)
            return
        else:
            if isinstance(x_y_map[mid], Straight):
                direction_x, direction_y = x_y_map[mid].direction
                if direction_x and not direction_y:
                    last_x, curr_x, last_t = self.last_detection[0][0], mid[0], self.last_detection[1]
                    if t != last_t: #This line prevents Zero Division Error
                        new_trajectory =  max((last_x - curr_x), (curr_x - last_x))//max((last_t - t), (t - last_t))
                        if new_trajectory and new_trajectory <= speed_threshold:
                            self.trajectory = new_trajectory
                else:
                    last_y, curr_y, last_t = self.last_detection[0][1], mid[1], self.last_detection[1]
                    if t != last_t: #This line prevents Zero Division Error
                        new_trajectory =  max((last_y - curr_y), (curr_y - last_y))//max((last_t - t), (t - last_t))
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
                          
class Straight():
    def __init__(self, point1: (int, int), point2: (int, int), direction: int) -> None:
        self.top_left = point1
        self.bottom_right = point2
        self.direction = direction
        
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
        '''Given an integer x return the respective y value from the quadratic formula'''
        a, b, c = self.quadratic_coef
        return a * (x ** 2) + b * x + c
    
def find_closest_droplet(drops_to_consider: {Droplet}, mid:(int, int), course, x_y_map) -> Droplet:
    '''As of right now the algorithm to find the closest droplet is Brute Force O(n^2)
    Designing a Recursive/Iterative Algorithm'''
    acceptable_distance = 5 #Some arbitrarily chosen acceptable distance 
    l, r = 0, len(drops_to_consider) - 1
    print([drop.id for drop in drops_to_consider]) #12/26/2023 6:37pm the current issue is that when reviewing the code the frame timing for which
    #The droplets are initialized are in a different order. The algorithm assumes droplets are sorted in distance travels. so right now at frame 
    # 157 droplets are in order of [1, 2, 3, 4] when it should be [1, 2, 4, 3] where the number is their order inserted and the ordering the distance in 
    # the course
    arr = drops_to_consider
    while l <= r:
        if len(arr) == 1:
            return arr[0]
        if l == r: #If pointers pointing to the same droplet return it since the rest of the array has been ignored
            return arr[l]   
        m = (l + r)//2
        m_d = arr[m] #m_d is middle droplet the naming is trying to not reassign the variable mid
        print()
        print(f"(Left, Right): {l, r}")
        print(f"Left: {[drop.id for drop in arr[:m]]} Right:{[drop.id for drop in arr[m:]]}.")   
        calc_dist = get_distance((m_d.x, m_d.y), mid)   
        # print(f"(Left, Right): {l, r}")
        # print(f"Calculated Distant from Detection to Current Droplet {m_d.id}: {calc_dist}")  
        if calc_dist <= acceptable_distance: #If the detection is within a reasonable detection to a droplet assume that's the closest and return it
            return m_d
        else:
            #Otherwise compare it's distance from the left most droplet and right most droplet
            
            #This isn't adding the droplet distance itself yet
            left_distance = determine_total_distance_traveled((arr[l].x, arr[l].y), course.segments_in_order[arr[l].current_section], course)
            detection_distance = determine_total_distance_traveled(mid, x_y_map[mid], course)
            right_distance = determine_total_distance_traveled((arr[r].x, arr[r].y),course.segments_in_order[arr[r].current_section], course)
            right_difference_detection = abs(right_distance - detection_distance)
            left_difference_detection = abs(detection_distance - left_distance)
            print(f"Detection Coordinate: {mid}")
            print(f"Distance Detection Traveled: {detection_distance}")
            print(f"Distance Left Droplet Traveled: {left_distance}   Difference: {left_difference_detection}")
            print(f"Distance Right Droplet Traveled: {right_distance}  Difference: {right_difference_detection}")
            if right_difference_detection <= acceptable_distance:
                #If right edge is acceptable close return it
                return arr[r]
            if  left_difference_detection  <= acceptable_distance:
                #if left edge is acceptably close return it
                return arr[l]
            if len(arr) == 2:
                #If length is only two then return the closer of the two
                if right_difference_detection < left_difference_detection:
                    return arr[r]
                else:
                    return arr[l]
            else:
                #Otherwise Binary Search
                if right_difference_detection < left_difference_detection: 
                    #If the detection is closer to the right most 
                    #Ignore the left half
                    l = m + 1
                else:
                    # Other wise ignore the right half.
                    r = m - 1
                print(f"New (Left, Right): {l, r}")
        # Otherwise compare the distance between left and right
       
        

def determine_total_distance_traveled(coordinate, curr_seg, course): #Coordinate can be droplet coordinate or 
    #Now let's get distance already traveled
    if not coordinate:
        return 0
    distanced_traveled = course.segments_index_map[curr_seg][1] #if n - 1 segments have already been traversed add the flat sum of n segments to the distanced traveled
    #Now calculate the distanced traveled in the current nth segment
    #Remember the grid is inverted 0, 0 is the top left so we can use math.abs to get the absolute value of the distanced traveled since the start will be on the right, traveling left the det x should be < than the right most x
    if isinstance(curr_seg, Straight): #if it's a straight
        horiz, vert = curr_seg.direction #Break the tuple into the horizontal, vertical direction
        if horiz and not vert: #If traveling horizontally and left <------------------------------- IMPORTANT: The Direction Tuple is in format of (1, 0) (0, 1) (-1, 0) (0, -1) denoting right, down, left, up
            if horiz == -1:
                #Add the distance from the detection's x to the right most x of the segment
                right_most_x = curr_seg.bottom_right[0]
                distanced_traveled += right_most_x - coordinate[0]
            else: # if traveling right
                #Add the distance from detection's x to the left msot x of the segment since traveling right det_x > left_most_x
                left_most_x = curr_seg.top_left[0]
                distanced_traveled += coordinate[0] - left_most_x 
        else: # Traveling vertically
            if vert == -1: #If traveling up: Subtract the detection from the bottom most y. Since Detection y < bottom most y
                bottom_most_y = curr_seg.bottom_right[1] 
                distanced_traveled += bottom_most_y - coordinate[1]
            else: #if traveling down, top most y < det y
                top_most_y = curr_seg.top_left[1]
                distanced_traveled += coordinate[1]  - top_most_y
    return distanced_traveled


def load_mac_files():
    model = YOLO("runs/detect/train10/weights/best.pt")
    video_cap = cv2.VideoCapture("droplet_videos/video_data_Rainbow 11-11-22.m4v")
    return model, video_cap

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

def label_course(frame, course) -> None:
    rgb = (0, 255, 0)
    thick = 2
    for segment in course.segments_in_order:
        cv2.rectangle(frame, segment.top_left, segment.bottom_right, rgb, thick)

def label_curves_s_m_e(frame, course) -> None:
    rgb = (0, 0, 200)
    thick = 2
    for segment in course.segments_in_order:
        if isinstance(segment, Curve):
            start_left, start_right = give_me_a_small_box(segment.start)
            mid_left, mid_right = give_me_a_small_box(segment.mid)
            end_left, end_right = give_me_a_small_box(segment.end)

            cv2.rectangle(frame, start_left, start_right, rgb, thick)
            cv2.rectangle(frame, mid_left, mid_right, rgb, thick)
            cv2.rectangle(frame, end_left, end_right, rgb, thick)

def get_droplets_on_screen(t : int, num_droplets: int, drops:{Droplet}, course) -> int:
    if t == 1:
        droplet_1 = Droplet(1, 450, 60, 2)
        insert_and_sort_droplets(droplet_1, drops, course)
        return 1
    elif t == 119:
        droplet_2 = Droplet(2, 315, 60, 1)
        insert_and_sort_droplets(droplet_2, drops, course)
        return 2
    elif t == 153:
        droplet_3 = Droplet(3, 450, 60, 1)
        insert_and_sort_droplets(droplet_3, drops, course)
        return 3
    elif t == 156:
        droplet_4 = Droplet(4, 315, 60, 1)
        insert_and_sort_droplets(droplet_4, drops, course)
        return 4
    elif t == 185:
        droplet_5 = Droplet(5, 450, 60, .5)
        insert_and_sort_droplets(droplet_5, drops, course)
        return 5
    elif t == 222:
        droplet_6 = Droplet(6, 450, 60, .5)
        insert_and_sort_droplets(droplet_6, drops, course)
        return 6
    elif t == 370:
        droplet_7 = Droplet(7, 460, 195, 1, 4)
        insert_and_sort_droplets(droplet_7, drops, course)
        return 7
    elif t == 515:
        droplet_8 = Droplet(8, 315, 195, 1, 4)
        insert_and_sort_droplets(droplet_8, drops, course)
        return 8 
    else:
        return num_droplets
    
def insert_and_sort_droplets(droplet, lst, course):
    if not lst:
        lst.append(droplet)
        return
    
    for i in range(len(lst)):
        # coordinate, curr_seg, course
        insertion_distance = determine_total_distance_traveled((droplet.x, droplet.y), course.segments_in_order[droplet.current_section], course)
        i_th_droplet_travel_distance = determine_total_distance_traveled((lst[i].x, lst[i].y), course.segments_in_order[lst[i].current_section], course)
        if insertion_distance < i_th_droplet_travel_distance:
            lst.insert(i, droplet)
            return
        else:
            try:
                lst.insert(i + 1, droplet)
                return
                #Try to insert at the index in front of the current one
                #If an index error occurs because its at the end of the list then append it to the end
            except IndexError:
                lst.append(droplet)
                return
    return
    
        
def where_droplets_should_start(frame) -> None:
    cv2.rectangle(frame, (445, 55), (455, 65), (255, 0, 0), 2)
    cv2.rectangle(frame, (315, 55), (325, 65), (255, 0, 0), 2)
    cv2.rectangle(frame, (445, 190), (455, 200), (255, 0, 0), 2) 
    cv2.rectangle(frame, (315, 190), (325, 200), (255, 0, 0), 2)
    
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

def get_distance(point1: (int, int), point2: (int, int)) -> float:
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)** 2 + (y2 - y1) ** 2)

def get_mid_point(xone: int, yone: int, xtwo: int, ytwo: int) -> (int, int):
    return ((xone + xtwo)//2, (yone + ytwo)//2)

def give_me_a_small_box(point: (int, int)) -> ((int, int), (int, int)):
    return (int(point[0] - 2), int(point[1] - 2)),(int(point[0] + 2), int(point[1] + 2))

def box_drops(drops: {Droplet}, frame) -> None:
    for drop in drops:
        left_predict, right_predict = give_me_a_small_box((drop.x, drop.y))
        cv2.rectangle(frame, left_predict, right_predict, (100, 0, 0), 4)

def handle_missings(drops: {Droplet}, found: set, map_course: Path, map_of_segments) -> None:
    missing = drops.difference(found)
    for drop in missing:
        drop.update_position(map_course, map_of_segments)
        found.add(drop)

def load_mac_files():
    model = YOLO("runs/detect/train10/weights/best.pt")
    video_cap = cv2.VideoCapture("droplet_videos/video_data_Rainbow 11-11-22.m4v")
    return model, video_cap 

def main(weights_path, video_path):
    all_droplets = []
    course = build_course()
    # print(course.segments_index_map)
    x_y_map = build_x_y_map(course)
    box = sv.BoxAnnotator(text_scale=0.3)
    speed_threshold = 5
    # model, video_cap = load_mac_files()
    model = YOLO(weights_path)
    video_cap = cv2.VideoCapture(video_path)

    if not video_cap.isOpened():
        print("Error: Video file could not be opened.")
        return
    
    t = 0
    droplets_on_screen = 0
    while video_cap.isOpened():
        t += 1 #Increment the time
        print(t)
        ret, frame = video_cap.read()
        # frame = cv2.resize(frame, (680, 480))

        if not ret:
            print("Video ended")
            break

        if t > 0:
            droplets_on_screen = get_droplets_on_screen(t, droplets_on_screen, all_droplets, course)
            result = model.track(frame, tracker="bytetrack.yaml", persist=True)[0]
            numbers_detected = len(result)
            found = set()
            labels = []
            try:
                for data in result.boxes.data.tolist():
                    try:
                        xone, yone, xtwo, ytwo, _, confidence, _ = data
                    except ValueError:
                        print("No Data given by detection")

                    try:
                        '''drops to consider is ideally always the drops in the segment closest to the detection'''
                        mid = get_mid_point(xone, yone, xtwo, ytwo)
                        
                    except KeyError:
                        print("Detection occurred outside of the course. Data: ", data)
                        continue
                    except Exception as e:
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)
                        continue
                    
                    closest_droplet = find_closest_droplet(all_droplets, mid, course, x_y_map)
                    # try:
                    #     print("Closest Droplet: " +  str(closest_droplet.id) + " Current Section: " + str(closest_droplet.current_section))
                    # except:
                    #     continue
                    found.add(closest_droplet)

                    closest_droplet.update_last_seen(mid, t, x_y_map, speed_threshold)
                
                    if x_y_map[mid] != course.segments_in_order[closest_droplet.current_section]:
                        closest_droplet.update_section(course, x_y_map)
                    
                    box_drops(all_droplets, frame)

                    if confidence:
                        labels.append(f"{closest_droplet.id} {confidence:0.2f}")

                if numbers_detected < droplets_on_screen:
                    print("Handling Missing Cases")
                    handle_missings(all_droplets, found, course, x_y_map)

            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)

        where_droplets_should_start(frame)

        detections = sv.Detections.from_ultralytics(result)
        label_course(frame, course) 
        label_curves_s_m_e(frame, course)

        frame = box.annotate(scene=frame, detections=detections, labels = labels)
        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(10) == 27):
            break

if __name__ == '__main__':
    '''Start Time and End Time is a timer to measure run time'''
    start_time = time.perf_counter()
    main("runs/detect/train_rainbow/weights/best.pt", "droplet_videos/video_data_Rainbow 11-11-22.m4v")
    # main("runs/detect/train_rainbow/weights/best.pt", "droplet_videos/20221111_rainbow2_notracking.mp4")
    # build() #Just a test function to isolate portions
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")