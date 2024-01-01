# Design and Implementation of a Greedy Binary Search Dropshop Algorithm
## Summary:
This implementation uses Binary Search which assumes the droplets are in a sorted array.
The array is sorted by the order in which droplets are closest to the end. 
The algorithm is greedy and thus takes an average case of O(nlogn) to run.
The worst case scenario is that the algorithm doesn't find a droplet within an acceptable threshold with binary search and runs the Brute Algorithm after it tries binary search.
This run time is O(nlogn + n^2) or O(n(logn + n)) where n^2 dominates and is O(n^2). This readme will only cover the primary implementation logic and highlight the differences from the initial design.

## Sorting the Array of Droplets

The sorting algorithm is determined by the distance in which a droplet travels. 
This distance is calculated by where the droplet is in the current section and summed to the total distance to travel previous segments.
Distance Traveled = (Droplet's position - Segments Start Position) + the sum of every segment before it.
Take for example the following.

![Example of Distance traveled](https://github.com/BrannonKLuong/dropshop_ML_overlay/blob/main/img_assets/distance_traveled_2.png)

Imagine That the droplet is in segment 5. We can safely assume that the sum of seg1 + seg 2 + seg 3 + seg 4 amounts to the total distance traveled. To get the remainder we take the droplet's exact position - segment 5's starting position denoted with the red dot in the corner.
The following function in the Path class when initialized maps each segment to the sum of previous segments so segment 1 is mapped to the sum of segment 0. Segment 2 is mapped to the distance of (segment 1 distance + segment 0 distance), ... etc. 

```
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
                    #If it's a curve get the entire arc length, 
                    #Pass in the Curve, the starting x, end y which is the entire duration of the curve
                    distance_traveled += get_arc_length(seg, seg.start[0], seg.end[0])
            return distance_traveled
```

The updated Vertical Aspect of update_last_seen() still requires an efficient way of dynamically changing the speeds. As well as implement a way to increase the speed of the droplet as it leaves curves.

```
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
```

## The Bulk of the Changes in the Find Closest Droplet Function Logic
The core of the logic is replaced with a binary search applied to a sorted array of droplets which is sorted by the sum of the distance traveled.   
Find Closest Droplet now has a hard-coded parameter called acceptable_distance which is an arbitrarily chosen value that determines whether or not a droplet is close enough to detection to be sufficiently returned/determined as the closest droplet. Inherently the algorithm follows the fundamentals of binary search and uses the distance values for comparisons. L is the left index and R is the right index. 

```
def find_closest_droplet(arr, mid:(int, int), course, x_y_map) -> Droplet:
    '''Iterative Binary Search Algorithm for DropShop using Distance as a Metric'''

    acceptable_distance = 10 #Some arbitrarily chosen acceptable distance

    l, r = 0, len(arr) - 1
    while l <= r:
        if len(arr) == 1:
            return arr[0]
```

#### Highlighting A Crucial Point
Recall that this algorithm is a Greedy Algorithm primarily because there is too much variability and never guarantees perfection in binary search. For example, there are times when the desired droplet is in the right half but the leftmost droplet is closest and will disregard it. To compensate for this factor in the case that the algorithm never finds an acceptable droplet within the desired threshold the algorithm will brute-force search for it. However, these cases are far less frequent and the average run time is O(nlogn).

The following conditional addresses this crucial point. If the entirety of the binary search is completed then eventually L will equal R. This means the entire array was traversed through a binary search and no droplet was detected
within the acceptable bounds. This means the algorithm didn't find the target and Brute Force searches through the original method of finding a droplet to find the closest droplet. 
            
```
        if l == r:
            return brute_force(arr, mid)
```
m is the middle index and m_d is the middle droplet object. The naming of m_d is similar to mid which is important to note so that there's no accidental reassignment. Calc_dist is the distance between the detection and droplet 

```
        m = (l + r)//2
        m_d = arr[m] #m_d is the middle droplet, being careful not to reassign mid

        calc_dist = get_distance((m_d.x, m_d.y), mid)
```

Print Statements can be uncommented to visualize how the algorithm is running.

```
        # print()
        # print(f"(Left, Middle, Right): {l, m, r} Left: {[drop.id for drop in arr[l:m]]} Right:{[drop.id for drop in arr[m:r + 1]]}.")    
        # print(f"Detection Coordinate: {mid}")
        # print(f"Every Droplet's Information: {[(drop.id, drop.x, drop.y) for drop in arr]}")
```
Left_distance is the distance of the left-most droplet. Detection_distance is how far along the detection is along the course. Right_distance is the distance of the right-most droplet.

```
        left_distance = determine_total_distance_traveled((arr[l].x, arr[l].y), course.segments_in_order[arr[l].current_section], course) 
        detection_distance = determine_total_distance_traveled(mid, x_y_map[mid], course) # Distance of the detection
        right_distance = determine_total_distance_traveled((arr[r].x, arr[r].y),course.segments_in_order[arr[r].current_section], course)
```
Right_difference_detection is the distance of the rightmost droplet - the distance of the detection.
Left Distance is the difference between the detection and the leftmost droplet. 

```
        right_difference_detection = abs(right_distance - detection_distance)
        left_difference_detection = abs(detection_distance - left_distance)
```

If the detection is within a reasonable detection to a droplet assume that's the closest and return it. This feature was added to replace the arr[m] == target: return function of a normal binary search. Since the algorithm needs something to classify as the target or known as the target a distance threshold should suffice. Imagine a radius around a droplet saying any detection within this radius is sufficiently close therefore that detection is associated with that droplet. Consider the following image:

![Acceptable Range Example](https://github.com/BrannonKLuong/dropshop_ML_overlay/blob/main/img_assets/dropshop_acceptable_range_ex.png)

In this let the acceptable range be arbitrarily chosen in this case 5. Detection 0, d0, is out of bounds at a range of 6 and disregarded. Detection 1, d1, is within the range at a distance of 3 therefore it is returned.

#### Future Implementations: Optimize the acceptable range threshold that won't under/overfit the binary search algorithm.

```
        if calc_dist <= acceptable_distance:
            return m_d
```

If the rightmost droplet or leftmost droplet is within the acceptable range return that droplet, otherwise binary search.

```
        if right_difference_detection <= acceptable_distance:
            return arr[r]
        if  left_difference_detection <= acceptable_distance:
            return arr[l]
        elif right_difference_detection < left_difference_detection:
            l = m + 1
        else:
            r = m - 1
```

#### The Math to Determine Total Distance
The main challenge for calculating the sum is for curves. The algorithm uses the arc length formula.

```
def determine_total_distance_traveled(coordinate, curr_seg, course): #Coordinate can be droplet coordinate or 
    if not coordinate:
        return 0
    coord_x, coord_y = coordinate[0], coordinate[1]
```

Distanced Traveled is the sum of all previous segments' distances. n - 1 segments have already been traversed add the flat sum of n - 1 segments to the distance traveled. When initializing x_y_map and mapping every (x, y) to a segment the algorithm also maps the distances. For example, segment 0 will be mapped to 0. segment 1 is mapped to the sum of 0's distance. Segment 2 is the sum of segment 1 and 0, etc. Then calculate how far the droplet/detection has traveled within its specific segment. 

**Note**: Remember the grid is inverted 0, 0 is the top left so we can use math.abs to get the absolute value of the distance traveled since the start will be on the right, traveling left the det x should be < than the right-most x

```
    distanced_traveled = course.segments_index_map[curr_seg][1]
  
    if isinstance(curr_seg, Straight): #if it's a straight
        horiz, vert = curr_seg.direction #Break the tuple into the horizontal, vertical direction
        if horiz and not vert: #If traveling horizontally IMPORTANT: The Direction Tuple is in format of (1, 0) (0, 1) (-1, 0) (0, -1) denoting right, down, left, up
            if horiz == -1:
                #Add the distance from the detection's x to the right most x of the segment
                right_most_x = curr_seg.bottom_right[0]
                distanced_traveled += right_most_x - coord_x
            else: # if traveling right
                #Add the distance from detection's x to the left msot x of the segment since traveling right det_x > left_most_x
                left_most_x = curr_seg.top_left[0]
                distanced_traveled += coord_x - left_most_x 
        else: # Traveling vertically
            if vert == -1: #If traveling up: Subtract the detection from the bottom most y. Since Detection y < bottom most y
                bottom_most_y = curr_seg.bottom_right[1] 
                distanced_traveled += bottom_most_y - coord_y
            else: #if traveling down, top most y < det y
                top_most_y = curr_seg.top_left[1]
                distanced_traveled += coord_y - top_most_y
    else: #If it's a curve
```

The algorithm measures the distance traveled along a quadratic curve using an arc length for quadratic formulas. 

```
        start_pt = curr_seg.start
        distanced_traveled += get_arc_length(curr_seg, start_pt[0], coord_x) #Get the arc length traversed across the given intervals 
    return round(distanced_traveled, 2)
```

The math of the get arc length takes the quadratic coefficients and writes the derivative. Since the algorithm has the benefit of knowing the curves always quadratic in nature the algorithm can arithmetically calculate the integral of the arc length. 

### Important Note: The coefficients have to be rounded to avoid Python bit inaccuracies when handling large numbers. 

```
def get_arc_length(curve, interval_1, interval_2):
    '''Take the coefficients from the curve of the quadratic formula.

     a and b have to be rounded or else the issue is that the python can not represent very large numbers accurately after n amount of bits. Rounding  to 8 runs well for me so I'll leave it at that and test it.

The following applies Python's numpy to apply arc length integrals to calculate the distance along a quadratic curve. Quad Integrand is a Python library and a designed function to execute the formula of an "Arc Length Parameterization" or "Arc Length of a Quadratic Curve". Technically the formula is Arc Length Parameterization but the formula denotes the derivative of both x(t) and y(t) but the nature of a quadratic formula you will factor out
    dx from the formula resulting the the sqrt of (1 + dy/dx**2)
    Deratives 1
'''


    a, b, _ = curve.quadratic_coef
    a, b = round(a, 8), round(b, 8)#This number is arbitrarily chosen. 
    
    result, _ = quad(calc_arc_length, interval_1, interval_2, args=(a, b))
    result = abs(round(result, 2)) #Absolute value and round it
    return result #Arbitrarily chosen decimal place to round to can be tested with more or less. Depends on the necessity for Accuracy

def calc_arc_length(x, a, b):
    return np.sqrt(1 + (2*a*x + b)**2)
```

#### Insertion Logic
The Implemented Insertion Logic handles the cases listed in the main readme.

```
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
        else: #if insertion distance is greater than i_th distance meaning droplet_i < insertion
            try:
                next_droplet_travel_distance = determine_total_distance_traveled((lst[i + 1].x, lst[i + 1].y), course.segments_in_order[lst[i + 1].current_section], course)
            except IndexError: #There isn't a droplet infront of the ith droplet meaning the ith droplet is the end of the  list
                lst.append(droplet)
                return
            if insertion_distance < next_droplet_travel_distance:
                #droplet_i < insertion < droplet_i + 1 Note python lst inserts at left of the droplet_i
                lst.insert(i + 1, droplet)
                return
            else:
                continue
    return
```

# Final Notes:
## With the rainbow video the algorithm runs a total of 5750 searches and of those 124 are brute forced while the rest are successfully binary searched. For this video brute force is used 2% of the time.

In the terms of O(n^2) vs O(nlogn). n is the number of droplets. It can be more accurate in the description saying the worst case scenario is O(n * d) where n is the number of droplets and d is the number of detections. Recall, that ideally, the machine learning model accurately detects each droplet. So ideally d == n.
Therefore the core of the old implementation was to check every detection with every droplet O(n * d) since d == n, n * n, or O(n^2). In the Greedy Algorithm, the implementation constantly disregards half of the array so the search is now O(d * log<sub>2</sub>(n)) and since d == n, O(n*log<sub>2</sub>n)) or just O(nlogn). For an average of 4 droplets a frame that would mean the original algorithm would've run 4^4 * 5750 = 92000 searches. Compared to the new algorithm which does binary search 98% but since it tries binary search every frame then the brute force, in this case, the result is of the time 5750(4log(4)) + 0.02(5750) * (4^2) or about 15687 searches.  

The next question is how accurate does binary search has to be before it becomes worse than the brute force case.  
In this case, the formula can be the frames * d(log(d)) + % of failures * frames * d^2 == frames * d^2. Since the values may vary on a case-by-case scenario it's best to plot the graphs on a demos chart. Consider the following graphs.

f(x) = 5750x^2  

f(d) = 5750 * (d * log<sub>2</sub>(d) + 0.2 * 5750 * d^2

