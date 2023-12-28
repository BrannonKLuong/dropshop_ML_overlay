# Design and Implementation of a Greedy Binary Search Dropshop Algorithm
## Summary:
This implementation uses Binary Search assuming the droplets are in a sorted array.
The array is sorted by the order in which droplets are closest to the end. 
The algorithm is greedy and thus takes an average case of O(nlogn) to run.
Worst case scenario the algorithm doesn't find a droplet within an acceptable threshold with binary search and runs the Brute Algorithm after it tries binary search.
This run time is  O(nlogn + n^2) where n^2 dominates and is O(n^2). This readme will only cover the primary implementation logic and highlight the differences from the initial design.

## Sorting the Array of Droplets
The sorting algorithm is determined by the distance in which a droplet travels. 
This distance is calculated by where the droplet is in the current section and summed to the total distance to travel previous segments.
Distance Traveled = (Droplet's position - Segments Start Position) + the sum of every segment before it.
Take for example the following.

![Example of Distance traveled](https://github.com/BrannonKLuong/dropshop_ML_overlay/blob/main/img_assets/distance_traveled_2.png)

Imagine That the droplet is in segment 5. We can safely assume that the sum of seg1 + seg 2 + seg 3 + seg 4 amounts to the total distance traveled. To get the remainder we take the droplet's exact position - segment 5's starting position denoted with the red dot in the corner.
The following function in the Path class when initialized maps each segment to the sum of previous segments so segment 1 is mapped to the sum of segment 0. Segment 2 is mapped to the distance of (segment 1 distance + segment 0 distance). 

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

#### Droplet's Update Last Seen Function has been resolved from the previous version

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

## The Bulk of the Changes in the Find Closest Droplet Logic
The core of the logic is replaced with a binary search applied to a sorted array of droplets which is sorted by the sum of the distance traveled.
Find Closest Droplet now has a hard-coded parameter  called acceptable_distance which is an arbitrarily chosen value that determines whether or not a droplet is close enough to detection to be sufficiently returned/determined as the closest droplet. Inherently the algorithm follows the fundamentals of binary search and uses the distance values for comparisons.
```
def find_closest_droplet(arr, mid:(int, int), course, x_y_map) -> Droplet:
    '''As of right now the algorithm to find the closest droplet is Brute Force O(n^2)
    Designing a Iterative Binary Search Algorithm'''
    acceptable_distance = 10 #Some arbitrarily chosen acceptable distance 
    l, r = 0, len(arr) - 1

    while l <= r:
        if len(arr) == 1:
            return arr[0]
        if l == r: #If pointers pointing to the same droplet return it since the rest of the array has been ignored and the array was traversed through binary search meaning\
            return brute_force(arr, mid)

        m = (l + r)//2
        m_d = arr[m] #m_d is middle droplet the naming is trying to not reassign the variable mid

        calc_dist = get_distance((m_d.x, m_d.y), mid)
        # print()
        # print(f"(Left, Middle, Right): {l, m, r} Left: {[drop.id for drop in arr[l:m]]} Right:{[drop.id for drop in arr[m:r + 1]]}.")    
        # print(f"Detection Coordinate: {mid}")
        # print(f"Every Droplet's Information: {[(drop.id, drop.x, drop.y) for drop in arr]}")
        
        left_distance = determine_total_distance_traveled((arr[l].x, arr[l].y), course.segments_in_order[arr[l].current_section], course)
        detection_distance = determine_total_distance_traveled(mid, x_y_map[mid], course)
        right_distance = determine_total_distance_traveled((arr[r].x, arr[r].y),course.segments_in_order[arr[r].current_section], course)
        
        right_difference_detection = abs(right_distance - detection_distance)
        left_difference_detection = abs(detection_distance - left_distance)
        if calc_dist <= acceptable_distance: #If the detection is within a reasonable detection to a droplet assume that's the closest and return it
            return m_d
        if right_difference_detection <= acceptable_distance:
            #If right edge is within the acceptable range return it
            return arr[r]
        if  left_difference_detection <= acceptable_distance:
            #if left edge is within the acceptable range return it
            return arr[l]
        elif right_difference_detection < left_difference_detection:
            l = m + 1
        else:
            r = m - 1
    #Can add a section that makes the algorithm worst case O(n^2) and average nlogn if the algorithm falls here that means no droplet was ever returned so we can just
    #Have it check it N^2 wise by comparing to every droplet. The since of the greedy algorithm attempts to avoid doing this in most cases drastically improving average run time
    #Ideally we never have to add the comparing every droplet to every detection portion. But given the nature of the inconsistencies it is possible this would be necessary
```

#### Highlighting A Crucial Point
Recall that this algorithm is a Greedy Algorithm primarily because there is too much variability and never guarantees perfection in the way of binary search. For example, there are times when the desired droplet is in the right half but the leftmost droplet is closest and will disregard it. To compensate for this factor in the case that the algorithm never finds an acceptable droplet within the desired threshold the algorithm will brute-force search for it. However, these cases are far less frequent and the average run time is O(nlogn).


