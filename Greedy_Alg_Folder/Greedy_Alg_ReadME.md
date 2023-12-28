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
The following function in Path class when initialized maps each segment to the sum of previous segments so segment 1 is mapped to the sum of segment 0. Segment 2 is mapped to the distance of (segment 1 distance + segment 0 distance). 

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

#### Droplet's Update Last Seen Function has been resolved from previous version

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
