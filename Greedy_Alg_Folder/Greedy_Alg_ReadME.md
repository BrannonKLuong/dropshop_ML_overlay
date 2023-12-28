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
