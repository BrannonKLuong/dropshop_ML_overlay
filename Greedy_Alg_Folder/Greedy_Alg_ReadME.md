# Design and Implementation of a Greedy Binary Search Dropshop Algorithm
## Summary:
This implementation uses Binary Search assuming the droplets are in a sorted array.
The array is sorted by the order in which droplets are closest to the end. The algorithm is greedy and thus takes
an average case of O(nlogn) to run. Worst case scenario the algorithm doesn't find a droplet with binary search and runs
O(nlogn + n^2) where n^2 dominates and is O(n^2). This readme will only cover the primary logic on how to implement it and highlight the difference from the initial design.

#### Thought Process:
Let's imagine what we know. we know we have the segments in order in an array of this format
the question is at any given (x, y) detection how would I know whether it comes before or after another droplet in the queue already

let's imagine a Randomly Generated Array of Distinct Numbers from 1 - 7:  [3, 6, 2, 7, 5, 1, 4]
let's imagine the droplets are actually like this in the course
the reason i propose generating a random selection is
similarly to how we would sort the droplets randomly given to us
it's also sorted in the fact we know where it's going and where the droplets are going to be inserted at
since they're no definitive singular section as in x, y
we don't know hwo to sort the droplets exactly
what we do know is
the order of the segments knowing
that the segments in light blue are ordered
so we have a way of sorting it
so let's take this random list again
[3, 6, 2, 7, 5, 1, 4]
we can now partition it in this way knowing that the order would roughly be in the form [1], [3, 2], [4], [6, 7, 5] 

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
