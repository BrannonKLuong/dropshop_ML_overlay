# Design and Implementation of a Greedy Binary Search Dropshop Algorithm
## Summary:
This implementation uses Binary Search assuming the droplets are in a sorted array.
The array is sorted by the order in which droplets are closest to the end. The algorithm is greedy and thus takes
an average case of O(nlogn) to run. Worst case scenario the algorithm doesn't find a droplet with binary search and runs
O(nlogn + n^2) where n^2 dominates and is O(n^2). This readme will only cover the primary logic on how to implement it and highlight the difference from the initial design

