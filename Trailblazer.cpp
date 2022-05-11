/*
 * File: Trailblazer.cpp
 * --------------------------
 * Name: Tom Welch
 * Section leader: Benson
 * The code in this file finds paths from one vertex to another in a graph through
 * four seperate algorithms. The first algorithm is a BFS which finds the path with the
 * least number of edges. The second is dikjstras algorithm which finds the shortes path
 * with edgeweights taken into consideraiton. The third is A* which is a more efficient version
 * of the second by using a heuristic. Finally the fourth algorithm finds an alternative route that
 * the fastest route that differs by a factor of 0.2.
 */

#include "Trailblazer.h"
#include "queue.h"
#include "priorityqueue.h"
#include "set.h"
using namespace std;

static const double SUFFICIENT_DIFFERENCE = 0.2;

//declares prototypes
Path helper(const RoadGraph& graph, RoadNode* start, RoadNode* end, bool useHeuristic, RoadEdge *e, double &length);
double heuristic(const RoadGraph& graph, RoadNode* start, RoadNode* end);
double difference(Path& a, Path& b);

/*This function visits starts at the starting vertex and checks all its neighbors
 * for the ending vertex, then all the neighbors neighbors and so on. The path with the least
 * jumps will be found first and thus returned. This funciton accepts the graph of the map, and the
 * start and end vertex and returns the path with the least jumps.
*/
Path breadthFirstSearch(const RoadGraph& graph, RoadNode* start, RoadNode* end) {
    //initializes queue and visited
    Queue<Path> paths;
    Set<RoadNode*> visited;
    visited.add(start);
    Path startP;
    startP.add(start);
    paths.enqueue(startP);
    //loops through paths adding neighbors to the last nodes until either a path is found or not
    while(!paths.isEmpty() && !visited.contains(end)){
       //gets path and last node from queue and adds to visited while highlighting
       Path p = paths.dequeue();
       RoadNode * v = p[p.size() - 1];
       (*v).setColor(Color::GREEN);
       visited.add(v);
       //if path is complete, returns it
       if(v == end){
           return p;
       }
       //searches adds new paths with the last node's unvisited neighbors
       for(RoadNode * r : graph.neighborsOf(v)){
           if(!visited.contains(r)){
               (*r).setColor(Color::YELLOW);
               Path newP = p;
               newP.add(r);
               paths.enqueue(newP);
           }
       }
    }
    return {};
}

/*This function visits starts at the starting vertex and checks all its NEAREST neighbors
 * for the ending vertex. It then iteratively searches the shortest paths from the starting vertex
 * and so on. The path with the shortest path will be found first and thus returned. This funciton
 * accepts the graph of the map, and the start and end vertex and returns the path with the shortest path.
*/
Path dijkstrasAlgorithm(const RoadGraph& graph, RoadNode* start, RoadNode* end) {
    //length is not used in this case but is neccessary for alternative route
    double length;
    //calls function that works for the last three algorithms
    return helper(graph, start, end, false, nullptr, length);
}


/* This funciton is essentially the same as dijkstras algorithm with the exception that
 * in calculating path length it gives higher priority to paths that are in the direction of the end goal
 * by using a heuristic. This function accepts the graph of the map, and the start and end vertex and returns
 * the path with the shortest path in less time that dijkstras algorithm.
*/
Path aStar(const RoadGraph& graph, RoadNode* start, RoadNode* end) {
    //length is not used in this case but is neccessary for alternative route
    double length;
    //calls function that works for the last three algorithms
    return helper(graph, start, end, true, nullptr, length);
}

/*This function does finds the shortes path for the last three algorithms. It accepts the graph of the map,
 * the start and end vertex, a boolean of whether to use the heuristic or not (true for A*), an edge to ignore
 * for finding an alternative route, and the refferenced length (should be passed in as empty) which is only used for
 * saving computational time in the fourth algorithm.
*/
Path helper(const RoadGraph& graph, RoadNode* start, RoadNode* end, bool useHeuristic, RoadEdge * e, double& length){
    //initializes priority queue and visited
    PriorityQueue<Path> paths;
    Set<RoadNode*> visited;
    visited.add(start);
    Path startP;
    startP.add(start);
    paths.enqueue(startP, heuristic(graph, start, end));
    //loops through paths (checking shorter ones first) adding neighbors to the last nodes until either a path is found or not
    while(!paths.isEmpty() && !visited.contains(end)){
       //gets path and last node from queue and adds to visited while highlighting
       double priority = paths.peekPriority();
       Path p = paths.dequeue();
       RoadNode * v = p[p.size() - 1];
       //takes off heuristic
       if(useHeuristic){
           priority -= heuristic(graph, v, end);
       }
       (*v).setColor(Color::GREEN);
       visited.add(v);
       //if path is complete, returns it and sets referenced length to path length
       if(v == end){
           length = priority;
           return p;
       }
       //searches adds new paths with the last node's unvisited neighbors (excluding edges if applicable)
       for(RoadNode * r : graph.neighborsOf(v)){
           if(!visited.contains(r) && graph.edgeBetween(v, r) != e){
               (*r).setColor(Color::YELLOW);
               //makes priority using time to travel edge plus heuristic
               RoadEdge * e = graph.edgeBetween(v, r);
               double time = (*e).cost();
               if(useHeuristic){
                   time += heuristic(graph, r, end);
               }
               Path newP = p;
               newP.add(r);
               paths.enqueue(newP, priority + time);
           }
       }
    }
    return {};
}

/*This function takes in the graph of the map, the current and end vertex,
 * and returns a double represnting the time to travel the distance as the crow
 * flies between the two at the maximum speed. Note: this heuristic is an
 * admissible heuristic - underestimate.
*/
double heuristic(const RoadGraph& graph, RoadNode* start, RoadNode* end){
    return graph.crowFlyDistanceBetween(start, end)/graph.maxRoadSpeed();
}

/*This function returns the fastest alternative route that differs by a
 * factor of 0.2 (using difference method)from the shortest path. The function
 *  accepts a graph of the map and the start and end vertex.
*/
Path alternativeRoute(const RoadGraph& graph, RoadNode* start, RoadNode* end) {
    //finds shortest path, and loops through edges of path
    Path p = aStar(graph, start, end);
    PriorityQueue<Path> paths;
    RoadNode * prev = start;
    RoadEdge * current;
    for(int i = 1; i < p.size(); i++){
       RoadNode * n = p[i];
       current = graph.edgeBetween(prev, n);
       double length;
       //finds shortest path ignoring edge from shortest path and adds to paths
       Path pNew = helper(graph, start, end, true, current, length);
       //length of path is utilized here when using the priority queue
       paths.enqueue(pNew, length);
       prev = n;
    }
    //finds difference for each path and returns first dequeued from priority queue
    //with difference greater than SUFFICIENT_DIFFERENCE
    int size = paths.size();
    for(int i = 0; i < size - 1; i++){
        Path a = paths.dequeue();
        double diff = difference(p, a);
        if(diff > SUFFICIENT_DIFFERENCE){
            return a;
        }
    }
    return p;
}

/*This function takes in two paths, a and b, which should be passed
 * as the shortest path and alternative path and returns the division of the
 * number of nodes that are in a but not in b by the size of b.
*/
double difference(Path& a, Path& b){
    //adds path a to set
    Set<RoadNode*> sA;
    for(RoadNode * n: a){
        sA.add(n);
    }
    //adds path b to different set
    Set<RoadNode*> sB;
    for(RoadNode * n: b){
        sB.add(n);
    }
    //the set of everything in A but not in B
    Set<RoadNode*> sDiff = sA - sB;
    return (1.0 * sDiff.size())/sB.size();
}

