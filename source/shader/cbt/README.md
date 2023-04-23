# Concurrent Binary Tree Library

This library provides a concurrent binary tree data-structure suitable for accelerating the processing of subdivision algorithms on multicore processors, including GPUs. More details are available in my paper ["Concurrent Binary Trees (with application to Longest Edge Bisection)"](https://onrendering.com/).

### Usage

**Initialization**
A CBT requires a maximum depth (typically the maximum depth of the subdivision algorithm you're interested in accelerating). 
```c
cbt_Tree *cbt = cbt_Create(myMaximumDepth);
```
Once this depth chosen, it can not be changed throughout the lifetime of the CBT instance. You can query the maximum depth of the CBT as follows:
```c
maximumDepth = cbt_MaxDepth(cbt);
```
By default, the CBT will be initialized at the root node. You can also initialize it at an explicit subdivision depth as follows:
```c
cbt_Tree *cbt = cbt_CreateAtDepth(myMaximumDepth, myInitializationDepth);
```
Note that the initialization depth must be less or equal to the maximum depth of the CBT.
Always remember to release the meomory once you're done with your CBT:
```c
cbt_Release(cbt);
```

**Resetting the tree**
Additionally, you can reset the subdivision by using any of the following routines:
```c
cbt_ResetToRoot(cbt); // resets the CBT to its root
cbt_ResetToCeil(cbt); // resets the CBT to its maximum depth
cbt_ResetToDepth(cbt, myInitializationDepth); // resets the CBT to a custom depth
```

**Updating the tree in parallel**
The main advantage of CBTs is their ability to update their topology in parallel. Nodes can be split or merged using respectively `cbt_SplitNode(cbt, node)` and `cbt_MergeNode(cbt, node)`. In order to process the operations in parallel, you can provide a custom callback that will be executed in parallel within an OpenMP parallel for loop. Here is a simple example that splits or merges nodes if their index is even:
```c
// update callback
void UpdateCallback(cbt_Tree *cbt, const cbt_Node node, const void *userData)
{
    if ((node.id & 1) == 0) {
#ifdef SPLIT
        cbt_SplitNode(cbt, node);
#else
        cbt_MergeNode(cbt, node);
#endif        
    }
}

// execute the update callback in parallel
cbt_Update(cbt, &UpdateCallback, NULL);
```
For a more complex example, see [this repo](https://github.com/jdupuy/LongestEdgeBisection2D).

**Queries**
You can query the number of leaf nodes in the CBT using 
```c
int64_t nodeCount = cbt_NodeCount(cbt);
```
You can retrieve the i-th leaf node using 
```c
cbt_Node node = cbt_DecodeNode(cbt, i);
```
Conversely, you can retrieve the index of an existing leaf node using
```c
int64_t nodeID = cbt_EncodeNode(cbt, node);
```


**Serialization**
Internally, the CBT uses a compact binary heap data-structure, i.e., a 1D array. This makes the CBT trivial to serialize. To access the heap, use 
```c
int64_t cbtByteSize = cbt_HeapByteSize(cbt); // size in Bytes of the CBT
char *cbtMemory = cbt_GetHeap(cbt); // CBT raw-data
```
 

**GPU implementation**
The GLSL folder provides a GLSL implementation of the library. An HLSL port of the library would also be welcome.
For a GPU implementation example, see [this repo](https://github.com/jdupuy/LongestEdgeBisection2D).


### License

The code from this repository is released in public domain. You can do anything you want with them. You have no legal obligation to do anything else, although I appreciate attribution.

It is also licensed under the MIT open source license, if you have lawyers who are unhappy with public domain.

