/* cbt.h - public domain library for building binary trees in parallel
by Jonathan Dupuy

   Do this:
      #define CBT_IMPLEMENTATION
   before you include this file in *one* C or C++ file to create the implementation.
   
   // i.e. it should look like this:
   #include ...
   #include ...
   #include ...
   #define CBT_IMPLEMENTATION
   #include "cbt.h"

   INTERFACING
   define CBT_ASSERT(x) to avoid using assert.h
   define CBT_MALLOC(x) to use your own memory allocator
   define CBT_FREE(x) to use your own memory deallocator
   define CBT_MEMCPY(dst, src, num) to use your own memcpy routine
*/

#ifndef CBT_INCLUDE_CBT_H
#define CBT_INCLUDE_CBT_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef CBT_STATIC
#define CBTDEF static
#else
#define CBTDEF extern
#endif

#include <stdint.h>
#include <stdbool.h>

typedef struct cbt_Tree cbt_Tree;
typedef struct {
    uint64_t id   : 58; // heapID
    uint64_t depth:  6; // log2(heapID)
} cbt_Node;

// create / destroy tree
CBTDEF cbt_Tree *cbt_Create(int64_t maxDepth);
CBTDEF cbt_Tree *cbt_CreateAtDepth(int64_t maxDepth, int64_t depth);
CBTDEF void cbt_Release(cbt_Tree *tree);

// loaders
CBTDEF void cbt_ResetToRoot(cbt_Tree *tree);
CBTDEF void cbt_ResetToCeil(cbt_Tree *tree);
CBTDEF void cbt_ResetToDepth(cbt_Tree *tree, int64_t depth);

// manipulation
CBTDEF void cbt_SplitNode_Fast(cbt_Tree *tree, const cbt_Node node);
CBTDEF void cbt_SplitNode     (cbt_Tree *tree, const cbt_Node node);
CBTDEF void cbt_MergeNode_Fast(cbt_Tree *tree, const cbt_Node node);
CBTDEF void cbt_MergeNode     (cbt_Tree *tree, const cbt_Node node);
typedef void (*cbt_UpdateCallback)(cbt_Tree *tree,
                                   const cbt_Node node,
                                   const void *userData);
CBTDEF void cbt_Update(cbt_Tree *tree,
                       cbt_UpdateCallback updater,
                       const void *userData);

// O(1) queries
CBTDEF int64_t cbt_MaxDepth(const cbt_Tree *tree);
CBTDEF int64_t cbt_NodeCount(const cbt_Tree *tree);
CBTDEF uint64_t cbt_HeapRead(const cbt_Tree *tree, const cbt_Node node);
CBTDEF bool cbt_IsLeafNode(const cbt_Tree *tree, const cbt_Node node);
CBTDEF bool cbt_IsCeilNode(const cbt_Tree *tree, const cbt_Node node);
CBTDEF bool cbt_IsRootNode(                      const cbt_Node node);
CBTDEF bool cbt_IsNullNode(                      const cbt_Node node);

// node constructors
CBTDEF cbt_Node cbt_CreateNode           (uint64_t id, int64_t depth);
CBTDEF cbt_Node cbt_CreateNodeFromHeapID (uint64_t heapID);
CBTDEF cbt_Node cbt_ParentNode           (const cbt_Node node);
CBTDEF cbt_Node cbt_ParentNode_Fast      (const cbt_Node node);
CBTDEF cbt_Node cbt_SiblingNode          (const cbt_Node node);
CBTDEF cbt_Node cbt_SiblingNode_Fast     (const cbt_Node node);
CBTDEF cbt_Node cbt_LeftSiblingNode      (const cbt_Node node);
CBTDEF cbt_Node cbt_LeftSiblingNode_Fast (const cbt_Node node);
CBTDEF cbt_Node cbt_RightSiblingNode     (const cbt_Node node);
CBTDEF cbt_Node cbt_RightSiblingNode_Fast(const cbt_Node node);
CBTDEF cbt_Node cbt_LeftChildNode        (const cbt_Node node);
CBTDEF cbt_Node cbt_LeftChildNode_Fast   (const cbt_Node node);
CBTDEF cbt_Node cbt_RightChildNode       (const cbt_Node node);
CBTDEF cbt_Node cbt_RightChildNode_Fast  (const cbt_Node node);

// O(depth) queries
CBTDEF cbt_Node cbt_DecodeNode(const cbt_Tree *tree, int64_t leafID);
CBTDEF int64_t cbt_EncodeNode(const cbt_Tree *tree, const cbt_Node node);

// serialization
CBTDEF int64_t cbt_HeapByteSize(const cbt_Tree *tree);
CBTDEF const char *cbt_GetHeap(const cbt_Tree *tree);
CBTDEF void cbt_SetHeap(cbt_Tree *tree, const char *heapToCopy);

#ifdef __cplusplus
} // extern "C"
#endif

//
//
//// end header file ///////////////////////////////////////////////////////////
#endif // CBT_INCLUDE_CBT_H

#ifdef CBT_IMPLEMENTATION

#ifndef CBT_ASSERT
#    include <assert.h>
#    define CBT_ASSERT(x) assert(x)
#endif

#ifndef CBT_MALLOC
#    include <stdlib.h>
#    define CBT_MALLOC(x) (malloc(x))
#    define CBT_FREE(x) (free(x))
#else
#    ifndef CBT_FREE
#        error CBT_MALLOC defined without CBT_FREE
#    endif
#endif

#ifndef CBT_MEMCPY
#    include <string.h>
#    define CBT_MEMCPY(dst, src, num) memcpy(dst, src, num)
#endif

#ifndef _OPENMP
#   define CBT_ATOMIC
#   define CBT_PARALLEL_FOR
#   define CBT_BARRIER
#else
#   if defined(_WIN32)
#       define CBT_ATOMIC          __pragma("omp atomic" )
#       define CBT_PARALLEL_FOR    __pragma("omp parallel for")
#       define CBT_BARRIER         __pragma("omp barrier")
#   else
#       define CBT_ATOMIC          _Pragma("omp atomic" )
#       define CBT_PARALLEL_FOR    _Pragma("omp parallel for")
#       define CBT_BARRIER         _Pragma("omp barrier")
#   endif
#endif


/*******************************************************************************
 * FindLSB -- Returns the position of the least significant bit
 *
 */
static inline int64_t cbt__FindLSB(uint64_t x)
{
    int64_t lsb = 0;

    while (((x >> lsb) & 1u) == 0u) {
        ++lsb;
    }

    return lsb;
}


/*******************************************************************************
 * FindMSB -- Returns the position of the most significant bit
 *
 */
static inline int64_t cbt__FindMSB(uint64_t x)
{
    int64_t msb = 0;

    while (x > 1u) {
        ++msb;
        x = x >> 1;
    }

    return msb;
}


/*******************************************************************************
 * MinValue -- Returns the minimum value between two inputs
 *
 */
static inline uint64_t cbt__MinValue(uint64_t a, uint64_t b)
{
    return a < b ? a : b;
}


/*******************************************************************************
 * SetBitValue -- Sets the value of a bit stored in a bitfield
 *
 */
static void
cbt__SetBitValue(uint64_t *bitField, int64_t bitID, uint64_t bitValue)
{
    const uint64_t bitMask = ~(1ULL << bitID);

CBT_ATOMIC
    (*bitField)&= bitMask;
CBT_ATOMIC
    (*bitField)|= (bitValue << bitID);
}


/*******************************************************************************
 * BitfieldInsert -- Inserts data in range [offset, offset + count - 1]
 *
 */
static inline void
cbt__BitFieldInsert(
    uint64_t *bitField,
    int64_t  bitOffset,
    int64_t  bitCount,
    uint64_t bitData
) {
    CBT_ASSERT(bitOffset < 64 && bitCount <= 64 && bitOffset + bitCount <= 64);
    uint64_t bitMask = ~(~(0xFFFFFFFFFFFFFFFFULL << bitCount) << bitOffset);
CBT_ATOMIC
    (*bitField)&= bitMask;
CBT_ATOMIC
    (*bitField)|= (bitData << bitOffset);
}


/*******************************************************************************
 * BitFieldExtract -- Extracts bits [bitOffset, bitOffset + bitCount - 1] from
 * a bitfield, returning them in the least significant bits of the result.
 *
 */
static inline uint64_t
cbt__BitFieldExtract(
    const uint64_t bitField,
    int64_t bitOffset,
    int64_t bitCount
) {
    CBT_ASSERT(bitOffset < 64 && bitCount < 64 && bitOffset + bitCount <= 64);
    uint64_t bitMask = ~(0xFFFFFFFFFFFFFFFFULL << bitCount);

    return (bitField >> bitOffset) & bitMask;
}


/*******************************************************************************
 * Parallel Binary Tree Data-Structure
 *
 */
struct cbt_Tree {
    uint64_t *heap;
};


/*******************************************************************************
 * IsCeilNode -- Checks if a node is a ceil node, i.e., that can not split further
 *
 */
CBTDEF bool cbt_IsCeilNode(const cbt_Tree *tree, const cbt_Node node)
{
    return (node.depth == cbt_MaxDepth(tree));
}


/*******************************************************************************
 * IsRootNode -- Checks if a node is a root node
 *
 */
CBTDEF bool cbt_IsRootNode(const cbt_Node node)
{
    return (node.id == 1u);
}


/*******************************************************************************
 * IsNullNode -- Checks if a node is a null node
 *
 */
CBTDEF bool cbt_IsNullNode(const cbt_Node node)
{
    return (node.id == 0u);
}


/*******************************************************************************
 * CreateNode -- Constructor for the Node data structure
 *
 */
CBTDEF cbt_Node cbt_CreateNodeFromHeapID(uint64_t heapID)
{
    return cbt_CreateNode(heapID, cbt__FindMSB(heapID));
}


/*******************************************************************************
 * CreateNode -- Constructor for the Node data structure
 *
 */
CBTDEF cbt_Node cbt_CreateNode(uint64_t id, int64_t depth)
{
    cbt_Node node;

    node.id = id;
    node.depth = depth;

    return node;
}


/*******************************************************************************
 * ParentNode -- Computes the parent of the input node
 *
 */
CBTDEF cbt_Node cbt_ParentNode_Fast(const cbt_Node node)
{
    return cbt_CreateNode(node.id >> 1, node.depth - 1);
}

CBTDEF cbt_Node cbt_ParentNode(const cbt_Node node)
{
     return cbt_IsNullNode(node) ? node : cbt_ParentNode_Fast(node);
}


/*******************************************************************************
 * CeilNode -- Returns the associated ceil node, i.e., the deepest possible leaf
 *
 */
static cbt_Node cbt__CeilNode_Fast(const cbt_Tree *tree, const cbt_Node node)
{
    int64_t maxDepth = cbt_MaxDepth(tree);

    return cbt_CreateNode(node.id << (maxDepth - node.depth), maxDepth);
}

static cbt_Node cbt__CeilNode(const cbt_Tree *tree, const cbt_Node node)
{
    return cbt_IsNullNode(node) ? node : cbt__CeilNode_Fast(tree, node);
}


/*******************************************************************************
 * SiblingNode -- Computes the sibling of the input node
 *
 */
CBTDEF cbt_Node cbt_SiblingNode_Fast(const cbt_Node node)
{
    return cbt_CreateNode(node.id ^ 1u, node.depth);
}

CBTDEF cbt_Node cbt_SiblingNode(const cbt_Node node)
{
    return cbt_IsNullNode(node) ? node : cbt_SiblingNode_Fast(node);
}


/*******************************************************************************
 * RightSiblingNode -- Computes the right sibling of the input node
 *
 */
CBTDEF cbt_Node cbt_RightSiblingNode_Fast(const cbt_Node node)
{
    return cbt_CreateNode(node.id | 1u, node.depth);
}

CBTDEF cbt_Node cbt_RightSiblingNode(const cbt_Node node)
{
    return cbt_IsNullNode(node) ? node : cbt_RightSiblingNode_Fast(node);
}


/*******************************************************************************
 * LeftSiblingNode -- Computes the left sibling of the input node
 *
 */
CBTDEF cbt_Node cbt_LeftSiblingNode_Fast(const cbt_Node node)
{
    return cbt_CreateNode(node.id & (~1u), node.depth);
}

CBTDEF cbt_Node cbt_LeftSiblingNode(const cbt_Node node)
{
    return cbt_IsNullNode(node) ? node : cbt_LeftSiblingNode_Fast(node);
}


/*******************************************************************************
 * RightChildNode -- Computes the right child of the input node
 *
 */
CBTDEF cbt_Node cbt_RightChildNode_Fast(const cbt_Node node)
{
    return cbt_CreateNode(node.id << 1u | 1u, node.depth + 1);
}

CBTDEF cbt_Node cbt_RightChildNode(const cbt_Node node)
{
    return cbt_IsNullNode(node) ? node : cbt_RightChildNode_Fast(node);
}


/*******************************************************************************
 * LeftChildNode -- Computes the left child of the input node
 *
 */
CBTDEF cbt_Node cbt_LeftChildNode_Fast(const cbt_Node node)
{
    return cbt_CreateNode(node.id << 1u, node.depth + 1);
}

CBTDEF cbt_Node cbt_LeftChildNode(const cbt_Node node)
{
    return cbt_IsNullNode(node) ? node : cbt_LeftChildNode_Fast(node);
}


/*******************************************************************************
 * HeapByteSize -- Computes the number of Bytes to allocate for the bitfield
 *
 * For a tree of max depth D, the number of Bytes is 2^(D-1).
 * Note that 2 bits are "wasted" in the sense that they only serve
 * to round the required number of bytes to a power of two.
 *
 */
static int64_t cbt__HeapByteSize(uint64_t treeMaxDepth)
{
    return 1LL << (treeMaxDepth - 1);
}


/*******************************************************************************
 * HeapUint64Size -- Computes the number of uints to allocate for the bitfield
 *
 */
static inline int64_t cbt__HeapUint64Size(int64_t treeMaxDepth)
{
    return cbt__HeapByteSize(treeMaxDepth) >> 3;
}


/*******************************************************************************
 * NodeBitID -- Returns the bit index that stores data associated with a given node
 *
 * For a tree of max depth D and given an index in [0, 2^(D+1) - 1], this
 * functions is used to emulate the behaviour of a lookup in an array, i.e.,
 * uint[nodeID]. It provides the first bit in memory that stores
 * information associated with the element of index nodeID.
 *
 * For data located at level d, the bit offset is 2^d x (3 - d + D)
 * We then offset this quantity by the index by (nodeID - 2^d) x (D + 1 - d)
 * Note that the null index (nodeID = 0) is also supported.
 *
 */
static inline int64_t cbt__NodeBitID(const cbt_Tree *tree, const cbt_Node node)
{
    int64_t tmp1 = 2LL << node.depth;
    int64_t tmp2 = 1LL + cbt_MaxDepth(tree) - node.depth;

    return tmp1 + node.id * tmp2;
}


/*******************************************************************************
 * NodeBitID_BitField -- Computes the bitfield bit location associated to a node
 *
 * Here, the node is converted into a final node and its bit offset is
 * returned, which is finalNodeID + 2^{D + 1}
 */
static int64_t
cbt__NodeBitID_BitField(const cbt_Tree *tree, const cbt_Node node)
{
    return cbt__NodeBitID(tree, cbt__CeilNode(tree, node));
}


/*******************************************************************************
 * NodeBitSize -- Returns the number of bits storing the input node value
 *
 */
static inline int64_t
cbt__NodeBitSize(const cbt_Tree *tree, const cbt_Node node)
{
    return cbt_MaxDepth(tree) - node.depth + 1;
}


/*******************************************************************************
 * HeapArgs
 *
 * The CBT heap data structure uses an array of 64-bit words to store its data.
 * Whenever we need to access a certain bit range, we need to query two such
 * words (because sometimes the requested bit range overlaps two 64-bit words).
 * The HeapArg data structure provides arguments for reading from and/or
 * writing to the two 64-bit words that bound the queries range.
 *
 */
typedef struct {
    uint64_t *bitFieldLSB, *bitFieldMSB;
    int64_t bitOffsetLSB;
    int64_t bitCountLSB, bitCountMSB;
} cbt__HeapArgs;

cbt__HeapArgs
cbt__CreateHeapArgs(const cbt_Tree *tree, const cbt_Node node, int64_t bitCount)
{
    int64_t alignedBitOffset = cbt__NodeBitID(tree, node);
    int64_t maxBufferIndex = cbt__HeapUint64Size(cbt_MaxDepth(tree)) - 1;
    int64_t bufferIndexLSB = (alignedBitOffset >> 6);
    int64_t bufferIndexMSB = cbt__MinValue(bufferIndexLSB + 1, maxBufferIndex);
    cbt__HeapArgs args;

    args.bitOffsetLSB = alignedBitOffset & 63;
    args.bitCountLSB = cbt__MinValue(64 - args.bitOffsetLSB, bitCount);
    args.bitCountMSB = bitCount - args.bitCountLSB;
    args.bitFieldLSB = &tree->heap[bufferIndexLSB];
    args.bitFieldMSB = &tree->heap[bufferIndexMSB];

    return args;
}


/*******************************************************************************
 * HeapWrite -- Sets bitCount bits located at nodeID to bitData
 *
 * Note that this procedure writes to at most two uint64 elements.
 * Two elements are relevant whenever the specified interval overflows 64-bit
 * words.
 *
 */
static void
cbt__HeapWriteExplicit(
    cbt_Tree *tree,
    const cbt_Node node,
    int64_t bitCount,
    uint64_t bitData
) {
    cbt__HeapArgs args = cbt__CreateHeapArgs(tree, node, bitCount);

    cbt__BitFieldInsert(args.bitFieldLSB,
                        args.bitOffsetLSB,
                        args.bitCountLSB,
                        bitData);
    cbt__BitFieldInsert(args.bitFieldMSB,
                        0u,
                        args.bitCountMSB,
                        bitData >> args.bitCountLSB);
}

static void
cbt__HeapWrite(cbt_Tree *tree, const cbt_Node node, uint64_t bitData)
{
    cbt__HeapWriteExplicit(tree, node, cbt__NodeBitSize(tree, node), bitData);
}


/*******************************************************************************
 * HeapRead -- Returns bitCount bits located at nodeID
 *
 * Note that this procedure reads from two uint64 elements.
 * This is because the data is not necessarily aligned with 64-bit
 * words.
 *
 */
static uint64_t
cbt__HeapReadExplicit(
    const cbt_Tree *tree,
    const cbt_Node node,
    int64_t bitCount
) {
    cbt__HeapArgs args = cbt__CreateHeapArgs(tree, node, bitCount);
    uint64_t lsb = cbt__BitFieldExtract(*args.bitFieldLSB,
                                        args.bitOffsetLSB,
                                        args.bitCountLSB);
    uint64_t msb = cbt__BitFieldExtract(*args.bitFieldMSB,
                                        0u,
                                        args.bitCountMSB);

    return (lsb | (msb << args.bitCountLSB));
}

CBTDEF uint64_t cbt_HeapRead(const cbt_Tree *tree, const cbt_Node node)
{
    return cbt__HeapReadExplicit(tree, node, cbt__NodeBitSize(tree, node));
}


/*******************************************************************************
 * HeapWrite_BitField -- Sets the bit associated to a leaf node to bitValue
 *
 * This is a dedicated routine to write directly to the bitfield.
 *
 */
static void
cbt__HeapWrite_BitField(
    cbt_Tree *tree,
    const cbt_Node node,
    const uint64_t bitValue
) {
    int64_t bitID = cbt__NodeBitID_BitField(tree, node);

    cbt__SetBitValue(&tree->heap[bitID >> 6], bitID & 63, bitValue);
}


/*******************************************************************************
 * ClearBitField -- Clears the bitfield
 *
 */
static void cbt__ClearBitfield(cbt_Tree *tree)
{
    int64_t maxDepth = cbt_MaxDepth(tree);
    int64_t bufferMinID = 1LL << (maxDepth - 5);
    int64_t bufferMaxID = cbt__HeapUint64Size(maxDepth);

CBT_PARALLEL_FOR
    for (int bufferID = bufferMinID; bufferID < bufferMaxID; ++bufferID) {
        tree->heap[bufferID] = 0;
    }
CBT_BARRIER
}


/*******************************************************************************
 * IsLeafNode -- Checks if a node is a leaf node, i.e., that has no descendants
 *
 */
CBTDEF bool cbt_IsLeafNode(const cbt_Tree *tree, const cbt_Node node)
{
    return (cbt_HeapRead(tree, node) == 1u);
}


/*******************************************************************************
 * GetHeapMemory -- Returns a read-only pointer to the heap memory
 *
 */
CBTDEF const char *cbt_GetHeap(const cbt_Tree *tree)
{
    return (const char *)tree->heap;
}


/*******************************************************************************
 * SetHeapMemory -- Sets the heap memory from a read-only buffer
 *
 */
CBTDEF void cbt_SetHeap(cbt_Tree *tree, const char *buffer)
{
    CBT_MEMCPY(tree->heap, buffer, cbt_HeapByteSize(tree));
}


/*******************************************************************************
 * HeapByteSize -- Returns the amount of bytes consumed by the CBT heap
 *
 */
CBTDEF int64_t cbt_HeapByteSize(const cbt_Tree *tree)
{
    return cbt__HeapByteSize(cbt_MaxDepth(tree));
}


/*******************************************************************************
 * ComputeSumReduction -- Sums the 2 elements below the current slot
 *
 */
static void cbt__ComputeSumReduction(cbt_Tree *tree)
{
    int64_t depth = cbt_MaxDepth(tree);
    uint64_t minNodeID = (1ULL << depth);
    uint64_t maxNodeID = (2ULL << depth);

    // prepass: processes deepest levels in parallel
CBT_PARALLEL_FOR
    for (uint64_t nodeID = minNodeID; nodeID < maxNodeID; nodeID+= 64u) {
        cbt_Node heapNode = cbt_CreateNode(nodeID, depth);
        int64_t alignedBitOffset = cbt__NodeBitID(tree, heapNode);
        uint64_t bitField = tree->heap[alignedBitOffset >> 6];
        uint64_t bitData = 0u;

        // 2-bits
        bitField = (bitField & 0x5555555555555555ULL)
                 + ((bitField >>  1) & 0x5555555555555555ULL);
        bitData = bitField;
        tree->heap[(alignedBitOffset - minNodeID) >> 6] = bitData;

        // 3-bits
        bitField = (bitField & 0x3333333333333333ULL)
                 + ((bitField >>  2) & 0x3333333333333333ULL);
        bitData = ((bitField >>  0) & (7ULL <<  0))
                | ((bitField >>  1) & (7ULL <<  3))
                | ((bitField >>  2) & (7ULL <<  6))
                | ((bitField >>  3) & (7ULL <<  9))
                | ((bitField >>  4) & (7ULL << 12))
                | ((bitField >>  5) & (7ULL << 15))
                | ((bitField >>  6) & (7ULL << 18))
                | ((bitField >>  7) & (7ULL << 21))
                | ((bitField >>  8) & (7ULL << 24))
                | ((bitField >>  9) & (7ULL << 27))
                | ((bitField >> 10) & (7ULL << 30))
                | ((bitField >> 11) & (7ULL << 33))
                | ((bitField >> 12) & (7ULL << 36))
                | ((bitField >> 13) & (7ULL << 39))
                | ((bitField >> 14) & (7ULL << 42))
                | ((bitField >> 15) & (7ULL << 45));
        cbt__HeapWriteExplicit(tree, cbt_CreateNode(nodeID >> 2, depth - 2), 48ULL, bitData);

        // 4-bits
        bitField = (bitField & 0x0F0F0F0F0F0F0F0FULL)
                 + ((bitField >>  4) & 0x0F0F0F0F0F0F0F0FULL);
        bitData = ((bitField >>  0) & (15ULL <<  0))
                | ((bitField >>  4) & (15ULL <<  4))
                | ((bitField >>  8) & (15ULL <<  8))
                | ((bitField >> 12) & (15ULL << 12))
                | ((bitField >> 16) & (15ULL << 16))
                | ((bitField >> 20) & (15ULL << 20))
                | ((bitField >> 24) & (15ULL << 24))
                | ((bitField >> 28) & (15ULL << 28));
        cbt__HeapWriteExplicit(tree, cbt_CreateNode(nodeID >> 3, depth - 3), 32ULL, bitData);

        // 5-bits
        bitField = (bitField & 0x00FF00FF00FF00FFULL)
                 + ((bitField >>  8) & 0x00FF00FF00FF00FFULL);
        bitData = ((bitField >>  0) & (31ULL <<  0))
                | ((bitField >> 11) & (31ULL <<  5))
                | ((bitField >> 22) & (31ULL << 10))
                | ((bitField >> 33) & (31ULL << 15));
        cbt__HeapWriteExplicit(tree, cbt_CreateNode(nodeID >> 4, depth - 4), 20ULL, bitData);

        // 6-bits
        bitField = (bitField & 0x0000FFFF0000FFFFULL)
                 + ((bitField >> 16) & 0x0000FFFF0000FFFFULL);
        bitData = ((bitField >>  0) & (63ULL << 0))
                | ((bitField >> 26) & (63ULL << 6));
        cbt__HeapWriteExplicit(tree, cbt_CreateNode(nodeID >> 5, depth - 5), 12ULL, bitData);

        // 7-bits
        bitField = (bitField & 0x00000000FFFFFFFFULL)
                 + ((bitField >> 32) & 0x00000000FFFFFFFFULL);
        bitData = bitField;
        cbt__HeapWriteExplicit(tree, cbt_CreateNode(nodeID >> 6, depth - 6),  7ULL, bitData);
    }
CBT_BARRIER
    depth-= 6;

    // iterate over elements atomically
    while (--depth >= 0) {
        uint64_t minNodeID = 1ULL << depth;
        uint64_t maxNodeID = 2ULL << depth;

CBT_PARALLEL_FOR
        for (uint64_t j = minNodeID; j < maxNodeID; ++j) {
            uint64_t x0 = cbt_HeapRead(tree, cbt_CreateNode(j << 1    , depth + 1));
            uint64_t x1 = cbt_HeapRead(tree, cbt_CreateNode(j << 1 | 1, depth + 1));

            cbt__HeapWrite(tree, cbt_CreateNode(j, depth), x0 + x1);
        }
CBT_BARRIER
    }
}


/*******************************************************************************
 * Buffer Ctor
 *
 */
CBTDEF cbt_Tree *cbt_CreateAtDepth(int64_t maxDepth, int64_t depth)
{
    CBT_ASSERT(maxDepth >=  5 && "maxDepth must be at least 5");
    CBT_ASSERT(maxDepth <= 58 && "maxDepth must be at most 58");
    cbt_Tree *tree = (cbt_Tree *)CBT_MALLOC(sizeof(*tree));

    tree->heap = (uint64_t *)CBT_MALLOC(cbt__HeapByteSize(maxDepth));
    tree->heap[0] = 1ULL << (maxDepth); // store max Depth

    cbt_ResetToDepth(tree, depth);

    return tree;
}

CBTDEF cbt_Tree *cbt_Create(int64_t maxDepth)
{
    return cbt_CreateAtDepth(maxDepth, 0);
}


/*******************************************************************************
 * Buffer Dtor
 *
 */
CBTDEF void cbt_Release(cbt_Tree *tree)
{
    CBT_FREE(tree->heap);
    CBT_FREE(tree);
}


/*******************************************************************************
 * ResetToDepth -- Initializes a CBT to its a specific subdivision level
 *
 */
CBTDEF void cbt_ResetToDepth(cbt_Tree *tree, int64_t depth)
{
    CBT_ASSERT(depth >= 0 && "depth must be at least equal to 0");
    CBT_ASSERT(depth <= cbt_MaxDepth(tree) && "depth must be at most equal to maxDepth");
    uint64_t minNodeID = 1ULL << depth;
    uint64_t maxNodeID = 2ULL << depth;

    cbt__ClearBitfield(tree);

CBT_PARALLEL_FOR
    for (uint64_t nodeID = minNodeID; nodeID < maxNodeID; ++nodeID) {
        cbt_Node node = cbt_CreateNode(nodeID, depth);

        cbt__HeapWrite_BitField(tree, node, 1u);
    }
CBT_BARRIER

    cbt__ComputeSumReduction(tree);
}


/*******************************************************************************
 * ResetToCeil -- Initializes a CBT to its maximum subdivision level
 *
 */
CBTDEF void cbt_ResetToCeil(cbt_Tree *tree)
{
    cbt_ResetToDepth(tree, cbt_MaxDepth(tree));
}


/*******************************************************************************
 * ResetToRoot -- Initializes a CBT to its minimum subdivision level
 *
 */
CBTDEF void cbt_ResetToRoot(cbt_Tree *tree)
{
    cbt_ResetToDepth(tree, 0);
}


/*******************************************************************************
 * Split -- Subdivides a node in two
 *
 * The _Fast version does not check if the node can actually split, so
 * use it wisely, i.e., when you're absolutely sure the node depth is
 * less than maxDepth.
 *
 */
CBTDEF void cbt_SplitNode_Fast(cbt_Tree *tree, const cbt_Node node)
{
    cbt__HeapWrite_BitField(tree, cbt_RightChildNode(node), 1u);
}

CBTDEF void cbt_SplitNode(cbt_Tree *tree, const cbt_Node node)
{
    if (!cbt_IsCeilNode(tree, node))
        cbt_SplitNode_Fast(tree, node);
}


/*******************************************************************************
 * Merge -- Merges the node with its neighbour
 *
 * The _Fast version does not check if the node can actually merge, so
 * use it wisely, i.e., when you're absolutely sure the node depth is
 * greater than 0.
 *
 */
CBTDEF void cbt_MergeNode_Fast(cbt_Tree *tree, const cbt_Node node)
{
    cbt__HeapWrite_BitField(tree, cbt_RightSiblingNode(node), 0u);
}

CBTDEF void cbt_MergeNode(cbt_Tree *tree, const cbt_Node node)
{
    if (!cbt_IsRootNode(node))
        cbt_MergeNode_Fast(tree, node);
}


/*******************************************************************************
 * Update -- Split or merge each node in parallel
 *
 * The user provides an updater function that is responsible for
 * splitting or merging each node.
 *
 */
CBTDEF void
cbt_Update(cbt_Tree *tree, cbt_UpdateCallback updater, const void *userData)
{
CBT_PARALLEL_FOR
    for (int64_t handle = 0; handle < cbt_NodeCount(tree); ++handle) {
        updater(tree, cbt_DecodeNode(tree, handle), userData);
    }
CBT_BARRIER

    cbt__ComputeSumReduction(tree);
}


/*******************************************************************************
 * MaxDepth -- Returns the max CBT depth
 *
 */
CBTDEF int64_t cbt_MaxDepth(const cbt_Tree *tree)
{
    return cbt__FindLSB(tree->heap[0]);
}


/*******************************************************************************
 * NodeCount -- Returns the number of triangles in the CBT
 *
 */
CBTDEF int64_t cbt_NodeCount(const cbt_Tree *tree)
{
    return cbt_HeapRead(tree, cbt_CreateNode(1u, 0));
}


/*******************************************************************************
 * DecodeNode -- Returns the leaf node associated to index nodeID
 *
 * This is procedure is for iterating over the nodes.
 *
 */
CBTDEF cbt_Node cbt_DecodeNode(const cbt_Tree *tree, int64_t handle)
{
    CBT_ASSERT(handle < cbt_NodeCount(tree) && "handle > NodeCount");
    CBT_ASSERT(handle >= 0 && "handle < 0");

    cbt_Node node = cbt_CreateNode(1u, 0);

    while (cbt_HeapRead(tree, node) > 1u) {
        cbt_Node heapNode = cbt_CreateNode(node.id<<= 1u, ++node.depth);
        uint64_t cmp = cbt_HeapRead(tree, heapNode);
        uint64_t b = (uint64_t)handle < cmp ? 0u : 1u;

        node.id|= b;
        handle-= cmp * b;
    }

    return node;
}


/*******************************************************************************
 * EncodeNode -- Returns the bit index associated with the Node
 *
 * This does the inverse of the DecodeNode routine.
 *
 */
CBTDEF int64_t cbt_EncodeNode(const cbt_Tree *tree, const cbt_Node node)
{
    CBT_ASSERT(cbt_IsLeafNode(tree, node) && "node is not a leaf");

    int64_t handle = 0u;
    cbt_Node nodeIterator = node;

    while (nodeIterator.id > 1u) {
        cbt_Node sibling = cbt_LeftSiblingNode_Fast(nodeIterator);
        uint64_t nodeCount = cbt_HeapRead(tree, sibling);

        handle+= (nodeIterator.id & 1u) * nodeCount;
        nodeIterator = cbt_ParentNode_Fast(nodeIterator);
    }

    return handle;
}


#undef CBT_ATOMIC
#undef CBT_PARALLEL_FOR
#undef CBT_BARRIER
#endif

