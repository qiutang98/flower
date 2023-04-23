/* leb.h - public domain Longest Edge Bisection library
by Jonathan Dupuy

   This is a library that builds upon the CBT library for computing
   longest-edge bisections.

   INTERFACING

   define LEB_ASSERT(x) to avoid using assert.h.

*/

#ifndef LEB_INCLUDE_LEB_H
#define LEB_INCLUDE_LEB_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef LEB_STATIC
#define LEBDEF static
#else
#define LEBDEF extern
#endif

// data structures
typedef struct {
    cbt_Node base, top;
} leb_DiamondParent;
LEBDEF leb_DiamondParent leb_DecodeDiamondParent       (const cbt_Node node);
LEBDEF leb_DiamondParent leb_DecodeDiamondParent_Square(const cbt_Node node);

// manipulation
LEBDEF void leb_SplitNode       (cbt_Tree *cbt, const cbt_Node node);
LEBDEF void leb_SplitNode_Square(cbt_Tree *cbt, const cbt_Node node);
LEBDEF void leb_MergeNode       (cbt_Tree *cbt,
                                 const cbt_Node node,
                                 const leb_DiamondParent diamond);
LEBDEF void leb_MergeNode_Square(cbt_Tree *cbt,
                                 const cbt_Node node,
                                 const leb_DiamondParent diamond);

// subdivision routine O(depth)
LEBDEF void leb_DecodeNodeAttributeArray       (const cbt_Node node,
                                                int64_t attributeArraySize,
                                                float attributeArray[][3]);
LEBDEF void leb_DecodeNodeAttributeArray_Square(const cbt_Node node,
                                                int64_t attributeArraySize,
                                                float attributeArray[][3]);

#ifdef __cplusplus
} // extern "C"
#endif

//
//
//// end header file ///////////////////////////////////////////////////////////
#endif // LEB_INCLUDE_LEB_H

#ifdef LEB_IMPLEMENTATION

#ifndef LEB_ASSERT
#    include <assert.h>
#    define LEB_ASSERT(x) assert(x)
#endif

typedef struct {
    uint64_t left, right, edge, node;
} leb__SameDepthNeighborIDs;

leb__SameDepthNeighborIDs
leb__CreateSameDepthNeighborIDs(
    uint64_t left,
    uint64_t right,
    uint64_t edge,
    uint64_t node
) {
    leb__SameDepthNeighborIDs neighborIDs;

    neighborIDs.left = left;
    neighborIDs.right = right;
    neighborIDs.edge = edge;
    neighborIDs.node = node;

    return neighborIDs;
}

leb_DiamondParent
leb__CreateDiamondParent(const cbt_Node base, const cbt_Node top)
{
    leb_DiamondParent diamond;

    diamond.base = base;
    diamond.top = top;

    return diamond;
}


/*******************************************************************************
 * GetBitValue -- Returns the value of a bit stored in a 64-bit word
 *
 */
static uint64_t leb__GetBitValue(const uint64_t bitField, int64_t bitID)
{
    return ((bitField >> bitID) & 1u);
}


/*******************************************************************************
 * SplitNodeIDs -- Updates the IDs of neighbors after one LEB split
 *
 * This code applies the following rules:
 * Split left:
 * LeftID  = 2 * NodeID + 1
 * RightID = 2 * EdgeID + 1
 * EdgeID  = 2 * RightID + 1
 *
 * Split right:
 * LeftID  = 2 * EdgeID
 * RightID = 2 * NodeID
 * EdgeID  = 2 * LeftID
 *
 * The _reserved channel stores NodeID, which is required for applying the
 * rules.
 *
 */
static leb__SameDepthNeighborIDs
leb__SplitNodeIDs(
    const leb__SameDepthNeighborIDs nodeIDs,
    const uint64_t splitBit
) {
    uint64_t n1 = nodeIDs.left, n2 = nodeIDs.right,
             n3 = nodeIDs.edge, n4 = nodeIDs.node;
    uint64_t b2 = (n2 == 0u) ? 0u : 1u,
             b3 = (n3 == 0u) ? 0u : 1u;

    if (splitBit == 0u) {
        return leb__CreateSameDepthNeighborIDs(
            n4 << 1 | 1, n3 << 1 | b3, n2 << 1 | b2, n4 << 1
        );
    } else {
        return leb__CreateSameDepthNeighborIDs(
            n3 << 1    , n4 << 1     , n1 << 1     , n4 << 1 | 1
        );
    }
}


/*******************************************************************************
 * DecodeNodeNeighborIDs -- Decodes the IDs of the cbt_Nodes neighbor to node
 *
 * The IDs are associated to the depth of the input node. As such, they
 * don't necessarily exist in the LEB subdivision.
 *
 */
leb__SameDepthNeighborIDs
leb_DecodeSameDepthNeighborIDs(const cbt_Node node)
{
    leb__SameDepthNeighborIDs nodeIDs =
        leb__CreateSameDepthNeighborIDs(0u, 0u, 0u, 1u);

    for (int64_t bitID = node.depth - 1; bitID >= 0; --bitID) {
        nodeIDs = leb__SplitNodeIDs(nodeIDs, leb__GetBitValue(node.id, bitID));
    }

    return nodeIDs;
}

leb__SameDepthNeighborIDs
leb_DecodeSameDepthNeighborIDs_Square(const cbt_Node node)
{
    int64_t bitID = node.depth > 0 ? node.depth - 1 : 0;
    uint64_t b = leb__GetBitValue(node.id, bitID);
    leb__SameDepthNeighborIDs nodeIDs =
        leb__CreateSameDepthNeighborIDs(0u, 0u, 3u - b, 2u + b);

    for (bitID = node.depth - 2; bitID >= 0; --bitID) {
        nodeIDs = leb__SplitNodeIDs(nodeIDs, leb__GetBitValue(node.id, bitID));
    }

    return nodeIDs;
}


/*******************************************************************************
 * EdgeNeighbor -- Computes the neighbour of the input node wrt to its longest edge
 *
 */
cbt_Node leb__EdgeNeighbor(const cbt_Node node)
{
    uint64_t nodeID = leb_DecodeSameDepthNeighborIDs(node).edge;

    return cbt_CreateNode(nodeID, (nodeID == 0u) ? 0 : node.depth);
}

cbt_Node leb__EdgeNeighbor_Square(const cbt_Node node)
{
    uint64_t nodeID = leb_DecodeSameDepthNeighborIDs_Square(node).edge;

    return cbt_CreateNode(nodeID, (nodeID == 0u) ? 0 : node.depth);
}


/*******************************************************************************
 * SplitNode -- Splits a node while producing a conforming LEB
 *
 */
void leb_SplitNode(cbt_Tree *cbt, const cbt_Node node)
{
    if (!cbt_IsCeilNode(cbt, node)) {
        const uint64_t minNodeID = 1u;
        cbt_Node nodeIterator = node;

        cbt_SplitNode(cbt, nodeIterator);
        nodeIterator = leb__EdgeNeighbor(nodeIterator);

        while (nodeIterator.id > minNodeID) {
            cbt_SplitNode(cbt, nodeIterator);
            nodeIterator = cbt_ParentNode_Fast(nodeIterator);
            cbt_SplitNode(cbt, nodeIterator);
            nodeIterator = leb__EdgeNeighbor(nodeIterator);
        }
    }
}

void leb_SplitNode_Square(cbt_Tree *cbt, const cbt_Node node)
{
    if (!cbt_IsCeilNode(cbt, node)) {
        const uint64_t minNodeID = 1u;
        cbt_Node nodeIterator = node;

        cbt_SplitNode(cbt, nodeIterator);
        nodeIterator = leb__EdgeNeighbor_Square(nodeIterator);

        while (nodeIterator.id > minNodeID) {
            cbt_SplitNode(cbt, nodeIterator);
            nodeIterator = cbt_ParentNode_Fast(nodeIterator);

            if (nodeIterator.id > minNodeID) {
                cbt_SplitNode(cbt, nodeIterator);
                nodeIterator = leb__EdgeNeighbor_Square(nodeIterator);
            }
        }
    }
}


/*******************************************************************************
 * DecodeDiamondParent -- Decodes the diamond associated to the Node
 *
 * If the neighbour part does not exist, the parentNode is copied instead.
 *
 */
leb_DiamondParent leb_DecodeDiamondParent(const cbt_Node node)
{
    cbt_Node parentNode = cbt_ParentNode_Fast(node);
    uint64_t edgeNeighborID = leb_DecodeSameDepthNeighborIDs(parentNode).edge;
    cbt_Node edgeNeighborNode = cbt_CreateNode(
        edgeNeighborID > 0u ? edgeNeighborID : parentNode.id,
        parentNode.depth
    );

    return leb__CreateDiamondParent(parentNode, edgeNeighborNode);
}

leb_DiamondParent leb_DecodeDiamondParent_Square(const cbt_Node node)
{
    cbt_Node parentNode = cbt_ParentNode_Fast(node);
    uint64_t edgeNeighborID = leb_DecodeSameDepthNeighborIDs_Square(parentNode).edge;
    cbt_Node edgeNeighborNode = cbt_CreateNode(
        edgeNeighborID > 0u ? edgeNeighborID : parentNode.id,
        parentNode.depth
    );

    return leb__CreateDiamondParent(parentNode, edgeNeighborNode);
}


/*******************************************************************************
 * HasDiamondParent -- Determines whether a diamond parent is actually stored
 *
 * This procedure checks that the diamond parent is encoded in the CBT.
 * We can perform this test by checking that both the base and top nodes
 * that form the diamond parent are split, i.e., CBT[base] = CBT[top] = 2.
 * This is a crucial operation for implementing the leb_Merge routine.
 *
 */
bool
leb__HasDiamondParent(
    cbt_Tree *cbt,
    const leb_DiamondParent diamondParent
) {
    bool canMergeBase = cbt_HeapRead(cbt, diamondParent.base) <= 2u;
    bool canMergeTop  = cbt_HeapRead(cbt, diamondParent.top) <= 2u;

    return canMergeBase && canMergeTop;
}


/*******************************************************************************
 * MergeNode -- Merges a node while producing a conforming LEB
 *
 * This routines makes sure that the children of a diamond (including the
 * input node) all exist in the LEB before calling a merge.
 *
 */
void
leb_MergeNode(
    cbt_Tree *cbt,
    const cbt_Node node,
    const leb_DiamondParent diamondParent
) {
    if (!cbt_IsRootNode(node) && leb__HasDiamondParent(cbt, diamondParent)) {
        cbt_MergeNode(cbt, node);
    }
}

void
leb_MergeNode_Square(
    cbt_Tree *cbt,
    const cbt_Node node,
    const leb_DiamondParent diamondParent
) {
    if ((node.depth > 1) && leb__HasDiamondParent(cbt, diamondParent)) {
        cbt_MergeNode(cbt, node);
    }
}


/******************************************************************************/
/* Standalone matrix 3x3 API
 *
 */
typedef float lebMatrix3x3[3][3];


/*******************************************************************************
 * IdentityMatrix3x3 -- Sets a 3x3 matrix to identity
 *
 */
static void leb__IdentityMatrix3x3(lebMatrix3x3 m)
{
    m[0][0] = 1.0f; m[0][1] = 0.0f; m[0][2] = 0.0f;
    m[1][0] = 0.0f; m[1][1] = 1.0f; m[1][2] = 0.0f;
    m[2][0] = 0.0f; m[2][1] = 0.0f; m[2][2] = 1.0f;
}


/*******************************************************************************
 * TransposeMatrix3x3 -- Transposes a 3x3 matrix
 *
 */
static void leb__TransposeMatrix3x3(const lebMatrix3x3 m, lebMatrix3x3 out)
{
    for (int64_t i = 0; i < 3; ++i)
    for (int64_t j = 0; j < 3; ++j)
        out[i][j] = m[j][i];
}


/*******************************************************************************
 * DotProduct -- Returns the dot product of two vectors
 *
 */
static float leb__DotProduct(int64_t argSize, const float *x, const float *y)
{
    float dp = 0.0f;

    for (int64_t i = 0; i < argSize; ++i)
        dp+= x[i] * y[i];

    return dp;
}


/*******************************************************************************
 * MulMatrix3x3 -- Computes the product of two 3x3 matrices
 *
 */
static void
leb__Matrix3x3Product(
    const lebMatrix3x3 m1,
    const lebMatrix3x3 m2,
    lebMatrix3x3 out
) {
    lebMatrix3x3 tra;

    leb__TransposeMatrix3x3(m2, tra);

    for (int64_t j = 0; j < 3; ++j)
    for (int64_t i = 0; i < 3; ++i)
        out[j][i] = leb__DotProduct(3, m1[j], tra[i]);
}


/*******************************************************************************
 * SplittingMatrix -- Computes a LEB splitting matrix from a split bit
 *
 */
static void
leb__SplittingMatrix(lebMatrix3x3 matrix, uint64_t bitValue)
{
    float b = (float)bitValue;
    float c = 1.0f - b;
    lebMatrix3x3 splitMatrix = {
        {c   , b   , 0.0f},
        {0.5f, 0.0f, 0.5f},
        {0.0f,    c,    b}
    };
    lebMatrix3x3 tmp;

    memcpy(tmp, matrix, sizeof(tmp));
    leb__Matrix3x3Product(splitMatrix, tmp, matrix);
}


/*******************************************************************************
 * SquareMatrix -- Computes a mirroring matrix from a split bit
 *
 */
static void leb__SquareMatrix(lebMatrix3x3 matrix, uint64_t bitValue)
{
    float b = (float)bitValue;
    float c = 1.0f - b;
    lebMatrix3x3 squareMatrix = {
        {c, 0.0f,    b},
        {b,    c,    b},
        {b, 0.0f,    c}
    };

    memcpy(matrix, squareMatrix, sizeof(squareMatrix));
}


/*******************************************************************************
 * WindingMatrix -- Computes a mirroring matrix from a split bit
 *
 */
static void leb__WindingMatrix(lebMatrix3x3 matrix, uint64_t bitValue)
{
    float b = (float)bitValue;
    float c = 1.0f - b;
    lebMatrix3x3 windingMatrix = {
        {c, 0.0f, b},
        {0, 1.0f, 0},
        {b, 0.0f, c}
    };
    lebMatrix3x3 tmp;

    memcpy(tmp, matrix, sizeof(tmp));
    leb__Matrix3x3Product(windingMatrix, tmp, matrix);
}


/*******************************************************************************
 * DecodeTransformationMatrix -- Computes the matrix associated to a LEB
 * node
 *
 */
static void
leb__DecodeTransformationMatrix(
    const cbt_Node node,
    lebMatrix3x3 matrix
) {
    leb__IdentityMatrix3x3(matrix);

    for (int64_t bitID = node.depth - 1; bitID >= 0; --bitID) {
        leb__SplittingMatrix(matrix, leb__GetBitValue(node.id, bitID));
    }

    leb__WindingMatrix(matrix, node.depth & 1);
}

static void
leb__DecodeTransformationMatrix_Square(
    const cbt_Node node,
    lebMatrix3x3 matrix
) {
    int64_t bitID = node.depth > 0 ? node.depth - 1 : 0;
    leb__SquareMatrix(matrix, leb__GetBitValue(node.id, bitID));

    for (bitID = node.depth - 2; bitID >= 0; --bitID) {
        leb__SplittingMatrix(matrix, leb__GetBitValue(node.id, bitID));
    }

    leb__WindingMatrix(matrix, (node.depth ^ 1) & 1);
}


/*******************************************************************************
 * DecodeNodeAttributeArray -- Compute the triangle attributes at the input node
 *
 */
LEBDEF void
leb_DecodeNodeAttributeArray(
    const cbt_Node node,
    int64_t attributeArraySize,
    float attributeArray[][3]
) {
    LEB_ASSERT(attributeArraySize > 0);

    lebMatrix3x3 m;
    float attributeVector[3];

    leb__DecodeTransformationMatrix(node, m);

    for (int64_t i = 0; i < attributeArraySize; ++i) {
        memcpy(attributeVector, attributeArray[i], sizeof(attributeVector));
        attributeArray[i][0] = leb__DotProduct(3, m[0], attributeVector);
        attributeArray[i][1] = leb__DotProduct(3, m[1], attributeVector);
        attributeArray[i][2] = leb__DotProduct(3, m[2], attributeVector);
    }
}

LEBDEF void
leb_DecodeNodeAttributeArray_Square(
    const cbt_Node node,
    int64_t attributeArraySize,
    float attributeArray[][3]
) {
    LEB_ASSERT(attributeArraySize > 0);

    lebMatrix3x3 m;
    float attributeVector[3];

    leb__DecodeTransformationMatrix_Square(node, m);

    for (int64_t i = 0; i < attributeArraySize; ++i) {
        memcpy(attributeVector, attributeArray[i], sizeof(attributeVector));
        attributeArray[i][0] = leb__DotProduct(3, m[0], attributeVector);
        attributeArray[i][1] = leb__DotProduct(3, m[1], attributeVector);
        attributeArray[i][2] = leb__DotProduct(3, m[2], attributeVector);
    }
}

#endif // LEB_IMPLEMENTATION
