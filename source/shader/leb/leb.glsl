
// data-structures
struct leb_DiamondParent
{
    cbt_Node base;
    cbt_Node top;
};
leb_DiamondParent leb_DecodeDiamondParent       (in const cbt_Node node);
leb_DiamondParent leb_DecodeDiamondParent_Square(in const cbt_Node node);

// manipulation
void leb_SplitNode       (const int cbtID, in const cbt_Node node);
void leb_SplitNode_Square(const int cbtID, in const cbt_Node node);
void leb_MergeNode(const int cbtID,
                   in const cbt_Node node,
                   in const leb_DiamondParent diamond);
void leb_MergeNode_Square(const int cbtID,
                          in const cbt_Node node,
                          in const leb_DiamondParent diamond);

// subdivision routine O(depth)
vec3   leb_DecodeNodeAttributeArray       (in const cbt_Node node, in const vec3 data);
mat2x3 leb_DecodeNodeAttributeArray       (in const cbt_Node node, in const mat2x3 data);
mat3x3 leb_DecodeNodeAttributeArray       (in const cbt_Node node, in const mat3x3 data);
mat4x3 leb_DecodeNodeAttributeArray       (in const cbt_Node node, in const mat4x3 data);
vec3   leb_DecodeNodeAttributeArray_Square(in const cbt_Node node, in const vec3 data);
mat2x3 leb_DecodeNodeAttributeArray_Square(in const cbt_Node node, in const mat2x3 data);
mat3x3 leb_DecodeNodeAttributeArray_Square(in const cbt_Node node, in const mat3x3 data);
mat4x3 leb_DecodeNodeAttributeArray_Square(in const cbt_Node node, in const mat4x3 data);


// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

struct leb_a_SameDepthNeighborIDs {
    uint left, right, edge, node;
};

leb_a_SameDepthNeighborIDs
leb_a_CreateSameDepthNeighborIDs(uint left, uint right, uint edge, uint node)
{
    leb_a_SameDepthNeighborIDs neighborIDs;

    neighborIDs.left = left;
    neighborIDs.right = right;
    neighborIDs.edge = edge;
    neighborIDs.node = node;

    return neighborIDs;
}

leb_DiamondParent
leb_a_CreateDiamondParent(in const cbt_Node base, in const cbt_Node top)
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
uint leb_a_GetBitValue(const uint bitField, int bitID)
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
 * The _reserved channel stores NodeID, which is recquired for applying the
 * rules.
 *
 */
leb_a_SameDepthNeighborIDs
leb_a_SplitNodeIDs(in const leb_a_SameDepthNeighborIDs nodeIDs, uint splitBit)
{
#if 1 // branchless version
    uint b = splitBit;
    uint c = splitBit ^ 1u;
    bool cb = bool(c);
    uvec4 idArray = uvec4(nodeIDs.left, nodeIDs.right, nodeIDs.edge, nodeIDs.node);
    return leb_a_CreateSameDepthNeighborIDs(
        (idArray[2 + b] << 1u) | uint(cb && bool(idArray[2 + b])),
        (idArray[2 + c] << 1u) | uint(cb && bool(idArray[2 + c])),
        (idArray[b    ] << 1u) | uint(cb && bool(idArray[b    ])),
        (idArray[3    ] << 1u) | b
    );

#else
    uint n1 = nodeIDs.left, n2 = nodeIDs.right,
         n3 = nodeIDs.edge, n4 = nodeIDs._reserved;
    uint b2 = (n2 == 0u) ? 0u : 1u,
         b3 = (n3 == 0u) ? 0u : 1u;

    if (splitBit == 0u) {
        return leb_a_SameDepthNeighborIDs(
            n4 << 1 | 1, n3 << 1 | b3, n2 << 1 | b2, n4 << 1
        );
    } else {
        return leb_a_SameDepthNeighborIDs(
            n3 << 1    , n4 << 1     , n1 << 1     , n4 << 1 | 1
        );
    }
#endif
}


/*******************************************************************************
 * DecodeNodeNeighborIDs -- Decodes the IDs of the cbt_Nodes neighbor to node
 *
 * The IDs are associated to the depth of the input node. As such, they
 * don't necessarily exist in the LEB subdivision.
 *
 */
leb_a_SameDepthNeighborIDs
leb_DecodeSameDepthNeighborIDs(in const cbt_Node node)
{
    leb_a_SameDepthNeighborIDs nodeIDs =
        leb_a_CreateSameDepthNeighborIDs(0u, 0u, 0u, 1u);

    for (int bitID = node.depth - 1; bitID >= 0; --bitID) {
        nodeIDs = leb_a_SplitNodeIDs(nodeIDs, leb_a_GetBitValue(node.id, bitID));
    }

    return nodeIDs;
}

leb_a_SameDepthNeighborIDs
leb_DecodeSameDepthNeighborIDs_Square(in const cbt_Node node)
{
    uint b = leb_a_GetBitValue(node.id, max(0, node.depth - 1));
    leb_a_SameDepthNeighborIDs nodeIDs =
        leb_a_CreateSameDepthNeighborIDs(0u, 0u, 3u - b, 2u + b);

    for (int bitID = node.depth - 2; bitID >= 0; --bitID) {
        nodeIDs = leb_a_SplitNodeIDs(nodeIDs, leb_a_GetBitValue(node.id, bitID));
    }

    return nodeIDs;
}


/*******************************************************************************
 * EdgeNeighbor -- Computes the neighbour of the input node wrt to its longest edge
 *
 */
cbt_Node leb_a_EdgeNeighbor(in const cbt_Node node)
{
    uint nodeID = leb_DecodeSameDepthNeighborIDs(node).edge;

    return cbt_CreateNode(nodeID, (nodeID == 0u) ? 0 : node.depth);
}

cbt_Node leb_a_EdgeNeighbor_Square(in const cbt_Node node)
{
    uint nodeID = leb_DecodeSameDepthNeighborIDs_Square(node).edge;

    return cbt_CreateNode(nodeID, (nodeID == 0u) ? 0 : node.depth);
}


/*******************************************************************************
 * SplitNode -- Splits a node while producing a conforming LEB
 *
 */
void leb_SplitNode(const int cbtID, in const cbt_Node node)
{
    if (!cbt_IsCeilNode(cbtID, node)) {
        const uint minNodeID = 1u;
        cbt_Node nodeIterator = node;

        cbt_SplitNode(cbtID, nodeIterator);
        nodeIterator = leb_a_EdgeNeighbor(nodeIterator);

        while (nodeIterator.id > minNodeID) {
            cbt_SplitNode(cbtID, nodeIterator);
            nodeIterator = cbt_ParentNode_Fast(nodeIterator);
            cbt_SplitNode(cbtID, nodeIterator);
            nodeIterator = leb_a_EdgeNeighbor(nodeIterator);
        }
    }
}

void leb_SplitNode_Square(const int cbtID, in const cbt_Node node)
{
    if (!cbt_IsCeilNode(cbtID, node)) {
        const uint minNodeID = 1u;
        cbt_Node nodeIterator = node;

        cbt_SplitNode(cbtID, nodeIterator);
        nodeIterator = leb_a_EdgeNeighbor_Square(nodeIterator);

        while (nodeIterator.id > minNodeID) {
            cbt_SplitNode(cbtID, nodeIterator);
            nodeIterator = cbt_ParentNode_Fast(nodeIterator);

            if (nodeIterator.id > minNodeID) {
                cbt_SplitNode(cbtID, nodeIterator);
                nodeIterator = leb_a_EdgeNeighbor_Square(nodeIterator);
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
leb_DiamondParent leb_DecodeDiamondParent(in const cbt_Node node)
{
    cbt_Node parentNode = cbt_ParentNode_Fast(node);
    uint edgeNeighborID = leb_DecodeSameDepthNeighborIDs(parentNode).edge;
    cbt_Node edgeNeighborNode = cbt_CreateNode(
        edgeNeighborID > 0u ? edgeNeighborID : parentNode.id,
        parentNode.depth
    );

    return leb_a_CreateDiamondParent(parentNode, edgeNeighborNode);
}

leb_DiamondParent leb_DecodeDiamondParent_Square(in const cbt_Node node)
{
    cbt_Node parentNode = cbt_ParentNode_Fast(node);
    uint edgeNeighborID = leb_DecodeSameDepthNeighborIDs_Square(parentNode).edge;
    cbt_Node edgeNeighborNode = cbt_CreateNode(
        edgeNeighborID > 0u ? edgeNeighborID : parentNode.id,
        parentNode.depth
    );

    return leb_a_CreateDiamondParent(parentNode, edgeNeighborNode);
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
leb_a_HasDiamondParent(
    const int cbtID,
    in const leb_DiamondParent diamondParent
) {
    bool canMergeBase = cbt_HeapRead(cbtID, diamondParent.base) <= 2u;
    bool canMergeTop  = cbt_HeapRead(cbtID, diamondParent.top) <= 2u;

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
    const int cbtID,
    in const cbt_Node node,
    in const leb_DiamondParent diamondParent
) {
    if (!cbt_IsRootNode(node) && leb_a_HasDiamondParent(cbtID, diamondParent)) {
        cbt_MergeNode(cbtID, node);
    }
}

void
leb_MergeNode_Square(
    const int cbtID,
    in const cbt_Node node,
    in const leb_DiamondParent diamondParent
) {
    if ((node.depth > 1) && leb_a_HasDiamondParent(cbtID, diamondParent)) {
        cbt_MergeNode(cbtID, node);
    }
}


/*******************************************************************************
 * SplitMatrix3x3 -- Computes a LEB splitting matrix from a split bit
 *
 */
mat3 leb_a_SplittingMatrix(uint splitBit)
{
    float b = float(splitBit);
    float c = 1.0f - b;

    return transpose(mat3(
        c   , b   , 0.0f,
        0.5f, 0.0f, 0.5f,
        0.0f,    c,    b
    ));
}


/*******************************************************************************
 * SquareMatrix3x3 -- Computes the matrix that affects the triangle to the square
 *
 */
mat3 leb_a_SquareMatrix(uint quadBit)
{
    float b = float(quadBit);
    float c = 1.0f - b;

    return transpose(mat3(
        c, 0.0f, b,
        b, c   , b,
        b, 0.0f, c
    ));
}


/*******************************************************************************
 * WindingMatrix -- Computes the matrix that garantees that triangles have same winding
 *
 */
mat3 leb_a_WindingMatrix(uint mirrorBit)
{
    float b = float(mirrorBit);
    float c = 1.0f - b;

    return mat3(
        c, 0.0f, b,
        0, 1.0f, 0,
        b, 0.0f, c
    );
}


/*******************************************************************************
 * DecodeTransformationMatrix -- Computes the splitting matrix associated to a LEB
 * node
 *
 */
mat3 leb_a_DecodeTransformationMatrix(in const cbt_Node node)
{
    mat3 xf = mat3(1.0f);

    for (int bitID = node.depth - 1; bitID >= 0; --bitID) {
        xf = leb_a_SplittingMatrix(leb_a_GetBitValue(node.id, bitID)) * xf;
    }

    return leb_a_WindingMatrix(node.depth & 1) * xf;
}

mat3 leb_a_DecodeTransformationMatrix_Square(in const cbt_Node node)
{
    int bitID = max(0, node.depth - 1);
    mat3 xf = leb_a_SquareMatrix(leb_a_GetBitValue(node.id, bitID));

    for (bitID = node.depth - 2; bitID >= 0; --bitID) {
        xf = leb_a_SplittingMatrix(leb_a_GetBitValue(node.id, bitID)) * xf;
    }

    return leb_a_WindingMatrix((node.depth ^ 1) & 1) * xf;
}


/*******************************************************************************
 * DecodeNodeAttributeArray -- Compute the triangle attributes at the input node
 *
 */
vec3 leb_DecodeNodeAttributeArray(in const cbt_Node node, in const vec3 data)
{
    return leb_a_DecodeTransformationMatrix(node) * data;
}

mat2x3 leb_DecodeNodeAttributeArray(in const cbt_Node node, in const mat2x3 data)
{
    return leb_a_DecodeTransformationMatrix(node) * data;
}

mat3x3 leb_DecodeNodeAttributeArray(in const cbt_Node node, in const mat3x3 data)
{
    return leb_a_DecodeTransformationMatrix(node) * data;
}

mat4x3 leb_DecodeNodeAttributeArray(in const cbt_Node node, in const mat4x3 data)
{
    return leb_a_DecodeTransformationMatrix(node) * data;
}

vec3 leb_DecodeNodeAttributeArray_Square(in const cbt_Node node, in const vec3 data)
{
    return leb_a_DecodeTransformationMatrix_Square(node) * data;
}

mat2x3 leb_DecodeNodeAttributeArray_Square(in const cbt_Node node, in const mat2x3 data)
{
    return leb_a_DecodeTransformationMatrix_Square(node) * data;
}

mat3x3 leb_DecodeNodeAttributeArray_Square(in const cbt_Node node, in const mat3x3 data)
{
    return leb_a_DecodeTransformationMatrix_Square(node) * data;
}

mat4x3 leb_DecodeNodeAttributeArray_Square(in const cbt_Node node, in const mat4x3 data)
{
    return leb_a_DecodeTransformationMatrix_Square(node) * data;
}
