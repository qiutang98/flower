#include <imgui/gizmo/ImSequencer.h>
#include <imgui/gizmo/ImCurveEdit.h>
#include <util/util.h>


struct RampEdit : public ImCurveEdit::Delegate
{
    RampEdit();
    size_t GetCurveCount() { return 3; }

    bool IsVisible(size_t curveIndex) { return bVisible[curveIndex]; }
    size_t GetPointCount(size_t curveIndex)
    {
        return pointCount[curveIndex];
    }

    uint32_t GetCurveColor(size_t curveIndex);

    ImVec2* GetPoints(size_t curveIndex)
    {
        return points[curveIndex];
    }
    virtual ImCurveEdit::CurveType GetCurveType(size_t curveIndex) const { return ImCurveEdit::CurveSmooth; }
    virtual int EditPoint(size_t curveIndex, int pointIndex, ImVec2 value) override;
    virtual void AddPoint(size_t curveIndex, ImVec2 value) override;

    virtual ImVec2& GetMax() { return maxValue; }
    virtual ImVec2& GetMin() { return minValue; }
    virtual unsigned int GetBackgroundColor() { return 0; }

    ImVec2 points[3][8];
    size_t pointCount[3];
    bool bVisible[3];

    ImVec2 minValue;
    ImVec2 maxValue;

private:
    void SortValues(size_t curveIndex);
};

class TimelineSequencer : public ImSequencer::SequenceInterface
{
public:
    enum class ESequencerType
    {
        PMX = 0,
        Camera,

        Max
    };

    inline static const char* toString(ESequencerType type)
    {
        switch (type)
        {
            case ESequencerType::PMX:    return "PMX";
            case ESequencerType::Camera: return "Camera";
        }
        UN_IMPLEMENT();
        return nullptr;
    }

    struct SequencerItem
    {
        ESequencerType type;
        int frameStart;
        int frameEnd;
        bool bExpanded;
    };

    TimelineSequencer()
        : m_frameMin(0), m_frameMax(0)
    {

    }

    virtual int GetFrameMin()  const override { return m_frameMin; }
    virtual int GetFrameMax()  const override { return m_frameMax; }
    virtual int GetItemCount() const override { return static_cast<int>(m_items.size()); }

    virtual int GetItemTypeCount() const override { return static_cast<int>(ESequencerType::Max); }
    virtual const char* GetItemTypeName(int typeIndex) const override;

    virtual const char* GetItemLabel(int index) const override;
    virtual void Get(int index, int** start, int** end, int* type, unsigned int* color) override;


    virtual void Add(int type) override; 
    virtual void Del(int index) override;
    virtual void Duplicate(int index) override;


    virtual size_t GetCustomHeight(int index) override;

    virtual void DoubleClick(int index) override;

    virtual void CustomDraw(int index, ImDrawList* draw_list, const ImRect& rc, const ImRect& legendRect, const ImRect& clippingRect, const ImRect& legendClippingRect) override;

    virtual void CustomDrawCompact(int index, ImDrawList* draw_list, const ImRect& rc, const ImRect& clippingRect) override;

public:
    RampEdit rampEdit;

private:
    int m_frameMin;
    int m_frameMax;

    std::vector<SequencerItem> m_items;
};