#include "sequencer.h"

const char* TimelineSequencer::GetItemTypeName(int typeIndex) const
{
	CHECK(typeIndex < GetItemTypeCount());
	return toString(static_cast<ESequencerType>(typeIndex));
}

const char* TimelineSequencer::GetItemLabel(int index) const
{
    static char tmps[512];
    snprintf(tmps, 512, "[%02d] %s", index, toString(m_items[index].type));
    return tmps;
}

void TimelineSequencer::Get(int index, int** start, int** end, int* type, unsigned int* color)
{
    SequencerItem& item = m_items[index];

    if (color)
    {
        *color = 0xFFAA8080; // same color for everyone, return color based on type
    }
    if (start)
    {
        *start = &item.frameStart;
    }
    if (end)
    {
        *end = &item.frameEnd;
    }

    if (type)
    {
        *type = (int)item.type;
    }
}

void TimelineSequencer::Add(int type)
{ 
    m_items.push_back(SequencerItem{
        .type = ESequencerType(type),
        .frameStart = 0,
        .frameEnd   = 10,
        .bExpanded  = false
    });
};

void TimelineSequencer::Del(int index)
{ 
    m_items.erase(m_items.begin() + index);
}

void TimelineSequencer::Duplicate(int index)
{
    m_items.push_back(m_items[index]);
}

size_t TimelineSequencer::GetCustomHeight(int index)
{
    return m_items[index].bExpanded ? 300 : 0;
}

void TimelineSequencer::DoubleClick(int index)
{
    if (m_items[index].bExpanded)
    {
        m_items[index].bExpanded = false;
        return;
    }

    for (auto& item : m_items)
    {
        item.bExpanded = false;
    }

    m_items[index].bExpanded = !m_items[index].bExpanded;
}

void TimelineSequencer::CustomDraw(
    int index, 
    ImDrawList* draw_list, 
    const ImRect& rc, 
    const ImRect& legendRect, 
    const ImRect& clippingRect, 
    const ImRect& legendClippingRect)
{
    static const char* labels[] = { "Translation", "Rotation" , "Scale" };

    rampEdit.maxValue = ImVec2(float(m_frameMax), 1.f);
    rampEdit.minValue = ImVec2(float(m_frameMin), 0.f);
    draw_list->PushClipRect(legendClippingRect.Min, legendClippingRect.Max, true);
    for (int i = 0; i < 3; i++)
    {
        ImVec2 pta(legendRect.Min.x + 30, legendRect.Min.y + i * 14.f);
        ImVec2 ptb(legendRect.Max.x, legendRect.Min.y + (i + 1) * 14.f);
        draw_list->AddText(pta, rampEdit.bVisible[i] ? 0xFFFFFFFF : 0x80FFFFFF, labels[i]);
        if (ImRect(pta, ptb).Contains(ImGui::GetMousePos()) && ImGui::IsMouseClicked(0))
        {
            rampEdit.bVisible[i] = !rampEdit.bVisible[i];
        }
    }
    draw_list->PopClipRect();

    ImGui::SetCursorScreenPos(rc.Min);
    ImCurveEdit::Edit(rampEdit, rc.Max - rc.Min, 137 + index, &clippingRect);
}

void TimelineSequencer::CustomDrawCompact(int index, ImDrawList* draw_list, const ImRect& rc, const ImRect& clippingRect)
{
    rampEdit.maxValue = ImVec2(float(m_frameMax), 1.f);
    rampEdit.minValue = ImVec2(float(m_frameMin), 0.f);
    draw_list->PushClipRect(clippingRect.Min, clippingRect.Max, true);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < rampEdit.pointCount[i]; j++)
        {
            float p = rampEdit.points[i][j].x;
            if (p < m_items[index].frameStart || p > m_items[index].frameEnd)
            {
                continue;
            }

            float r = (p - m_frameMin) / float(m_frameMax - m_frameMin);
            float x = ImLerp(rc.Min.x, rc.Max.x, r);
            draw_list->AddLine(ImVec2(x, rc.Min.y + 6), ImVec2(x, rc.Max.y - 4), 0xAA000000, 4.f);
        }
    }
    draw_list->PopClipRect();
}


///
RampEdit::RampEdit()
{
    points[0][0] = ImVec2(-10.f, 0);
    points[0][1] = ImVec2(20.f, 0.6f);
    points[0][2] = ImVec2(25.f, 0.2f);
    points[0][3] = ImVec2(70.f, 0.4f);
    points[0][4] = ImVec2(120.f, 1.f);
    pointCount[0] = 5;

    points[1][0] = ImVec2(-50.f, 0.2f);
    points[1][1] = ImVec2(33.f, 0.7f);
    points[1][2] = ImVec2(80.f, 0.2f);
    points[1][3] = ImVec2(82.f, 0.8f);
    pointCount[1] = 4;


    points[2][0] = ImVec2(40.f, 0);
    points[2][1] = ImVec2(60.f, 0.1f);
    points[2][2] = ImVec2(90.f, 0.82f);
    points[2][3] = ImVec2(150.f, 0.24f);
    points[2][4] = ImVec2(200.f, 0.34f);
    points[2][5] = ImVec2(250.f, 0.12f);
    pointCount[2] = 6;
    bVisible[0] = bVisible[1] = bVisible[2] = true;
    maxValue = ImVec2(1.f, 1.f);
    minValue = ImVec2(0.f, 0.f);
}

uint32_t RampEdit::GetCurveColor(size_t curveIndex)
{
    uint32_t cols[] = { 0xFF0000FF, 0xFF00FF00, 0xFFFF0000 };
    return cols[curveIndex];
}

int RampEdit::EditPoint(size_t curveIndex, int pointIndex, ImVec2 value)
{
    points[curveIndex][pointIndex] = ImVec2(value.x, value.y);

    SortValues(curveIndex);
    for (size_t i = 0; i < GetPointCount(curveIndex); i++)
    {
        if (points[curveIndex][i].x == value.x)
        {
            return (int)i;
        }
    }
    return pointIndex;
}

void RampEdit::AddPoint(size_t curveIndex, ImVec2 value)
{
    if (pointCount[curveIndex] >= 8)
    {
        return;
    }

    points[curveIndex][pointCount[curveIndex]++] = value;
    SortValues(curveIndex);
}

void RampEdit::SortValues(size_t curveIndex)
{
    auto b = std::begin(points[curveIndex]);
    auto e = std::begin(points[curveIndex]) + GetPointCount(curveIndex);
    std::sort(b, e, [](ImVec2 a, ImVec2 b) { return a.x < b.x; });
}