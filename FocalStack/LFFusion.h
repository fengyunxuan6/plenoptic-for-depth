/********************************************************************
file base:      LFFusion.h
author:         LZD
created:        2025/01/13
purpose:        虚拟深度图和全聚焦图像的融合生成
*********************************************************************/
#ifndef LFMVS_FUSION_H
#define LFMVS_FUSION_H

namespace LFMVS
{
    class LFFusion
    {
    public:
        LFFusion();

        ~LFFusion();

    public:
        void VirtualDepthFusion();

        void FullFocusedImageFusion();

    private:

    };

}
#endif // LFMVS_FUSION_H
