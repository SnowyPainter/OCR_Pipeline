using System;
using System.Drawing;
using System.Numerics;
using System.Runtime.InteropServices;
internal static class Dpi
{
    // DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = (IntPtr)(-4)
    private static readonly IntPtr PMV2 = (IntPtr)(-4);

    [DllImport("user32.dll")]
    private static extern bool SetProcessDpiAwarenessContext(IntPtr dpiContext);

    public static void EnablePerMonitorV2()
    {
        // 가능한 가장 먼저 호출 (Main 진입 직후 등)
        try { SetProcessDpiAwarenessContext(PMV2); } catch { /* ignore */ }
    }
}

public static class CaptureUtil
{
    [DllImport("user32.dll")] static extern int GetSystemMetrics(int nIndex);
    const int SM_XVIRTUALSCREEN = 76;
    const int SM_YVIRTUALSCREEN = 77;
    const int SM_CXVIRTUALSCREEN = 78;
    const int SM_CYVIRTUALSCREEN = 79;

    /// <summary>
    /// DPI‑세이프 Letterbox 캡처(디바이스 픽셀 기준).
    /// 요청한 (center, width, height) 영역을 고정 해상도 캔버스에 복사하고,
    /// 화면 밖은 패딩(Black)으로 채운다. 반환 변환 T는 캔버스 좌표→스크린 좌표(디바이스 픽셀) 변환.
    /// </summary>
    public static (Bitmap bmp, Matrix3x2 transform) CaptureAroundDpiSafe(
        int centerX, int centerY, int width, int height, Color? pad = null)
    {
        // 가상 데스크톱(디바이스 픽셀) 경계
        int vsLeft   = GetSystemMetrics(SM_XVIRTUALSCREEN);
        int vsTop    = GetSystemMetrics(SM_YVIRTUALSCREEN);
        int vsWidth  = GetSystemMetrics(SM_CXVIRTUALSCREEN);
        int vsHeight = GetSystemMetrics(SM_CYVIRTUALSCREEN);
        int vsRight  = vsLeft + vsWidth;
        int vsBottom = vsTop + vsHeight;

        // 요청 사각형(디바이스 픽셀)
        int left   = centerX - width  / 2;
        int top    = centerY - height / 2;
        int right  = left + width;
        int bottom = top  + height;

        // 실제로 스크린과 교차하는 소스 사각형
        int srcLeft   = Math.Max(left,   vsLeft);
        int srcTop    = Math.Max(top,    vsTop);
        int srcRight  = Math.Min(right,  vsRight);
        int srcBottom = Math.Min(bottom, vsBottom);

        int srcW = Math.Max(0, srcRight - srcLeft);
        int srcH = Math.Max(0, srcBottom - srcTop);

        // 교차영역이 캔버스 내 어디에 들어가야 하는지(목표 캔버스 좌표)
        int dstX = srcLeft - left;   // 보정된 시작 오프셋
        int dstY = srcTop  - top;

        var bmp = new Bitmap(width, height, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
        using (var g = Graphics.FromImage(bmp))
        {
            g.CompositingMode = System.Drawing.Drawing2D.CompositingMode.SourceCopy;
            g.SmoothingMode   = System.Drawing.Drawing2D.SmoothingMode.None;
            g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.NearestNeighbor;
            g.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.Half;

            g.Clear(pad ?? Color.Black);

            if (srcW > 0 && srcH > 0)
            {
                // 주의: 여기서의 좌표/크기는 모두 디바이스 픽셀이어야 함
                g.CopyFromScreen(srcLeft, srcTop, dstX, dstY, new Size(srcW, srcH), CopyPixelOperation.SourceCopy);
            }
        }

        // 캔버스(px) → 스크린(px) 변환(디바이스 픽셀): T(x,y) = (x+left, y+top)
        var T = Matrix3x2.CreateTranslation(left, top);
        return (bmp, T);
    }
}
