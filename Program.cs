using System;
using System.Collections.Generic;
using System.Drawing;               // Bitmap
using System.Drawing.Imaging;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Forms;         // VirtualScreen
using OCRPipeline;
using SharpHook;
using Tesseract;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using Cv = OpenCvSharp;
using System.Numerics;

namespace OCRPipeline
{
    class Program
    {
        private static SimpleGlobalHook? _hook;
        private static readonly object _lock = new();
        private static bool _busy = false;

        private static readonly ImagePipeline _pipeline = new(
            new ImagePipeline.Options
            {
                // 작은 글자 대비를 위해 업스케일↑
                Upscale = 3.0,

                // 전처리 기본값 유지
                ApplyBinarize = true,
                ApplyDenoise = true,

                // 얇은 글자(10~12px) 보강
                BoostThinText = true,
                BoostScale = 4,   // 2~3 사이 튜닝 포인트

                // 박스 필터 튜닝
                MinArea = 60,   // ↓ 120 → 60 (작은 글자 조각 통과)
                MaxAreaRatio = 0.75, // ↓ 0.85 → 0.75 (버튼 전체 박스 억제)
                MinAspectRatio = 0.08, // ↑ 0.05 → 0.08 (너무 가느다란 노이즈 컷)
                MaxAspectRatio = 15.0  // ↓ 20.0 → 15.0 (가로로 과도하게 긴 박스 컷)
            }
        );


        static async Task Main(string[] args)
        {
            Dpi.EnablePerMonitorV2();

            // outputs 폴더 초기화
            string outputsDir = Path.Combine(AppContext.BaseDirectory, "outputs");
            if (Directory.Exists(outputsDir))
            {
                Directory.Delete(outputsDir, true);
                Console.WriteLine("✓ Cleared outputs folder");
            }

            Console.WriteLine("▶ 전역 마우스 후킹 시작");
            Console.WriteLine("   - 마우스 클릭: 해당 위치 기준 영역 캡처 → 점진적 ROI 축소로 5회 OCR 처리");
            Console.WriteLine("   - ESC: 종료\n");

            _hook = new SimpleGlobalHook();

            // 마우스 클릭 이벤트
            _hook.MousePressed += OnMousePressed;

            // 훅 실행
            var hookTask = _hook.RunAsync();

            // ESC로 종료
            while (true)
            {
                if (Console.KeyAvailable && Console.ReadKey(true).Key == ConsoleKey.Escape)
                    break;
                await Task.Delay(50);
            }

            _hook.Dispose();
            Console.WriteLine("▶ 종료");
        }

        private static void OnMousePressed(object? sender, MouseHookEventArgs e)
        {
            // 멀티클릭 쓰로틀링(처리 중이면 무시)
            lock (_lock)
            {
                if (_busy) return;
                _busy = true;
            }

            Task.Run(() =>
            {
                try
                {
                    int x = (int)e.Data.X;
                    int y = (int)e.Data.Y;

                    const int captureWidth = 300;
                    const int captureHeight = 100;

                    var (bmp, canvasToScreen) = CaptureUtil.CaptureAroundDpiSafe(x, y, captureWidth, captureHeight);
                    using var src = bmp.ToMat();

                    // 출력 루트 디렉터리 생성
                    string runRoot = Path.Combine(
                        AppContext.BaseDirectory,
                        "outputs",
                        DateTime.Now.ToString("yyyyMMdd_HHmmss_fff")
                    );
                    Directory.CreateDirectory(runRoot);

                    // 캡처 원본 저장 (기존과 동일한 경로 규칙)
                    var rawsDir = Path.Combine(AppContext.BaseDirectory, "raws");
                    Directory.CreateDirectory(rawsDir);
                    var rawPath = Path.Combine(
                        rawsDir,
                        $"raw_{DateTime.Now:yyyyMMdd_HHmmss_fff}_{x}x{y}.png"
                    );
                    bmp.Save(rawPath, System.Drawing.Imaging.ImageFormat.Png);

                    var result = _pipeline.Process(src, runRoot, canvasToScreen);
                    var trimmed = OCRService.TrimConcatenatedText(result.BestText);

                    Console.WriteLine($"\n=== 최종 결과 ===");
                    Console.WriteLine($"✓ 최고 OCR: \"{result.BestText}\" ({result.BestConfidence:F1}%)");
                    Console.WriteLine($"✓ 정리된 텍스트: \"{trimmed}\"");
                    Console.WriteLine($"✓ 스텝별 이미지 {result.StepPaths.Count}개 저장됨");
                    Console.WriteLine($"✓ 최종 결과: {result.FinalAnnotatedPath}\n");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"✗ Pipeline error: {ex.Message}");
                }
                finally
                {
                    lock (_lock) { _busy = false; }
                }
            });
        }
        /// <summary>
        /// 중심 좌표 기준으로 width x height 화면 캡처
        /// </summary>
        private static Bitmap CaptureAround(int centerX, int centerY, int width, int height, out string savedPath)
        {
            var vs = SystemInformation.VirtualScreen;

            int left = centerX - width / 2;
            int top = centerY - height / 2;

            left = Math.Max(vs.Left, Math.Min(left, vs.Right - width));
            top = Math.Max(vs.Top, Math.Min(top, vs.Bottom - height));

            // ⚠️ 반환할 Bitmap은 Dispose하면 안 되므로 using 제거
            var bmp = new Bitmap(width, height);
            using (var g = Graphics.FromImage(bmp))
            {
                g.CopyFromScreen(left, top, 0, 0, new System.Drawing.Size(width, height), CopyPixelOperation.SourceCopy);
            }

            var dir = Path.Combine(AppContext.BaseDirectory, "raws");
            Directory.CreateDirectory(dir);

            savedPath = Path.Combine(
                dir,
                $"raw_{DateTime.Now:yyyyMMdd_HHmmss_fff}_{centerX}x{centerY}.png"
            );
            bmp.Save(savedPath, System.Drawing.Imaging.ImageFormat.Png);
            return bmp;
        }
    }
}
