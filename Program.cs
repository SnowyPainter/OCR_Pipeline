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
                Upscale = 2.0,
                ApplyBinarize = true,
                ApplyDenoise = true,
                MinArea = 120,
                MaxAreaRatio = 0.85,
                MinAspectRatio = 0.05,
                MaxAspectRatio = 20.0
            }
        );

        static async Task Main(string[] args)
        {
            Console.WriteLine("▶ 전역 마우스 후킹 시작");
            Console.WriteLine("   - 마우스 클릭: 해당 위치 기준 영역 캡처 → OpenCV 파이프라인 처리");
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

                    // 캡처 크기는 필요 시 조정(가로 300, 세로 150)
                    using var bmp = CaptureAround(x, y, 300, 150, out string rawPath);
                    Console.WriteLine($"✓ Raw saved: {rawPath}");

                    // 출력 루트 디렉터리 생성
                    string runRoot = Path.Combine(
                        AppContext.BaseDirectory,
                        "outputs",
                        DateTime.Now.ToString("yyyyMMdd_HHmmss")
                    );
                    Directory.CreateDirectory(runRoot);

                    // 파이프라인 처리
                    var result = _pipeline.Process(bmp, runRoot);

                    using var ocr = new OCRService(new OCRService.Options
                    {
                        Languages = "kor+eng",
                        Psm = PageSegMode.SingleLine,          // 한 줄만이면 SingleLine, UI 버튼이면 SparseText 등으로 바꿔도 OK
                        CharWhitelist = null,            // 숫자만 필요하면 "0123456789-:()" 등
                        TessdataPath = null,             // 기본: ./tessdata
                        EngineMode = EngineMode.LstmOnly
                    });

                    var ocrResult = ocr.RecognizeAsync(result.PreprocessedPath);
                    var trimmed = OCRService.TrimConcatenatedText(ocrResult.Text);
                    Console.WriteLine($"✓ OCR: {ocrResult.Text} -> {trimmed}");
                    ocr.Dispose();
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
