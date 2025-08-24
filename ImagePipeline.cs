using System;
using System.Collections.Generic;
using System.Drawing;               // Bitmap 용
using System.IO;
using OpenCvSharp;                  // Mat, Cv2 등
using OpenCvSharp.Extensions;       // Bitmap <-> Mat
// 네임스페이스 별칭
using SD = System.Drawing;
using Cv = OpenCvSharp;

namespace OCRPipeline
{
    public sealed class ImagePipeline
    {
        public sealed class Options
        {
            public double MinArea { get; init; } = 100;
            public double MaxAreaRatio { get; init; } = 0.85;
            public double MinAspectRatio { get; init; } = 0.05;
            public double MaxAspectRatio { get; init; } = 20.0;

            public double Upscale { get; init; } = 2.0;
            public bool ApplyBinarize { get; init; } = true;
            public bool ApplyDenoise { get; init; } = true;
            public bool BoostThinText { get; init; } = true;
            public int BoostScale { get; init; } = 2;
        }

        public sealed class Result
        {
            public string PreprocessedPath { get; set; } = "";
        }

        private readonly Options _opt;

        public ImagePipeline(Options? options = null)
        {
            _opt = options ?? new Options();
            Console.WriteLine("ImagePipeline initialized (OpenCV)");
        }

        public Result Process(Bitmap bitmap, string outputRoot)
        {
            Directory.CreateDirectory(outputRoot);
            var partsDir = Path.Combine(outputRoot, "parts");
            var preDir = Path.Combine(outputRoot, "pre");
            Directory.CreateDirectory(partsDir);
            Directory.CreateDirectory(preDir);

            using Cv.Mat src = bitmap.ToMat(); // BGR
            int imgW = src.Width;
            int imgH = src.Height;
            double imgArea = imgW * imgH;

            double imgCenterX = src.Width / 2;
            double imgCenterY = src.Height / 2;

            using Cv.Mat gray = new();
            Cv2.CvtColor(src, gray, Cv.ColorConversionCodes.BGR2GRAY);

            using Cv.Mat blur = new();
            Cv2.GaussianBlur(gray, blur, new Cv.Size(3, 3), 0);

            using Cv.Mat bw = new();
            Cv2.AdaptiveThreshold(
                blur, bw, 255,
                Cv.AdaptiveThresholdTypes.MeanC,
                Cv.ThresholdTypes.BinaryInv,
                21, 10
            );

            int approxCharH = Math.Max(12, gray.Rows / 60);          // 대략치
            int kernelW = Math.Max(approxCharH / 2, 8);               // 글자 간 간격만 살짝 메울 정도
            using (var hKernel = Cv2.GetStructuringElement(Cv.MorphShapes.Rect, new Cv.Size(kernelW, 1)))
                Cv2.Dilate(bw, bw, hKernel, iterations: 1);

            // === 여기 타입이 중요! OpenCvSharp.Point[][]
            Cv.Point[][] contours;
            Cv.HierarchyIndex[] hierarchy;
            Cv2.FindContours(
                bw,
                out contours,
                out hierarchy,
                Cv.RetrievalModes.External,
                Cv.ContourApproximationModes.ApproxSimple
            );

            using Cv.Mat annotated = src.Clone();

            var rects = new List<Cv.Rect>();
            foreach (var cnt in contours)
            {
                if (cnt.Length < 3) continue;

                Cv.Rect rect = Cv2.BoundingRect(cnt);
                double area = rect.Width * rect.Height;
                if (area < _opt.MinArea) continue;
                if (area > imgArea * _opt.MaxAreaRatio) continue;

                double ar = (double)rect.Width / Math.Max(1, rect.Height);
                if (ar < _opt.MinAspectRatio || ar > _opt.MaxAspectRatio) continue;

                rects.Add(rect);
            }

            
            rects.Sort((a, b) =>
            {
                int y = a.Y.CompareTo(b.Y);
                return Math.Abs(a.Y - b.Y) < 10 ? a.X.CompareTo(b.X) : y;
            });
            rects = MergeRectsByLine(rects, yOverlapThresh: 0.5, xGapFactor: 0.6);

            var centerRect = GetCenterRect(rects, imgCenterX, imgCenterY);

            var now = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            var result = new Result();
            if (centerRect.HasValue)
            {
                Cv2.Rectangle(annotated, centerRect.Value, Cv.Scalar.Red, 2);
                using Cv.Mat roi = new Cv.Mat(src, centerRect.Value);
                using Cv.Mat pre = PreprocessForOCR(roi);

                string prePath = Path.Combine(preDir, $"part_{now}_pre.png");
                Cv2.ImWrite(prePath, pre);

                result.PreprocessedPath = prePath;

                return result;
            }

            Cv2.ImWrite(Path.Combine(partsDir, $"annotated_{now}.png"), annotated);
            
            return result;
            
        }

        private Cv.Mat PreprocessForOCR(Cv.Mat roiBgr)
        {
            // 0) 업스케일 (얇은 획 보강의 전제)
            Cv.Mat up = new();
            double scale = Math.Max(1.0, _opt.Upscale * (_opt.BoostThinText ? _opt.BoostScale : 1));
            Cv2.Resize(roiBgr, up, new Cv.Size(), scale, scale, Cv.InterpolationFlags.Lanczos4);

            // 1) 그레이 + 대비 강화 (CLAHE)
            using var gray = new Cv.Mat();
            Cv2.CvtColor(up, gray, Cv.ColorConversionCodes.BGR2GRAY);

            using var claheOut = new Cv.Mat();
            using var clahe = Cv.CLAHE.Create(clipLimit: 2.0, tileGridSize: new Cv.Size(8, 8));
            clahe.Apply(gray, claheOut);

            // 2) 소프트 디노이즈 (얇은 획 보존 위해 과도하지 않게)
            Cv.Mat denoised = new();
            if (_opt.ApplyDenoise)
                Cv2.GaussianBlur(claheOut, denoised, new Cv.Size(3, 3), 0.0);
            else
                denoised = claheOut.Clone();

            // 3) 지역 이진화 (얇은 글자에 유리)
            //    - AdaptiveThreshold 또는 ximgproc의 Niblack/Sauvola가 있으면 Sauvola 권장
            Cv.Mat bw = new();
            Cv2.AdaptiveThreshold(
                denoised, bw, 255,
                Cv.AdaptiveThresholdTypes.MeanC,
                Cv.ThresholdTypes.Binary,  // 얇은 폰트는 Binary가 더 안정적인 경우 多
                21, 5
            );

            if (_opt.BoostThinText)
            {
                // 4) 미세 팽창(획 두껍게). 가로 커널이 보통 안정적.
                int k = Math.Max(1, bw.Rows / 200);  // 영상 크기 따라 1~2픽셀 정도
                using var kernel = Cv2.GetStructuringElement(Cv.MorphShapes.Rect, new Cv.Size(1 + 2 * k, 1));
                Cv2.Dilate(bw, bw, kernel, iterations: 1);

                // 5) 점 노이즈 제거
                using var small = Cv2.GetStructuringElement(Cv.MorphShapes.Rect, new Cv.Size(2, 2));
                Cv2.MorphologyEx(bw, bw, Cv.MorphTypes.Open, small, iterations: 1);
            }

            up.Dispose();
            denoised.Dispose();
            return bw;
        }


        private static List<Cv.Rect> MergeRectsByLine(IList<Cv.Rect> boxes, double yOverlapThresh = 0.5, double xGapFactor = 0.5)
        {
            var sorted = new List<Cv.Rect>(boxes);
            sorted.Sort((a, b) => a.Y.CompareTo(b.Y));

            var merged = new List<Cv.Rect>();
            var line = new List<Cv.Rect>();

            void FlushLine()
            {
                if (line.Count == 0) return;

                line.Sort((a, b) => a.X.CompareTo(b.X));
                var cur = line[0];
                for (int i = 1; i < line.Count; i++)
                {
                    var nxt = line[i];
                    int gap = nxt.X - (cur.X + cur.Width);
                    int h = Math.Min(cur.Height, nxt.Height);
                    // 수평 간격이 라인 높이의 xGapFactor 배 이하이면 같은 덩어리로 본다
                    if (gap <= h * xGapFactor)
                    {
                        cur = new Cv.Rect(
                            Math.Min(cur.X, nxt.X),
                            Math.Min(cur.Y, nxt.Y),
                            Math.Max(cur.X + cur.Width, nxt.X + nxt.Width) - Math.Min(cur.X, nxt.X),
                            Math.Max(cur.Y + cur.Height, nxt.Y + nxt.Height) - Math.Min(cur.Y, nxt.Y)
                        );
                    }
                    else
                    {
                        merged.Add(cur);
                        cur = nxt;
                    }
                }
                merged.Add(cur);
                line.Clear();
            }

            foreach (var r in sorted)
            {
                if (line.Count == 0)
                {
                    line.Add(r);
                    continue;
                }
                // 같은 라인인지: 수직 겹침 비율로 판단
                var refBox = line[0];
                int yTop = Math.Max(refBox.Y, r.Y);
                int yBot = Math.Min(refBox.Y + refBox.Height, r.Y + r.Height);
                int overlap = Math.Max(0, yBot - yTop);
                double overlapRatio = (double)overlap / Math.Min(refBox.Height, r.Height);

                if (overlapRatio >= yOverlapThresh)
                {
                    line.Add(r);
                }
                else
                {
                    FlushLine();
                    line.Add(r);
                }
            }
            FlushLine();
            return merged;
        }

        private static Cv.Rect? GetCenterRect(List<Cv.Rect> rects, double imgCenterX, double imgCenterY)
        {
            if (rects == null || rects.Count == 0)
                return null;

            Cv.Rect? bestRect = null;
            double bestDist = double.MaxValue;

            foreach (var r in rects)
            {
                // 사각형 중심 좌표
                double cx = r.X + r.Width / 2.0;
                double cy = r.Y + r.Height / 2.0;

                // 이미지 센터와 거리 계산
                double dist = Math.Sqrt(Math.Pow(cx - imgCenterX, 2) + Math.Pow(cy - imgCenterY, 2));

                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestRect = r;
                }
            }

            return bestRect;
        }
    }
}
