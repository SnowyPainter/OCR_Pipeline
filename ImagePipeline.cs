using System;
using System.Collections.Generic;
using System.Drawing;               // Bitmap 용
using System.IO;
using OpenCvSharp;                  // Mat, Cv2 등
using OpenCvSharp.Extensions;       // Bitmap <-> Mat
// 네임스페이스 별칭
using SD = System.Drawing;
using Cv = OpenCvSharp;
using System.Numerics;

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
            public string FinalAnnotatedPath { get; set; } = "";
            public string BestText { get; set; } = "";
            public float BestConfidence { get; set; } = 0f;
            public List<string> StepPaths { get; set; } = new();
        }

        private readonly Options _opt;

        public ImagePipeline(Options? options = null)
        {
            _opt = options ?? new Options();
            Console.WriteLine("ImagePipeline initialized (OpenCV)");
        }

        public Result Process(Cv.Mat src, string outputRoot, Matrix3x2 canvasToScreen)
        {
            Directory.CreateDirectory(outputRoot);

            // 입력은 BGR Mat로 가정
            int imgW = src.Width;
            int imgH = src.Height;

            // OCR 서비스 초기화
            using var ocr = new OCRService(new OCRService.Options
            {
                Languages = "kor+eng",
                Psm = Tesseract.PageSegMode.SingleLine,
                EngineMode = Tesseract.EngineMode.LstmOnly,
                TessdataPath = null,
                CharWhitelist = null
            });

            string bestText = string.Empty;
            float bestConf = -1f;
            Cv.Rect? bestRect = null;
            var stepPaths = new List<string>();

            // ROI 기반: 커서 중심 작은 영역에서 다중 이진화(양/음 폴라리티 + Sauvola)
            // 위치를 좌/우/상/하로 소폭 랜덤 시프트하여 총 4회 처리
            var contoursAll = new List<Cv.Rect>();
            var rnd = new Random();
            int maxShiftX = Math.Max(4, imgW / 20);   // 약 5% 폭
            int maxShiftY = Math.Max(2, imgH / 20);   // 약 5% 높이
            for (int i = 0; i < 4; i++)
            {
                int dx = rnd.Next(-maxShiftX, maxShiftX + 1);
                int dy = rnd.Next(-maxShiftY, maxShiftY + 1);
                var (roiContours, roiStepPaths) = FindContoursAroundCursorMulti(src, outputRoot, dx, dy);
                stepPaths.AddRange(roiStepPaths);
                contoursAll.AddRange(roiContours);
            }

            // 단일 패스: 전체 캔버스 해상도에서 컨투어 탐지 및 OCR
            using var annotatedAll = src.Clone();
            var globalContours = FindContours(src);
            contoursAll.AddRange(globalContours);

            foreach (var rect in contoursAll)
            {
                var clip = rect & new Cv.Rect(0, 0, imgW, imgH);
                if (clip.Width <= 10 || clip.Height <= 10) continue;

                using var contourRoi = new Cv.Mat(src, clip);
                var bin = PreprocessForOCRVariants(contourRoi, outputRoot);

                var res1 = ocr.RecognizeMat(bin);

                if (!string.IsNullOrWhiteSpace(res1.Text) && HasValidKoreanOrEnglish(res1.Text) && res1.MeanConfidence > bestConf)
                {
                    bestConf = res1.MeanConfidence;
                    bestText = res1.Text;
                    bestRect = clip;
                }

                bin.Dispose();
            }

            // 컨투어/최고 결과 오버레이 저장
            foreach (var r in contoursAll)
            {
                var c = r & new Cv.Rect(0, 0, annotatedAll.Width, annotatedAll.Height);
                if (c.Width > 0 && c.Height > 0)
                    Cv2.Rectangle(annotatedAll, c, new Cv.Scalar(255, 0, 0), 1);
            }
            if (bestRect.HasValue)
            {
                var c = bestRect.Value & new Cv.Rect(0, 0, annotatedAll.Width, annotatedAll.Height);
                if (c.Width > 0 && c.Height > 0)
                {
                    Cv2.Rectangle(annotatedAll, c, Cv.Scalar.Red, 2);
                    if (!string.IsNullOrWhiteSpace(bestText))
                    {
                        int baseY = Math.Max(10, c.Y - 6);
                        Cv2.PutText(annotatedAll, $"{bestText} ({bestConf:F1}%)",
                            new Cv.Point(c.X, baseY),
                            Cv.HersheyFonts.HersheySimplex, 0.4, Cv.Scalar.Lime, 1, Cv.LineTypes.AntiAlias);
                    }
                }
            }
            var nowAll = DateTime.Now.ToString("yyyyMMdd_HHmmss_fff");
            string contoursPath = Path.Combine(outputRoot, $"contours_{imgW}x{imgH}_{nowAll}.png");
            Cv2.ImWrite(contoursPath, annotatedAll);
            stepPaths.Add(contoursPath);

            // 최종 전체 이미지에 결과 오버레이
            using var finalAnnotated = src.Clone();
            if (bestRect.HasValue)
            {
                Cv2.Rectangle(finalAnnotated, bestRect.Value, Cv.Scalar.Red, 3);
                if (!string.IsNullOrWhiteSpace(bestText))
                {
                    int baseY = Math.Max(20, bestRect.Value.Y - 10);
                    Cv2.PutText(finalAnnotated, $"BEST: {bestText} ({bestConf:F1}%)",
                        new Cv.Point(bestRect.Value.X, baseY),
                        Cv.HersheyFonts.HersheySimplex, 0.8, Cv.Scalar.Lime, 2, Cv.LineTypes.AntiAlias);
                }
            }

            var finalNow = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            string finalPath = Path.Combine(outputRoot, $"final_result_{finalNow}.png");
            Cv2.ImWrite(finalPath, finalAnnotated);

            return new Result
            {
                FinalAnnotatedPath = finalPath,
                BestText = bestText,
                BestConfidence = Math.Max(0, bestConf),
                StepPaths = stepPaths
            };
        }

        private static bool HasValidKoreanOrEnglish(string text)
        {
            if (string.IsNullOrWhiteSpace(text)) return false;
            foreach (var ch in text)
            {
                if ((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z')) return true;
                if (ch >= '\uAC00' && ch <= '\uD7A3') return true; // Hangul syllables
            }
            return false;
        }

        private List<Cv.Rect> FindContours(Cv.Mat src)
        {
            // 기본 전처리: Gray -> Blur
            using var gray = new Cv.Mat();
            Cv2.CvtColor(src, gray, Cv.ColorConversionCodes.BGR2GRAY);

            using var blur = new Cv.Mat();
            Cv2.GaussianBlur(gray, blur, new Cv.Size(3, 3), 0);

            // 자동 이진화
            using var bw = AutoBinarizeForBoxes(blur);

            // 수평 팽창
            int approxCharH = Math.Max(12, gray.Rows / 60);
            int kernelW = Math.Max(approxCharH / 2, 8);
            using (var hKernel = Cv2.GetStructuringElement(Cv.MorphShapes.Rect, new Cv.Size(kernelW, 1)))
                Cv2.Dilate(bw, bw, hKernel, iterations: 1);

            // 외곽 찾기
            Cv.Point[][] contours;
            Cv.HierarchyIndex[] hierarchy;
            Cv2.FindContours(
                bw,
                out contours,
                out hierarchy,
                Cv.RetrievalModes.External,
                Cv.ContourApproximationModes.ApproxSimple
            );

            var rects = new List<Cv.Rect>();
            double imgArea = src.Width * src.Height;

            foreach (var cnt in contours)
            {
                if (cnt.Length < 3) continue;

                var rect = Cv2.BoundingRect(cnt);
                double area = rect.Width * rect.Height;
                if (area < _opt.MinArea) continue;
                if (area > imgArea * _opt.MaxAreaRatio) continue;

                double ar = (double)rect.Width / Math.Max(1, rect.Height);
                if (ar < _opt.MinAspectRatio || ar > _opt.MaxAspectRatio) continue;

                rects.Add(rect);
            }

            // 정렬/라인 병합
            rects.Sort((a, b) =>
            {
                int y = a.Y.CompareTo(b.Y);
                return Math.Abs(a.Y - b.Y) < 10 ? a.X.CompareTo(b.X) : y;
            });

            return MergeRectsByLine(rects, yOverlapThresh: 1.0, xGapFactor: 0.5);
        }

        // 커서 중심 작은 ROI에 대해 다중 이진화(밝은-글자/어두운-글자 + Sauvola) 후 컨투어 생성
        private (List<Cv.Rect> rects, List<string> stepPaths) FindContoursAroundCursorMulti(Cv.Mat src, string outputRoot, int offsetX = 0, int offsetY = 0)
        {
            var stepPaths = new List<string>();

            int imgW = src.Width;
            int imgH = src.Height;

            // 캔버스는 이미 커서 주변을 캡쳐한 이미지이므로 중앙 기반 소형 ROI 사용
            int roiW = Math.Max(80, imgW / 2);
            int roiH = Math.Max(40, imgH / 2);
            int centerX = imgW / 2;
            int centerY = imgH / 2;
            int roiX = Math.Max(0, centerX - roiW / 2 + offsetX);
            int roiY = Math.Max(0, centerY - roiH / 2 + offsetY);
            var roiRect = new Cv.Rect(roiX, roiY, Math.Min(roiW, imgW - roiX), Math.Min(roiH, imgH - roiY));

            using var roi = new Cv.Mat(src, roiRect);

            // 회색조
            using var roiGray = new Cv.Mat();
            Cv2.CvtColor(roi, roiGray, Cv.ColorConversionCodes.BGR2GRAY);

            // 3종 이진화: (1) 밝은-글자(BinaryInv), (2) 어두운-글자(Binary), (3) Sauvola
            using var bwDarkOnLight = new Cv.Mat();   // 어두운 글자(검정) → BinaryInv가 더 유리할 수 있으나 이름 통일
            using var bwLightOnDark = new Cv.Mat();   // 밝은 글자(흰색)
            Cv.Mat bwSauvola = new Cv.Mat();

            int blockSize = Math.Max(15, (Math.Min(roiRect.Width, roiRect.Height) / 15) | 1); // 홀수 보장
            double C = 10;

            // Adaptive (MeanC)
            Cv2.AdaptiveThreshold(
                roiGray, bwLightOnDark, 255,
                Cv.AdaptiveThresholdTypes.MeanC,
                Cv.ThresholdTypes.Binary, blockSize, C
            );
            Cv2.AdaptiveThreshold(
                roiGray, bwDarkOnLight, 255,
                Cv.AdaptiveThresholdTypes.MeanC,
                Cv.ThresholdTypes.BinaryInv, blockSize, C
            );

            // Sauvola 수식 기반 임계값(의사구현)
            bwSauvola = SauvolaThreshold(roiGray, blockSize, k: 0.2, R: 128.0, binaryInv: false);

            string ts = DateTime.Now.ToString("yyyyMMdd_HHmmss_fff");
            var p1 = Path.Combine(outputRoot, $"roi_bin_light_{ts}.png");
            var p2 = Path.Combine(outputRoot, $"roi_bin_dark_{ts}.png");
            var p3 = Path.Combine(outputRoot, $"roi_sauvola_{ts}.png");
            Cv2.ImWrite(p1, bwLightOnDark);
            Cv2.ImWrite(p2, bwDarkOnLight);
            Cv2.ImWrite(p3, bwSauvola);
            stepPaths.AddRange(new[] { p1, p2, p3 });

            // 세 마스크 각각에서 컨투어→BBox
            var all = new List<Cv.Rect>();
            void ExtractRectsFromMask(Cv.Mat mask)
            {
                using var work = mask.Clone();

                // 수평 연결 강화
                int approxCharH = Math.Max(12, work.Rows / 20);
                int kernelW = Math.Max(approxCharH / 2, 6);
                using (var hKernel = Cv2.GetStructuringElement(Cv.MorphShapes.Rect, new Cv.Size(kernelW, 1)))
                    Cv2.Dilate(work, work, hKernel, iterations: 1);

                Cv.Point[][] contours;
                Cv.HierarchyIndex[] hierarchy;
                Cv2.FindContours(
                    work,
                    out contours,
                    out hierarchy,
                    Cv.RetrievalModes.External,
                    Cv.ContourApproximationModes.ApproxSimple
                );

                double imgArea = src.Width * src.Height;
                foreach (var cnt in contours)
                {
                    if (cnt.Length < 3) continue;
                    var r = Cv2.BoundingRect(cnt);
                    // ROI 좌표 → 전체 좌표로 보정
                    r = new Cv.Rect(r.X + roiRect.X, r.Y + roiRect.Y, r.Width, r.Height);

                    double area = r.Width * r.Height;
                    if (area < _opt.MinArea) continue;
                    if (area > imgArea * _opt.MaxAreaRatio) continue;

                    double ar = (double)r.Width / Math.Max(1, r.Height);
                    if (ar < _opt.MinAspectRatio || ar > _opt.MaxAspectRatio) continue;
                    all.Add(r);
                }
            }

            ExtractRectsFromMask(bwLightOnDark);
            ExtractRectsFromMask(bwDarkOnLight);
            ExtractRectsFromMask(bwSauvola);

            // 라인 병합 및 정렬
            var merged = MergeRectsByLine(all, yOverlapThresh: 1.0, xGapFactor: 0.5);
            return (merged, stepPaths);
        }

        // Sauvola 지역 임계값 이진화 (ximgproc 미사용 버전)
        private static Cv.Mat SauvolaThreshold(Cv.Mat gray8u, int blockSize, double k = 0.2, double R = 128.0, bool binaryInv = false)
        {
            if (blockSize % 2 == 0) blockSize += 1;
            blockSize = Math.Max(3, blockSize);

            using var gray32 = new Cv.Mat();
            gray8u.ConvertTo(gray32, Cv.MatType.CV_32F);

            using var mean = new Cv.Mat();
            using var sqmean = new Cv.Mat();

            Cv2.Blur(gray32, mean, new Cv.Size(blockSize, blockSize));
            using var graySq = new Cv.Mat();
            Cv2.Multiply(gray32, gray32, graySq);
            Cv2.Blur(graySq, sqmean, new Cv.Size(blockSize, blockSize));

            using var meanSq = new Cv.Mat();
            Cv2.Multiply(mean, mean, meanSq);
            using var variance = new Cv.Mat();
            Cv2.Subtract(sqmean, meanSq, variance);
            using var zero = new Cv.Mat(variance.Size(), variance.Type(), new Cv.Scalar(0));
            using var varClamped = new Cv.Mat();
            Cv2.Max(variance, zero, varClamped);
            using var stddev = new Cv.Mat();
            Cv2.Sqrt(varClamped, stddev);

            using var Rmat = new Cv.Mat(stddev.Size(), stddev.Type(), new Cv.Scalar(R));
            using var sOverR = new Cv.Mat();
            Cv2.Divide(stddev, Rmat, sOverR);

            using var kMat = new Cv.Mat(sOverR.Size(), sOverR.Type(), new Cv.Scalar(k));
            using var one = new Cv.Mat(sOverR.Size(), sOverR.Type(), new Cv.Scalar(1));
            using var inner = new Cv.Mat();
            Cv2.Subtract(sOverR, one, inner);
            using var kInner = new Cv.Mat();
            Cv2.Multiply(kMat, inner, kInner);
            using var onePlus = new Cv.Mat();
            Cv2.Add(one, kInner, onePlus);

            using var thresh = new Cv.Mat();
            Cv2.Multiply(mean, onePlus, thresh);

            // gray32 - thresh > 0 => 255 else 0
            using var diff = new Cv.Mat();
            Cv2.Subtract(gray32, thresh, diff);
            using var mask = new Cv.Mat();
            Cv2.Threshold(diff, mask, 0, 255, Cv.ThresholdTypes.Binary);
            if (binaryInv)
            {
                using var maskInv = new Cv.Mat();
                Cv2.BitwiseNot(mask, maskInv);
                using var maskInv8u = new Cv.Mat();
                maskInv.ConvertTo(maskInv8u, Cv.MatType.CV_8U);
                return maskInv8u.Clone();
            }
            else
            {
                using var mask8u = new Cv.Mat();
                mask.ConvertTo(mask8u, Cv.MatType.CV_8U);
                return mask8u.Clone();
            }
        }

        private Cv.Mat PreprocessForOCRVariants(Cv.Mat roiBgr, string? outputPath = null)
        {
            // 0) 업스케일
            var up = new Cv.Mat();
            double scale = Math.Max(1.0, _opt.Upscale * (_opt.BoostThinText ? _opt.BoostScale : 1));
            Cv2.Resize(roiBgr, up, new Cv.Size(), scale, scale, Cv.InterpolationFlags.Lanczos4);

            string ts = DateTime.Now.ToString("yyyyMMdd_HHmmss_fff");
            if (!string.IsNullOrEmpty(outputPath)) Cv2.ImWrite(Path.Combine(outputPath, $"preprocess_1_upscaled_{ts}.png"), up);

            // 1) Lab → L 채널 (8U 보장)
            using var lab = new Cv.Mat();
            Cv2.CvtColor(up, lab, Cv.ColorConversionCodes.BGR2Lab);
            Cv2.Split(lab, out Cv.Mat[] labCh);
            using var L = labCh[0]; // 0~255
            if (!string.IsNullOrEmpty(outputPath)) Cv2.ImWrite(Path.Combine(outputPath, $"preprocess_2_L_channel_{ts}.png"), L);

            // 2) 대비 증폭: L + a*TopHat - b*BlackHat
            int ks = Math.Max(3, (int)Math.Round(Math.Min(L.Rows, L.Cols) / 100.0) | 1); // 홀수
            using var k = Cv2.GetStructuringElement(Cv.MorphShapes.Rect, new Cv.Size(ks, ks));
            using var topHat = new Cv.Mat();
            using var blackHat = new Cv.Mat();
            Cv2.MorphologyEx(L, topHat, Cv.MorphTypes.TopHat, k);
            Cv2.MorphologyEx(L, blackHat, Cv.MorphTypes.BlackHat, k);

            double a = 1.0, b = 1.0; // 필요시 옵션으로 노출
            using var tmp = new Cv.Mat();
            Cv2.AddWeighted(L, 1.0, topHat, a, 0, tmp);          // tmp = L + a*TopHat
            using var enhanced = new Cv.Mat();
            Cv2.Subtract(tmp, blackHat * b, enhanced);           // enhanced = tmp - b*BlackHat
                                                                 // (OpenCvSharp의 연산자 오버로드 사용 가능)

            // 2.5) CLAHE로 지역 대비 추가 개선
            using var clahe = Cv.CLAHE.Create(clipLimit: 2.0, tileGridSize: new Cv.Size(8, 8));
            using var enhancedClahe = new Cv.Mat();
            clahe.Apply(enhanced, enhancedClahe);

            if (!string.IsNullOrEmpty(outputPath))
            {
                Cv2.ImWrite(Path.Combine(outputPath, $"preprocess_3a_enhanced_{ts}.png"), enhanced);
                Cv2.ImWrite(Path.Combine(outputPath, $"preprocess_3b_enhanced_clahe_{ts}.png"), enhancedClahe);
            }

            // 3) 디노이즈
            Cv.Mat denoised = new();
            if (_opt.ApplyDenoise)
                Cv2.GaussianBlur(enhancedClahe, denoised, new Cv.Size(3, 3), 0);
            else
                denoised = enhancedClahe.Clone();

            if (!string.IsNullOrEmpty(outputPath))
                Cv2.ImWrite(Path.Combine(outputPath, $"preprocess_4_denoised_{ts}.png"), denoised);

            var bin = denoised.Clone();
            
            // 5) 얇은 글자 보강(선택)
            if (_opt.BoostThinText)
            {
                int kThin = Math.Max(1, Math.Min(bin.Rows, bin.Cols) / 400);
                using var hKernel = Cv2.GetStructuringElement(Cv.MorphShapes.Rect, new Cv.Size(1 + 2 * kThin, 1));
                Cv2.Dilate(bin, bin, hKernel, iterations: 1);

                using var small = Cv2.GetStructuringElement(Cv.MorphShapes.Rect, new Cv.Size(2, 2));
                Cv2.MorphologyEx(bin, bin, Cv.MorphTypes.Open, small, iterations: 1);
            }

            if (!string.IsNullOrEmpty(outputPath))
                Cv2.ImWrite(Path.Combine(outputPath, $"preprocess_5_binary_{ts}.png"), bin);

            // 자원 정리
            up.Dispose();
            denoised.Dispose();
            labCh[1]?.Dispose();
            labCh[2]?.Dispose();

            return bin;
        }


        // 자동 이진화(컨투어용)
        private static Cv.Mat AutoBinarizeForBoxes(Cv.Mat grayOrBlur, bool forceOpposite = false)
        {
            var bin = new Cv.Mat();
            var binInv = new Cv.Mat();

            Cv2.AdaptiveThreshold(
                grayOrBlur, bin, 255,
                Cv.AdaptiveThresholdTypes.MeanC,
                Cv.ThresholdTypes.Binary, 21, 10
            );

            Cv2.AdaptiveThreshold(
                grayOrBlur, binInv, 255,
                Cv.AdaptiveThresholdTypes.MeanC,
                Cv.ThresholdTypes.BinaryInv, 21, 10
            );

            if (forceOpposite)
            {
                double sBin = ScoreMaskForText(bin);
                double sInv = ScoreMaskForText(binInv);
                return (sBin >= sInv) ? binInv : bin;
            }
            else
            {
                double sBin = ScoreMaskForText(bin);
                double sInv = ScoreMaskForText(binInv);
                return (sInv > sBin) ? binInv : bin;
            }
        }

        // 텍스트스코어: 글자같은 CC 개수/균형
        private static double ScoreMaskForText(Cv.Mat bw)
        {
            var mean = Cv2.Mean(bw).Val0;
            double balance = 255.0 - Math.Abs(127.5 - mean) * 2.0;
            balance = Math.Max(0, balance);

            using var tmp = bw.Clone();
            using (var k = Cv2.GetStructuringElement(Cv.MorphShapes.Rect, new Cv.Size(2, 1)))
                Cv2.Dilate(tmp, tmp, k, iterations: 1);

            Cv.Point[][] cnts;
            Cv.HierarchyIndex[] hier;
            Cv2.FindContours(tmp, out cnts, out hier,
                Cv.RetrievalModes.External, Cv.ContourApproximationModes.ApproxSimple);

            int good = 0;
            foreach (var c in cnts)
            {
                if (c.Length < 3) continue;
                var r = Cv2.BoundingRect(c);
                if (r.Width < 2 || r.Height < 2) continue;
                double ar = (double)r.Width / Math.Max(1, r.Height);
                if (ar < 0.15 || ar > 15) continue;
                good++;
            }

            return good * 10.0 + balance;
        }

        // 라인 병합
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
                var refBox = line[0];
                int yTop = Math.Max(refBox.Y, r.Y);
                int yBot = Math.Min(refBox.Y + refBox.Height, r.Y + r.Height);
                int overlap = Math.Max(0, yBot - yTop);
                double overlapRatio = (double)overlap / Math.Min(refBox.Height, r.Height);

                if (overlapRatio >= yOverlapThresh) line.Add(r);
                else { FlushLine(); line.Add(r); }
            }
            FlushLine();
            return merged;
        }
    }
}