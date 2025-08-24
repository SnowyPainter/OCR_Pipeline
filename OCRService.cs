using System;
using System.Text.RegularExpressions;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Tesseract;

namespace OCRPipeline
{
    public sealed class OCRService : IDisposable
    {
        public sealed class Options
        {
            /// <summary>언어 코드. 예: "kor+eng"</summary>
            public string Languages { get; init; } = "kor+eng";

            /// <summary>tessdata 위치. null이면 실행파일 경로의 ./tessdata</summary>
            public string? TessdataPath { get; init; } = null;

            /// <summary>페이지 세그먼트 모드. 기본은 Auto(PSM.Auto)</summary>
            public PageSegMode Psm { get; init; } = PageSegMode.Auto;

            /// <summary>문자 화이트리스트(선택). 예: "0123456789-:()[]"</summary>
            public string? CharWhitelist { get; init; } = null;

            /// <summary>LSTM 전용 엔진 사용 여부 (true 권장)</summary>
            public EngineMode EngineMode { get; init; } = EngineMode.LstmOnly;

            /// <summary>이미지 로딩 실패/빈 텍스트도 결과에 포함할지</summary>
            public bool IncludeEmpty { get; init; } = false;
        }

        public sealed class ItemResult
        {
            public string Path { get; init; } = "";
            public string Text { get; init; } = "";
            public float MeanConfidence { get; init; } = 0f;
            public bool Success { get; init; } = false;
            public string? Error { get; init; }
        }

        private readonly Options _opt;
        private readonly TesseractEngine _engine;

        public OCRService(Options? options = null)
        {
            _opt = options ?? new Options();

            var baseDir = AppContext.BaseDirectory;
            var tessPath = _opt.TessdataPath ?? System.IO.Path.Combine(baseDir, "tessdata");

            if (!Directory.Exists(tessPath))
            {
                throw new DirectoryNotFoundException($"tessdata not found: {tessPath}");
            }

            // 환경변수로도 힌트 제공(선택)
            Environment.SetEnvironmentVariable("TESSDATA_PREFIX", tessPath);

            _engine = new TesseractEngine(tessPath, _opt.Languages, _opt.EngineMode);
            _engine.SetVariable("user_defined_dpi", "300"); // DPI 힌트(저해상도 스샷 개선)
            if (!string.IsNullOrEmpty(_opt.CharWhitelist))
                _engine.SetVariable("tessedit_char_whitelist", _opt.CharWhitelist);

            Console.WriteLine($"OCRService initialized (lang={_opt.Languages}, tessdata={tessPath})");
        }

        /// <summary>
        /// 전처리된 이미지 경로 배열을 받아 순차 OCR 수행.
        /// </summary>
        public ItemResult RecognizeAsync(
            string preprocessedPath,
            CancellationToken ct = default)
        {
            if (!File.Exists(preprocessedPath))
                throw new FileNotFoundException("Preprocessed image not found", preprocessedPath);
            using var bmp = new Bitmap(preprocessedPath);
            using var pix = Pix.LoadFromFile(preprocessedPath);
            using var page = _engine.Process(pix, _opt.Psm);
            string text = page.GetText()?.Trim() ?? "";
            float conf = page.GetMeanConfidence() * 100.0f;
            var result = new ItemResult
            {
                Path = preprocessedPath,
                Text = text,
                MeanConfidence = conf,
                Success = true
            };

            return result;
        }

        /// <summary>
        /// 여러 결과 텍스트를 합쳐 하나로 반환(파일 경계는 --- 구분선)
        /// </summary>
        public static string ConcatenateTexts(IEnumerable<ItemResult> items)
        {
            var sw = new System.Text.StringBuilder();
            foreach (var it in items)
            {
                if (!it.Success || string.IsNullOrWhiteSpace(it.Text) || it.MeanConfidence < 70) continue;
                sw.AppendLine(it.Text);
            }
            return sw.ToString().Trim();
        }

        public static string TrimConcatenatedText(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return string.Empty;

            string cleaned = Regex.Replace(text, @"\s+", "");
            cleaned = Regex.Replace(cleaned, @"[^가-힣a-zA-Z0-9]", "");

            return cleaned;
        }

        public void Dispose()
        {
            _engine?.Dispose();
        }
    }
}
