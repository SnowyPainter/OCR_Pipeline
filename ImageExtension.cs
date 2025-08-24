using System.Drawing;
using System.Drawing.Imaging;
using System.IO;

public static class ImageExtension
{
    /// <summary>
    /// Converts a Bitmap object to a byte array in a specified image format.
    /// </summary>
    /// <param name="bitmap">The Bitmap object to convert.</param>
    /// <param name="format">The ImageFormat to save the bitmap as (e.g., ImageFormat.Bmp, ImageFormat.Png).</param>
    /// <returns>A byte array representing the bitmap in the specified format.</returns>
    public static byte[] ConvertBitmapToByteArray(Bitmap bitmap, ImageFormat format)
    {
        if (bitmap == null)
        {
            throw new ArgumentNullException(nameof(bitmap), "Bitmap cannot be null.");
        }

        using (MemoryStream ms = new MemoryStream())
        {
            // Save the bitmap to the MemoryStream in the specified format
            bitmap.Save(ms, format);

            // Convert the MemoryStream to a byte array
            return ms.ToArray();
        }
    }
}