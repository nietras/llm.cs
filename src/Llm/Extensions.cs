using System;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace nietras.LargeLanguageModel;

internal static class Extensions
{
    public static unsafe void ReadExactlyUnmanaged<T>(this FileStream file, Span<T> values)
        where T : unmanaged
    {
        fixed (T* ptr = values)
        {
            ReadExactlyUnmanaged(file, ptr, values.Length);
        }
    }

    public static unsafe void ReadExactlyUnmanaged<T>(this FileStream file, T* values, nint count)
        where T : unmanaged
    {
        Span<T> buffer = stackalloc T[(256 * 1024) / Unsafe.SizeOf<T>()];
        var totalReadCount = 0;
        while (totalReadCount < count)
        {
            var countToRead = (int)Math.Min(buffer.Length, count - totalReadCount);
            var bufferToRead = buffer.Slice(0, countToRead);
            var span = MemoryMarshal.Cast<T, byte>(bufferToRead);
            file.ReadExactly(span);
            bufferToRead.CopyTo(new Span<T>(values + totalReadCount, countToRead));
            totalReadCount += countToRead;
        }
    }

    public static nint Product(this nint[] values) => Product(values.AsSpan());

    public static nint Product(this ReadOnlySpan<nint> values)
    {
        if (values.Length == 0) { return 0; }
        var product = values[0];
        for (var i = 1; i < values.Length; i++)
        {
            product *= values[i];
        }
        return product;
    }

    public static string ToShapeText(this ReadOnlySpan<nint> values)
    {
        Span<char> buffer = stackalloc char[1024];
        var handler = new DefaultInterpolatedStringHandler(values.Length - 1 + 2, values.Length, null, buffer);
        handler.AppendLiteral("[");
        for (var i = 0; i < values.Length; i++)
        {
            if (i > 0) { handler.AppendLiteral(", "); }
            handler.AppendFormatted(values[i], "D");
        }
        handler.AppendLiteral("]");
        return handler.ToString();
    }
}
