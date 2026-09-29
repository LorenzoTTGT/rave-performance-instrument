"""Make JUCE 7.0.9's native-window snapshot compile with the macOS 15 SDK.

JUCE 7 uses CoreGraphics' removed CGWindowListCreateImage call. Keep its older
SDK implementation and use AppKit's view cache with the macOS 15 SDK. This patch
only changes the fetched JUCE checkout; the pinned JUCE version and GPL terms
stay the same.
"""

import argparse
from pathlib import Path


START = "        // CGWindowListCreateImage is replaced by functions in the ScreenCaptureKit framework, but"
END = "        Image result (Image::ARGB, (int) [bitmapRep size].width, (int) [bitmapRep size].height, true);"
MARKER = (
    "        // RAVE: AppKit snapshot for SDKs that removed CGWindowListCreateImage."
)
RELEASE = "        CGImageRelease (screenShot);"


def patch(source):
    path = Path(source) / "modules/juce_gui_basics/native/juce_Windowing_mac.mm"
    original = path.read_bytes()
    newline = b"\r\n" if b"\r\n" in original else b"\n"
    content = original.replace(b"\r\n", b"\n").decode("utf-8")
    if MARKER in content:
        return False
    if content.count(START) != 1 or content.count(RELEASE) != 1:
        raise ValueError("Unsupported JUCE native-window snapshot source")
    start = content.index(START)
    end = content.index(END, start)
    old = content[start:end]
    if (
        old.count("CGWindowListCreateImage (") != 1
        or old.count("NSBitmapImageRep* bitmapRep") != 1
    ):
        raise ValueError("Unsupported JUCE native-window snapshot implementation")
    replacement = (
        """       #if defined (MAC_OS_VERSION_15_0) && MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_VERSION_15_0
"""
        + MARKER
        + """
        NSView* view = [nsWindow contentView];
        if (view == nil)
            return {};
        NSBitmapImageRep* bitmapRep = [[view bitmapImageRepForCachingDisplayInRect: [view bounds]] retain];
        if (bitmapRep == nil)
            return {};
        [view cacheDisplayInRect: [view bounds] toBitmapImageRep: bitmapRep];
       #else
"""
        + old
        + """       #endif

"""
    )
    content = content[:start] + replacement + content[end:]
    content = content.replace(
        RELEASE,
        """       #if ! (defined (MAC_OS_VERSION_15_0) && MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_VERSION_15_0)
"""
        + RELEASE
        + """
       #endif""",
    )
    path.write_bytes(content.encode("utf-8").replace(b"\n", newline))
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("juce_source", type=Path)
    args = parser.parse_args()
    print(
        "JUCE macOS 15 snapshot patch applied"
        if patch(args.juce_source)
        else "JUCE macOS 15 patch already applied"
    )
