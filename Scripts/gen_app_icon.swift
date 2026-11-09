// gen_app_icon.swift — Renders Qwegn AppIcon.iconset PNGs using CoreGraphics.
// Draws the dark macOS squircle tile and the canonical 600x600 QG microchip mark.

import AppKit
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

func drawQGMark(in ctx: CGContext, size: CGFloat) {
    ctx.saveGState()
    // Flip to top-left origin inside padded inner frame
    ctx.translateBy(x: 0, y: size)
    ctx.scaleBy(x: 1, y: -1)
    let pad = size * 0.11
    let inner = size - 2 * pad
    ctx.translateBy(x: pad, y: pad)
    let s = inner / 600.0
    ctx.scaleBy(x: s, y: s)

    ctx.setStrokeColor(CGColor(srgbRed: 0.91, green: 0.95, blue: 0.99, alpha: 1.0))
    ctx.setLineWidth(24)
    ctx.setLineCap(.round)
    ctx.setLineJoin(.round)

    for x: CGFloat in [220, 260, 300, 340, 380] {
        ctx.move(to: CGPoint(x: x, y: 62)); ctx.addLine(to: CGPoint(x: x, y: 108))
        ctx.move(to: CGPoint(x: x, y: 492)); ctx.addLine(to: CGPoint(x: x, y: 538))
    }
    ctx.move(to: CGPoint(x: 62, y: 220)); ctx.addLine(to: CGPoint(x: 108, y: 220))
    ctx.move(to: CGPoint(x: 62, y: 260)); ctx.addLine(to: CGPoint(x: 108, y: 260))
    ctx.move(to: CGPoint(x: 62, y: 300)); ctx.addLine(to: CGPoint(x: 108, y: 300))
    ctx.move(to: CGPoint(x: 62, y: 340)); ctx.addLine(to: CGPoint(x: 340, y: 340))
    ctx.move(to: CGPoint(x: 62, y: 380)); ctx.addLine(to: CGPoint(x: 108, y: 380))
    for y: CGFloat in [220, 260, 300, 340, 380] {
        ctx.move(to: CGPoint(x: 492, y: y)); ctx.addLine(to: CGPoint(x: 538, y: y))
    }
    ctx.strokePath()

    ctx.move(to: CGPoint(x: 276, y: 135))
    ctx.addLine(to: CGPoint(x: 180, y: 135))
    ctx.addQuadCurve(to: CGPoint(x: 135, y: 180), control: CGPoint(x: 135, y: 135))
    ctx.addLine(to: CGPoint(x: 135, y: 276))

    ctx.move(to: CGPoint(x: 324, y: 135))
    ctx.addLine(to: CGPoint(x: 420, y: 135))
    ctx.addQuadCurve(to: CGPoint(x: 465, y: 180), control: CGPoint(x: 465, y: 135))
    ctx.addLine(to: CGPoint(x: 465, y: 276))

    ctx.move(to: CGPoint(x: 135, y: 324))
    ctx.addLine(to: CGPoint(x: 135, y: 420))
    ctx.addQuadCurve(to: CGPoint(x: 180, y: 465), control: CGPoint(x: 135, y: 465))
    ctx.addLine(to: CGPoint(x: 276, y: 465))

    ctx.move(to: CGPoint(x: 465, y: 324))
    ctx.addLine(to: CGPoint(x: 465, y: 420))
    ctx.addQuadCurve(to: CGPoint(x: 420, y: 465), control: CGPoint(x: 465, y: 465))
    ctx.addLine(to: CGPoint(x: 324, y: 465))
    ctx.strokePath()

    ctx.setStrokeColor(CGColor(srgbRed: 0.38, green: 0.86, blue: 0.95, alpha: 1.0))
    let qRect = CGRect(x: 180, y: 175, width: 112, height: 250)
    ctx.addPath(CGPath(roundedRect: qRect, cornerWidth: 36, cornerHeight: 36, transform: nil))
    ctx.move(to: CGPoint(x: 246, y: 336))
    ctx.addLine(to: CGPoint(x: 346, y: 436))
    ctx.strokePath()

    ctx.move(to: CGPoint(x: 324, y: 175))
    ctx.addLine(to: CGPoint(x: 380, y: 175))
    ctx.addQuadCurve(to: CGPoint(x: 412, y: 207), control: CGPoint(x: 412, y: 175))
    ctx.addLine(to: CGPoint(x: 412, y: 268))
    ctx.move(to: CGPoint(x: 324, y: 326))
    ctx.addLine(to: CGPoint(x: 412, y: 326))
    ctx.addLine(to: CGPoint(x: 412, y: 393))
    ctx.addQuadCurve(to: CGPoint(x: 380, y: 425), control: CGPoint(x: 412, y: 425))
    ctx.addLine(to: CGPoint(x: 324, y: 425))
    ctx.strokePath()

    ctx.restoreGState()
}

func renderIcon(pixels: Int, to url: URL) throws {
    let cs = CGColorSpace(name: CGColorSpace.sRGB)!
    guard let ctx = CGContext(
        data: nil, width: pixels, height: pixels,
        bitsPerComponent: 8, bytesPerRow: pixels * 4, space: cs,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else {
        throw NSError(domain: "gen_app_icon", code: 1)
    }
    let s = CGFloat(pixels)
    let inset = s * 0.08
    let tile = CGRect(x: inset, y: inset, width: s - 2 * inset, height: s - 2 * inset)
    let radius = tile.width * 0.225
    let path = CGPath(roundedRect: tile, cornerWidth: radius, cornerHeight: radius, transform: nil)

    ctx.saveGState()
    ctx.addPath(path)
    ctx.clip()
    let colors = [
        CGColor(srgbRed: 0.11, green: 0.14, blue: 0.20, alpha: 1.0),
        CGColor(srgbRed: 0.05, green: 0.07, blue: 0.10, alpha: 1.0),
    ] as CFArray
    let grad = CGGradient(colorsSpace: cs, colors: colors, locations: [0.0, 1.0])!
    ctx.drawLinearGradient(grad, start: CGPoint(x: s * 0.5, y: s), end: CGPoint(x: s * 0.5, y: 0), options: [])
    ctx.restoreGState()

    ctx.addPath(path)
    ctx.setStrokeColor(CGColor(srgbRed: 0.26, green: 0.34, blue: 0.46, alpha: 0.85))
    ctx.setLineWidth(max(1.0, s * 0.012))
    ctx.strokePath()

    drawQGMark(in: ctx, size: s)

    guard let img = ctx.makeImage(),
          let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
        throw NSError(domain: "gen_app_icon", code: 2)
    }
    CGImageDestinationAddImage(dest, img, nil)
    guard CGImageDestinationFinalize(dest) else {
        throw NSError(domain: "gen_app_icon", code: 3)
    }
}

guard CommandLine.arguments.count == 2 else {
    fputs("Usage: swift Scripts/gen_app_icon.swift <AppIcon.iconset>\n", stderr)
    exit(2)
}
let outDir = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
let specs: [(String, Int)] = [
    ("icon_16x16.png", 16), ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32), ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128), ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256), ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512), ("icon_512x512@2x.png", 1024),
]
for (name, px) in specs {
    try renderIcon(pixels: px, to: outDir.appendingPathComponent(name))
}
