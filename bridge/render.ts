/**
 * Bun bridge for hermes-snapcompact: reads a JSON request on stdin,
 * renders text into snapcompact PNG frames, writes JSON response to stdout.
 *
 * Actions:
 *   render   — render text into base64 PNG image blocks
 *   geometry — return frame geometry and shape info without rendering
 *   frames   — count frames needed without rendering
 */

import {
  renderMany,
  frames,
  geometry,
  normalize,
  resolveShape,
  resolveShapeForText,
  type ShapeTarget,
  type ShapeVariantName,
} from "@oh-my-pi/snapcompact";

interface RenderRequest {
  action: "render";
  text: string;
  model?: ShapeTarget;
  variant?: ShapeVariantName | "auto";
  maxFrames?: number;
}

interface GeometryRequest {
  action: "geometry";
  text?: string;
  model?: ShapeTarget;
  variant?: ShapeVariantName | "auto";
}

interface FramesRequest {
  action: "frames";
  text: string;
  model?: ShapeTarget;
  variant?: ShapeVariantName | "auto";
}

type Request = RenderRequest | GeometryRequest | FramesRequest;

async function main() {
  const input = await Bun.stdin.text();
  let request: Request;
  try {
    request = JSON.parse(input);
  } catch {
    process.stdout.write(JSON.stringify({ error: "Invalid JSON input" }));
    process.exit(1);
  }

  try {
    switch (request.action) {
      case "render": {
        const shape = request.text
          ? resolveShapeForText(request.text, request.model, request.variant)
          : resolveShape(request.model, request.variant);
        const geo = geometry(shape);
        const images = await renderMany(request.text, {
          shape,
          maxFrames: request.maxFrames,
        });
        process.stdout.write(
          JSON.stringify({
            images,
            shape: {
              font: shape.font,
              cellWidth: shape.cellWidth,
              cellHeight: shape.cellHeight,
              variant: shape.variant,
              lineRepeat: shape.lineRepeat,
              frameSize: shape.frameSize,
              frameTokenEstimate: shape.frameTokenEstimate,
              columns: shape.columns,
              stopwordDim: shape.stopwordDim,
              imageDetail: shape.imageDetail,
            },
            geometry: geo,
            frameCount: images.length,
          })
        );
        break;
      }

      case "geometry": {
        const shape = request.text
          ? resolveShapeForText(request.text, request.model, request.variant)
          : resolveShape(request.model, request.variant);
        const geo = geometry(shape);
        process.stdout.write(
          JSON.stringify({
            shape: {
              font: shape.font,
              cellWidth: shape.cellWidth,
              cellHeight: shape.cellHeight,
              variant: shape.variant,
              lineRepeat: shape.lineRepeat,
              frameSize: shape.frameSize,
              frameTokenEstimate: shape.frameTokenEstimate,
              columns: shape.columns,
              imageDetail: shape.imageDetail,
            },
            geometry: geo,
          })
        );
        break;
      }

      case "frames": {
        const count = frames(request.text, { model: request.model });
        process.stdout.write(JSON.stringify({ frames: count }));
        break;
      }

      default:
        process.stdout.write(
          JSON.stringify({ error: `Unknown action: ${(request as unknown as Record<string, unknown>).action}` })
        );
        process.exit(1);
    }
  } catch (err: any) {
    process.stdout.write(
      JSON.stringify({ error: err?.message ?? String(err) })
    );
    process.exit(1);
  }
}

main();
