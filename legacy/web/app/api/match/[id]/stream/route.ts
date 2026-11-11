import { NextRequest } from "next/server";

import { PublicMatch, publicView, tickMatch } from "@/lib/match";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

/** How often the server re-checks the clock. Two samples per default tick. */
const POLL_MS = Number(process.env.GENMOD_POLL_MS ?? 150);
/** Close well before the platform's function ceiling; EventSource reconnects itself. */
const LIFETIME_MS = Number(process.env.GENMOD_STREAM_MS ?? 45_000);

/**
 * Server-sent events carrying the authoritative board.
 *
 * Vercel functions are short-lived, so this stream deliberately ends after
 * `LIFETIME_MS` and lets the browser's `EventSource` reconnect. Clients treat
 * every message as a full snapshot, so a reconnect needs no catch-up logic.
 */
export async function GET(req: NextRequest, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;
  const token = req.nextUrl.searchParams.get("token");

  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      let closed = false;
      const send = (event: string, data: unknown) => {
        if (closed) return;
        try {
          controller.enqueue(encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`));
        } catch {
          closed = true;
        }
      };

      const stop = () => {
        if (closed) return;
        closed = true;
        try {
          controller.close();
        } catch {
          /* already closed */
        }
      };
      req.signal.addEventListener("abort", stop);

      const deadline = Date.now() + LIFETIME_MS;
      let lastTick = -1;
      let lastStatus = "";

      while (!closed && Date.now() < deadline) {
        let view: PublicMatch | null = null;
        try {
          const match = await tickMatch(id);
          view = match ? publicView(match, token) : null;
        } catch (e) {
          send("warn", { message: String(e instanceof Error ? e.message : e) });
        }

        if (!view) {
          send("gone", { id });
          break;
        }
        if (view.snapshot.tick !== lastTick || view.status !== lastStatus) {
          lastTick = view.snapshot.tick;
          lastStatus = view.status;
          send("state", view);
        }
        if (view.status === "finished") break;
        await new Promise((r) => setTimeout(r, POLL_MS));
      }

      send("bye", { reconnect: true });
      stop();
    },
  });

  return new Response(stream, {
    headers: {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache, no-transform",
      connection: "keep-alive",
      "x-accel-buffering": "no",
    },
  });
}
