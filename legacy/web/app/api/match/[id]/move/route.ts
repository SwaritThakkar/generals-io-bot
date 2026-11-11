import { NextRequest, NextResponse } from "next/server";

import { appendMoves, clearQueue, publicView } from "@/lib/match";
import { Move } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * Queue moves, or clear the queue.
 *
 * The browser sends each keypress with a monotonic `seq`, so a retry after a
 * dropped connection re-sends safely: the server ignores anything it has already
 * applied. The queue lives on the server, which means network hiccups delay your
 * moves but never lose them.
 */
export async function POST(req: NextRequest, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;
  const body = await req.json().catch(() => ({}));
  const token = String(body.token ?? "");
  if (!token) return NextResponse.json({ error: "missing token" }, { status: 401 });

  if (body.clear) {
    const result = await clearQueue(id, token);
    if ("error" in result) return NextResponse.json(result, { status: 400 });
    return NextResponse.json(publicView(result, token));
  }

  const raw = Array.isArray(body.moves) ? body.moves : [];
  const moves = raw
    .filter((m: unknown): m is Move & { seq: number } => {
      const o = m as Record<string, unknown>;
      return (
        o !== null &&
        typeof o === "object" &&
        Number.isInteger(o.from) &&
        Number.isInteger(o.dir) &&
        (o.dir as number) >= 0 &&
        (o.dir as number) < 4 &&
        Number.isInteger(o.seq)
      );
    })
    .slice(0, 64);

  const result = await appendMoves(id, token, moves);
  if ("error" in result) return NextResponse.json(result, { status: 400 });
  return NextResponse.json(publicView(result, token));
}
