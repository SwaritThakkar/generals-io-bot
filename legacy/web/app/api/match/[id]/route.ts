import { NextRequest, NextResponse } from "next/server";

import { publicView, tickMatch } from "@/lib/match";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/** Current state of a match. Advancing the clock is a side effect of asking. */
export async function GET(req: NextRequest, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;
  const match = await tickMatch(id);
  if (!match) return NextResponse.json({ error: "no such match" }, { status: 404 });
  const token = req.nextUrl.searchParams.get("token");
  return NextResponse.json(publicView(match, token));
}
