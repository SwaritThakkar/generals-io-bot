import { NextRequest, NextResponse } from "next/server";

import { joinMatch, publicView } from "@/lib/match";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: NextRequest, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;
  const body = await req.json().catch(() => ({}));
  const result = await joinMatch(id, String(body.name ?? ""), body.token ?? null);
  if ("error" in result) return NextResponse.json(result, { status: 409 });
  return NextResponse.json({
    token: result.token,
    player: result.player,
    match: publicView(result.match, result.token),
  });
}
