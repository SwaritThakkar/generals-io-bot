import { NextRequest, NextResponse } from "next/server";

import { BotRef, registeredBots } from "@/lib/bots";
import { createMatch, publicView } from "@/lib/match";
import { DEFAULT_CONFIG, GameConfig } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function clamp(v: unknown, lo: number, hi: number, fallback: number): number {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? Math.min(hi, Math.max(lo, Math.round(n))) : fallback;
}

export async function POST(req: NextRequest) {
  const body = await req.json().catch(() => ({}));
  const mode = body.mode === "bot" ? "bot" : "friend";

  const size = clamp(body.size, 5, 32, DEFAULT_CONFIG.height);
  const cfg: Partial<GameConfig> = {
    height: size,
    width: size,
    tickMs: clamp(body.tickMs, 100, 5000, DEFAULT_CONFIG.tickMs),
    maxTicks: clamp(body.maxTicks, 50, 20000, DEFAULT_CONFIG.maxTicks),
    minSpawnDist: clamp(body.minSpawnDist, 2, 60, Math.min(DEFAULT_CONFIG.minSpawnDist, size - 6)),
  };
  if (typeof body.maxSpawnDist === "number") {
    cfg.maxSpawnDist = clamp(body.maxSpawnDist, cfg.minSpawnDist!, 1e9, 1e9);
  }

  let bot: BotRef | null = null;
  if (mode === "bot") {
    if (typeof body.botUrl === "string" && body.botUrl.trim()) {
      // Ad-hoc endpoint, so a laptop behind a tunnel can play without a redeploy.
      bot = { id: "custom", name: body.botName?.trim() || "Custom endpoint", url: body.botUrl.trim() };
    } else {
      bot = registeredBots().find((b) => b.id === body.botId) ?? null;
      if (!bot) {
        return NextResponse.json(
          {
            error: `unknown opponent ${JSON.stringify(body.botId ?? null)}. ` +
              "Pick one of the built-ins, register yours in GENMOD_BOTS, or pass botUrl.",
          },
          { status: 400 },
        );
      }
    }
  }

  try {
    const { match, token } = await createMatch({
      mode,
      cfg,
      hostName: typeof body.name === "string" ? body.name : undefined,
      bot,
      seed: typeof body.seed === "number" ? body.seed >>> 0 : undefined,
    });
    return NextResponse.json({ token, match: publicView(match, token) });
  } catch (e) {
    return NextResponse.json({ error: String(e instanceof Error ? e.message : e) }, { status: 400 });
  }
}
