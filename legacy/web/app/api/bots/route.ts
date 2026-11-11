import { NextResponse } from "next/server";

import { registeredBots } from "@/lib/bots";
import { isMisconfigured, isPersistent } from "@/lib/store";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/** Opponents this deployment can seat, plus a warning when storage is ephemeral. */
export async function GET() {
  return NextResponse.json({
    bots: registeredBots().map((b) => ({
      id: b.id,
      name: b.name,
      description: b.description ?? "",
      builtin: !!b.builtin,
    })),
    persistent: isPersistent,
    misconfigured: isMisconfigured,
  });
}
