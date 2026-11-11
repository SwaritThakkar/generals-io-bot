/** Compact board <-> JSON conversion, shared by the API routes and the client. */

import { Board, GameConfig, Move } from "./types";
import { score } from "./engine";

const b64 =
  typeof Buffer !== "undefined"
    ? {
        enc: (u: Uint8Array) => Buffer.from(u).toString("base64"),
        dec: (s: string) => new Uint8Array(Buffer.from(s, "base64")),
      }
    : {
        enc: (u: Uint8Array) => btoa(String.fromCharCode(...u)),
        dec: (s: string) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0)),
      };

const bytes = (a: ArrayBufferView) =>
  new Uint8Array(a.buffer, a.byteOffset, a.byteLength);

export interface WireBoard {
  h: number;
  w: number;
  n: number;
  tick: number;
  done: boolean;
  winner: number;
  army: string;
  owner: string;
  mountain: string;
  castle: string;
  general: string;
  generalIdx: number[];
  alive: number[];
}

export function packBoard(b: Board): WireBoard {
  return {
    h: b.h,
    w: b.w,
    n: b.n,
    tick: b.tick,
    done: b.done,
    winner: b.winner,
    army: b64.enc(bytes(b.army)),
    owner: b64.enc(bytes(b.owner)),
    mountain: b64.enc(b.mountain),
    castle: b64.enc(b.castle),
    general: b64.enc(b.general),
    generalIdx: Array.from(b.generalIdx),
    alive: Array.from(b.alive),
  };
}

export function unpackBoard(p: WireBoard): Board {
  const u8 = (s: string) => b64.dec(s);
  const raw = (s: string) => {
    const u = u8(s);
    return u.buffer.slice(u.byteOffset, u.byteOffset + u.byteLength);
  };
  return {
    h: p.h,
    w: p.w,
    n: p.n,
    army: new Int32Array(raw(p.army)),
    owner: new Int8Array(raw(p.owner)),
    mountain: u8(p.mountain),
    castle: u8(p.castle),
    general: u8(p.general),
    generalIdx: Int32Array.from(p.generalIdx),
    alive: Uint8Array.from(p.alive),
    tick: p.tick,
    done: p.done,
    winner: p.winner,
  };
}

/** The shape `shared/board.mjs` renders. Plain arrays -- easy to read in devtools. */
export interface Snapshot {
  h: number;
  w: number;
  n: number;
  tick: number;
  done: boolean;
  winner: number;
  owner: number[];
  army: number[];
  mountain: number[];
  castle: number[];
  general: number[];
  alive: number[];
  land: number[];
  armyTotal: number[];
  lastMoves: (Move | null)[];
}

export function toSnapshot(b: Board, lastMoves: (Move | null)[]): Snapshot {
  const sb = score(b);
  return {
    h: b.h,
    w: b.w,
    n: b.n,
    tick: b.tick,
    done: b.done,
    winner: b.winner,
    owner: Array.from(b.owner),
    army: Array.from(b.army),
    mountain: Array.from(b.mountain),
    castle: Array.from(b.castle),
    general: Array.from(b.general),
    alive: Array.from(b.alive),
    land: sb.land,
    armyTotal: sb.army,
    lastMoves,
  };
}

/** The payload sent to a bot endpoint -- see `server/bot_protocol.py`. */
export function botStateOf(b: Board, cfg: GameConfig, player: number) {
  const sb = score(b);
  return {
    h: b.h,
    w: b.w,
    n: b.n,
    tick: b.tick,
    player,
    owner: Array.from(b.owner),
    army: Array.from(b.army),
    mountain: Array.from(b.mountain),
    castle: Array.from(b.castle),
    general: Array.from(b.general),
    generalIdx: Array.from(b.generalIdx),
    alive: Array.from(b.alive),
    land: sb.land,
    armyTotal: sb.army,
    config: cfg,
  };
}
