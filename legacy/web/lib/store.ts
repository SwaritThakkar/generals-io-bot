/**
 * Key-value storage with a per-match lock.
 *
 * Uses Upstash Redis when its environment variables are present (that is what
 * the Vercel Redis integration provisions), and falls back to an in-process map
 * so `npm run dev` works with no configuration. The fallback is single-instance
 * only -- fine locally, wrong for a deployment, which is why `isMisconfigured`
 * is surfaced and the lobby refuses to pretend otherwise.
 *
 * Round trips are the entire performance story here: from Vercel to Upstash each
 * command costs well over 100ms, so the read-modify-write cycle is expressed as
 * `lockedUpdate`, which pipelines lock+read into one request and write+unlock
 * into another. Three round trips instead of five, and reads that change nothing
 * take a single GET with no lock at all.
 */

import { Redis } from "@upstash/redis";

const url = process.env.KV_REST_API_URL || process.env.UPSTASH_REDIS_REST_URL;
const token = process.env.KV_REST_API_TOKEN || process.env.UPSTASH_REDIS_REST_TOKEN;

const redis = url && token ? new Redis({ url, token }) : null;

export const isPersistent = redis !== null;

/** True when deployed on Vercel with no Redis: matches will not survive a request. */
export const isMisconfigured = !isPersistent && process.env.VERCEL === "1";

const DEFAULT_TTL = 60 * 60 * 6;
const LOCK_MS = 5000;
const RELEASE =
  'if redis.call("get", KEYS[1]) == ARGV[1] then return redis.call("del", KEYS[1]) else return 0 end';

// ---- in-memory fallback ----------------------------------------------------
const mem = new Map<string, { value: string; expires: number }>();
const memLocks = new Map<string, Promise<void>>();

function memGet(key: string): string | null {
  const hit = mem.get(key);
  if (!hit) return null;
  if (hit.expires && hit.expires < Date.now()) {
    mem.delete(key);
    return null;
  }
  return hit.value;
}

async function memLock<T>(key: string, fn: () => Promise<T>): Promise<T> {
  const prev = memLocks.get(key) ?? Promise.resolve();
  let release!: () => void;
  const next = new Promise<void>((r) => (release = r));
  memLocks.set(key, prev.then(() => next));
  await prev;
  try {
    return await fn();
  } finally {
    release();
    if (memLocks.get(key) === next) memLocks.delete(key);
  }
}

// ---- plain access ----------------------------------------------------------
export async function get<T>(key: string): Promise<T | null> {
  if (redis) return (await redis.get<T>(key)) ?? null;
  const raw = memGet(key);
  return raw === null ? null : (JSON.parse(raw) as T);
}

export async function set(key: string, value: unknown, ttlSeconds = DEFAULT_TTL) {
  if (redis) {
    await redis.set(key, value, { ex: ttlSeconds });
    return;
  }
  mem.set(key, { value: JSON.stringify(value), expires: Date.now() + ttlSeconds * 1000 });
}

export async function del(key: string) {
  if (redis) await redis.del(key);
  else mem.delete(key);
}

// ---- locked read-modify-write ---------------------------------------------
export interface Update<T, R> {
  /** Persist this. Omit to leave the stored value untouched. */
  next?: T;
  result: R;
}

/**
 * Read `key` under an exclusive lock, hand it to `fn`, and write back whatever
 * `fn` returns as `next`.
 *
 * This is the only safe way to mutate a match: two clients polling the same
 * match must not both advance it, and a keypress must not be lost to a
 * concurrent tick.
 */
export async function lockedUpdate<T, R>(
  key: string,
  fn: (current: T | null) => Promise<Update<T, R>>,
  { ttlSeconds = DEFAULT_TTL, waitMs = 4000 } = {},
): Promise<R> {
  if (!redis) {
    return memLock(key, async () => {
      const current = await get<T>(key);
      const { next, result } = await fn(current);
      if (next !== undefined) await set(key, next, ttlSeconds);
      return result;
    });
  }

  const lockKey = `lock:${key}`;
  const token = Math.random().toString(36).slice(2);
  const deadline = Date.now() + waitMs;

  for (;;) {
    // One round trip: take the lock and read the value together. The read is
    // wasted when the lock is contended, which is far cheaper than a second
    // request when it is not.
    const [acquired, current] = (await redis
      .pipeline()
      .set(lockKey, token, { nx: true, px: LOCK_MS })
      .get<T>(key)
      .exec()) as [string | null, T | null];

    if (acquired) {
      try {
        const { next, result } = await fn(current ?? null);
        // One round trip: write and release.
        const p = redis.pipeline();
        if (next !== undefined) p.set(key, next, { ex: ttlSeconds });
        p.eval(RELEASE, [lockKey], [token]);
        await p.exec();
        return result;
      } catch (e) {
        await redis.eval(RELEASE, [lockKey], [token]).catch(() => {});
        throw e;
      }
    }

    if (Date.now() > deadline) throw new Error(`timed out waiting for lock on ${key}`);
    await new Promise((r) => setTimeout(r, 25 + Math.random() * 50));
  }
}
